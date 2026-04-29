"""Service for running inference on trained models."""

import asyncio
import logging
from datetime import datetime, timezone

import mlflow
import numpy as np

from src.core.config import settings
from src.models import MODEL_REGISTRY
from src.schemas.inference import ForecastStepPrediction
from src.schemas.model import ArchitectureType, ModelConfig
from src.schemas.performance import LocalExplanationResponse
from src.services.data_preparation import calculate_timestamps, prepare_last_sequence
from src.services.data_storage_client import DataStorageClient
from src.services.mlflow_service import MLflowService

logger = logging.getLogger(__name__)

try:
    from dlt_client import analytics_from_env, inference_from_env as _dlt_inference_from_env
    _DLT_AVAILABLE = True
except ImportError:
    _DLT_AVAILABLE = False
    logger.debug("dlt-client not installed; inference traceability disabled")

_dlt_analytics = None
_dlt_inference = None


def _get_dlt_clients():
    global _dlt_analytics, _dlt_inference
    if not _DLT_AVAILABLE or not settings.DLT_ENABLED:
        return None, None
    if _dlt_analytics is None:
        try:
            _dlt_analytics = analytics_from_env()
            _dlt_inference = _dlt_inference_from_env()
        except Exception as exc:
            logger.warning("DLT client init failed (non-fatal): %s", exc)
            return None, None
    return _dlt_analytics, _dlt_inference


async def _dlt_trace_inference(
    model_run_id, model_name, model_version,
    query_params, data_payload, anomaly_score, decision,
    time_range_start, time_range_end,
):
    analytics, inference = _get_dlt_clients()
    if analytics is None:
        return
    data_fetch_ref = ""
    try:
        data_fetch_ref = await analytics.record_data_fetch(
            mlflow_run_id=model_run_id or "",
            model_name=model_name,
            model_version=str(model_version),
            query_params=query_params,
            data_payload=data_payload,
            time_range_start=time_range_start,
            time_range_end=time_range_end,
        )
    except Exception as exc:
        logger.warning("DLT record_data_fetch failed (non-fatal): %s", exc)
    try:
        await inference.record_inference(
            mlflow_run_id=model_run_id or "",
            model_name=model_name,
            model_version=str(model_version),
            data_fetch_ref=data_fetch_ref,
            input_data=data_payload,
            anomaly_score=anomaly_score,
            decision=decision,
        )
    except Exception as exc:
        logger.warning("DLT record_inference failed (non-fatal): %s", exc)

# (model_id, run_id) -> X_background (invalidated automatically when model version changes)
_background_cache: dict[tuple[str, str], np.ndarray] = {}


class InferenceService:
    """Service for loading trained models and running predictions."""

    def __init__(
        self,
        mlflow_service: MLflowService,
        data_storage_client: DataStorageClient,
    ):
        self.mlflow_service = mlflow_service
        self.data_storage_client = data_storage_client

    async def predict(
        self, output_field: str, tags: dict, model_id: str | None = None, explain: bool = False
    ) -> dict:
        """Run inference using tag filters (snssai/dnn/event). explain=True adds KernelSHAP."""
        configs = self.mlflow_service.ml_config_service.list_all()
        event_type = tags.get("event", "") if isinstance(tags, dict) else ""

        # Tags use format: best_for:{event_type}:{field} = "true"
        def _is_best(model_id: str) -> bool:
            rm_tags = self.mlflow_service.client.get_registered_model(model_id).tags
            tag_key = f"best_for:{event_type}:{output_field}" if event_type else None
            if tag_key and rm_tags.get(tag_key) == "true":
                return True
            # Fallback: scan all best_for:*:{output_field} tags
            suffix = f":{output_field}"
            return any(v == "true" and k.endswith(suffix) for k, v in rm_tags.items())

        best_id = next((c.model_id for c in configs if _is_best(c.model_id)), None)

        if best_id is None:
            raise ValueError(
                f"No best model designated for field '{output_field}'. "
                f"Run POST /v1/performance/{output_field}/evaluate first."
            )

        if model_id is None:
            model_id = best_id
        else:
            config = next(
                (config for config in configs if config.model_id == model_id), None
            )
            if config is None or output_field not in config.output_fields:
                raise ValueError(
                    f"Model '{model_id}' does not predict field '{output_field}'."
                )

        # Load model detail (config + version info) - sync call, run in thread
        model_detail = await asyncio.to_thread(self.mlflow_service.get_model, model_id)
        config = model_detail.config

        # Validate model has been trained
        if model_detail.latest_version is None:
            raise ValueError(
                f"Model '{model_id}' has no trained versions. "
                f"Train the model first via POST /v1/training/train"
            )

        # Load trained model from MLflow - sync call, run in thread
        model = await asyncio.to_thread(
            self._load_trained_model,
            model_id=model_id,
            version=model_detail.latest_version,
            config=config,
        )

        lookback_seconds = config.lookback_steps * config.window_duration_seconds
        start_ts, end_ts = calculate_timestamps(lookback_seconds + config.window_duration_seconds)

        cell_data = await self.data_storage_client.fetch_data(
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            window_duration_seconds=config.window_duration_seconds,
            snssai_sst=tags.get("snssai_sst"),
            dnn=tags.get("dnn"),
            snssai_sd=tags.get("snssai_sd"),
            event=tags.get("event"),
        )

        if not cell_data:
            raise ValueError(f"No data for tags {tags} in last {lookback_seconds}s")

        # Sort by timestamp
        cell_data.sort(key=lambda x: x.get("window_start_time", 0))

        # Prepare input sequence
        X = prepare_last_sequence(
            cell_data=cell_data,
            input_fields=config.input_fields,
            lookback_steps=config.lookback_steps,
        )

        X = np.nan_to_num(X, nan=0.0)

        # Run prediction
        try:
            raw_predictions = model.predict(X)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

        asyncio.create_task(_dlt_trace_inference(
            model_run_id=model_detail.mlflow_run_id or "",
            model_name=model_detail.name or model_id,
            model_version=model_detail.latest_version,
            query_params={"cell_id": cell_id, "output_field": output_field, "lookback_seconds": lookback_seconds},
            data_payload=cell_data,
            anomaly_score=0.0,
            decision="NORMAL",
            time_range_start=str(start_ts),
            time_range_end=str(end_ts),
        ))

        # Calculate input data timestamps and window overlap
        from datetime import datetime
        used_data = cell_data[-config.lookback_steps :]

        # Get timestamps from used data
        first_window = used_data[0]
        last_window = used_data[-1]

        # Extract timestamps (handle both int and ISO string formats)
        def extract_timestamp(window, field_name):
            ts = window.get(field_name, 0)
            if isinstance(ts, str):
                return int(datetime.fromisoformat(ts).timestamp())
            return int(ts)

        input_data_start = extract_timestamp(first_window, "window_start_time")
        last_window_end = extract_timestamp(last_window, "window_end_time")
        if last_window_end == 0:
            # If window_end_time not available, calculate it
            last_window_end = extract_timestamp(last_window, "window_start_time") + config.window_duration_seconds

        # Calculate window overlap from first two windows
        window_overlap = 0
        if len(used_data) >= 2:
            first_start = extract_timestamp(used_data[0], "window_start_time")
            second_start = extract_timestamp(used_data[1], "window_start_time")
            step_size = second_start - first_start
            window_overlap = config.window_duration_seconds - step_size

        # Structure predictions with timestamps
        predictions = self._structure_predictions(
            raw_predictions=raw_predictions,
            output_fields=config.output_fields,
            forecast_steps=config.forecast_steps,
            last_window_end=last_window_end,
            window_duration_seconds=config.window_duration_seconds,
            window_overlap=window_overlap,
        )

        explanation = None
        if explain:
            try:
                explanation = await self._explain_kernelshap(
                    model=model,
                    config=config,
                    model_id=model_id,
                    version=model_detail.latest_version,
                    field_name=output_field,
                    tags=tags,
                    X_instance=X,
                )
            except Exception as e:
                logger.warning("KernelSHAP explanation failed tags=%s field='%s': %s", tags, output_field, e)

        return {
            "model_id": model_id,
            "model_name": model_detail.name,
            "model_version": model_detail.latest_version,
            "architecture": config.architecture,
            "tags": tags,
            "lookback_steps": config.lookback_steps,
            "forecast_steps": config.forecast_steps,
            "window_duration_seconds": config.window_duration_seconds,
            "window_overlap": window_overlap,
            "input_data_start": input_data_start,
            "input_data_end": last_window_end,
            "input_fields": config.input_fields,
            "output_fields": config.output_fields,
            "predictions": predictions,
            "used_data": used_data,
            "explanation": explanation,
        }

    async def predict_for_tags(
        self, output_field: str, tags: dict, model_id: str
    ) -> dict:
        """Run forecast using snssai/dnn tags (pipeline path, no cell_id).

        model_id must be the designated best model for this event_type:output_field.
        Returns {"model_info": {...}, "input": {...}, "output": {...}}.
        """
        from src.services.data_preparation import calculate_timestamps, prepare_last_sequence

        model_detail = await asyncio.to_thread(self.mlflow_service.get_model, model_id)
        config = model_detail.config

        if model_detail.latest_version is None:
            raise ValueError(f"Model '{model_id}' has no trained versions.")

        model = await asyncio.to_thread(
            self._load_trained_model,
            model_id=model_id,
            version=model_detail.latest_version,
            config=config,
        )

        lookback_seconds = config.lookback_steps * config.window_duration_seconds
        start_ts, end_ts = calculate_timestamps(lookback_seconds + config.window_duration_seconds)

        cell_data = await self.data_storage_client.fetch_data(
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            window_duration_seconds=config.window_duration_seconds,
            snssai_sst=tags.get("snssai_sst"),
            dnn=tags.get("dnn"),
            snssai_sd=tags.get("snssai_sd"),
            event=tags.get("event"),
        )

        if not cell_data:
            raise ValueError(f"No data for tags {tags} in last {lookback_seconds}s")

        cell_data.sort(key=lambda x: x.get("window_start_time", 0))

        X = prepare_last_sequence(
            cell_data=cell_data,
            input_fields=config.input_fields,
            lookback_steps=config.lookback_steps,
        )
        X = np.nan_to_num(X, nan=0.0)

        try:
            raw_predictions = model.predict(X)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

        def _ts(window, field):
            ts = window.get(field, 0)
            if isinstance(ts, str):
                from datetime import datetime as _dt
                return int(_dt.fromisoformat(ts).timestamp())
            return int(ts)

        used_data = cell_data[-config.lookback_steps:]
        input_start = _ts(used_data[0], "window_start_time")
        last_end = _ts(used_data[-1], "window_end_time") or (
            _ts(used_data[-1], "window_start_time") + config.window_duration_seconds
        )

        window_overlap = 0
        if len(used_data) >= 2:
            step = _ts(used_data[1], "window_start_time") - _ts(used_data[0], "window_start_time")
            window_overlap = config.window_duration_seconds - step

        predictions = self._structure_predictions(
            raw_predictions=raw_predictions,
            output_fields=config.output_fields,
            forecast_steps=config.forecast_steps,
            last_window_end=last_end,
            window_duration_seconds=config.window_duration_seconds,
            window_overlap=window_overlap,
        )

        return {
            "model_info": {
                "id": model_id,
                "name": model_detail.name,
                "version": model_detail.latest_version,
                "architecture": config.architecture,
            },
            "input": {
                "fields": config.input_fields,
                "start": input_start,
                "end": last_end,
                "windows": len(used_data),
            },
            "output": {
                "fields": config.output_fields,
                "predictions": [
                    p.model_dump() if hasattr(p, "model_dump") else p
                    for p in predictions
                ],
            },
        }

    def _load_background(self, model_id: str, run_id: str) -> np.ndarray:
        """Load KernelSHAP background from MLflow artifact, with in-memory cache."""
        cache_key = (model_id, run_id)
        if cache_key in _background_cache:
            return _background_cache[cache_key]

        artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="background/background.npz")
        data = np.load(artifact_path)
        X_background = data["X_background"].astype(np.float32)
        _background_cache[cache_key] = X_background
        logger.info(
            "Loaded KernelSHAP background for model %s run %s - %d cells cached",
            model_id, run_id, len(X_background),
        )
        return X_background

    async def _explain_kernelshap(
        self,
        model,
        config: ModelConfig,
        model_id: str,
        version: int,
        field_name: str,
        tags: dict,
        X_instance: np.ndarray,
        n_samples: int = 200,
    ) -> LocalExplanationResponse:
        """Explain a single prediction using KernelSHAP with training-time background."""
        from alibi.explainers import KernelShap
        from src.services.performance_service import _SequenceIndexBridge

        client = mlflow.tracking.MlflowClient()
        mv = client.get_model_version(model_id, str(version))
        run_id = mv.run_id

        X_background = self._load_background(model_id, run_id)

        field_idx = config.output_fields.index(field_name)
        num_out = len(config.output_fields)
        n_fields = len(config.input_fields)

        # Pool: target instance at index 0, training background at indices 1..N
        all_sequences = np.concatenate([X_instance, X_background], axis=0)
        bridge = _SequenceIndexBridge(all_sequences)

        def predict_fn(X_2d: np.ndarray) -> np.ndarray:
            X_3d = bridge.reconstruct(X_2d)
            preds = model.predict(X_3d)
            return preds[:, field_idx::num_out].mean(axis=1)

        x_instance_2d = np.array([[0.0] * n_fields])       # index 0 = target
        background_2d = bridge.encode(n_fields)[1:]         # indices 1..N = training cells

        explainer = KernelShap(predict_fn, feature_names=config.input_fields)
        explainer.fit(background_2d)
        explanation = explainer.explain(x_instance_2d, nsamples=n_samples)

        shap_values = explanation.data["shap_values"][0][0]
        expected_value = float(explanation.data["expected_value"][0])

        attributions = {
            name: round(float(v), 6)
            for name, v in zip(config.input_fields, shap_values)
        }
        pred_for_field = model.predict(X_instance)[0][field_idx::num_out].tolist()

        logger.info(
            "KernelSHAP tags=%s field='%s' baseline=%.4f attributions=%s",
            tags, field_name, expected_value,
            {k: v for k, v in sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)},
        )

        return LocalExplanationResponse(
            field_name=field_name,
            model_id=model_id,
            method="kernelshap",
            tags=tags,
            prediction=pred_for_field,
            attributions=attributions,
            baseline=round(expected_value, 6),
            computed_at=datetime.now(timezone.utc),
        )

    def _load_trained_model(
        self,
        model_id: str,
        version: int,
        config: ModelConfig,
    ):
        """
        Load a trained model from the MLflow model registry.

        Args:
            model_id: Model ID (MLflow registered model name).
            version: Model version to load.
            config: Model configuration.

        Returns:
            ModelInterface instance with loaded weights.

        Raises:
            ValueError: If architecture is unsupported or model cannot be loaded.
        """
        model_class = MODEL_REGISTRY.get(config.architecture)
        if not model_class:
            raise ValueError(f"Unsupported architecture: {config.architecture}")

        model_uri = f"models:/{model_id}/{version}"
        logger.info(f"Loading model from {model_uri}")

        try:
            loaded_pytorch_model = mlflow.pytorch.load_model(model_uri)
        except Exception as e:
            raise ValueError(
                f"Failed to load model artifact from {model_uri}: {str(e)}"
            )

        # Create wrapper with config
        if config.architecture == ArchitectureType.LSTM:
            model = model_class(
                input_fields=config.input_fields,
                output_fields=config.output_fields,
                window_duration_seconds=config.window_duration_seconds,
                lookback_steps=config.lookback_steps,
                forecast_steps=config.forecast_steps,
                hidden_size=config.hidden_size,
                num_layers=2,
            )
        else:
            model = model_class(
                input_fields=config.input_fields,
                output_fields=config.output_fields,
                window_duration_seconds=config.window_duration_seconds,
                lookback_steps=config.lookback_steps,
                forecast_steps=config.forecast_steps,
                hidden_size=config.hidden_size,
            )

        model.model = loaded_pytorch_model
        return model

    def _structure_predictions(
        self,
        raw_predictions: np.ndarray,
        output_fields: list[str],
        forecast_steps: int,
        last_window_end: int,
        window_duration_seconds: int,
        window_overlap: int = 0,
    ) -> list[ForecastStepPrediction]:
        """
        Convert raw model output into structured predictions.

        The model returns shape (1, forecast_steps * num_output_fields).
        This reshapes it into a list of ForecastStepPrediction objects.

        Args:
            raw_predictions: Raw model output, shape (1, forecast_steps * num_outputs).
            output_fields: List of output field names.
            forecast_steps: Number of forecast steps.
            last_window_end: End timestamp of last input window (epoch seconds).
            window_duration_seconds: Duration of each window in seconds.
            window_overlap: Overlap between consecutive windows in seconds (0 for tumbling).

        Returns:
            List of ForecastStepPrediction with field-value mappings per step.
        """
        flat = raw_predictions[0]
        num_outputs = len(output_fields)
        expected_length = forecast_steps * num_outputs

        if len(flat) != expected_length:
            raise RuntimeError(
                f"Model output shape mismatch: got {len(flat)} values, "
                f"expected {expected_length} (forecast_steps={forecast_steps} "
                f"* output_fields={num_outputs})"
            )

        # Calculate step size (distance between window starts)
        step_size = window_duration_seconds - window_overlap

        predictions = []
        for step_idx in range(forecast_steps):
            start = step_idx * num_outputs
            end = start + num_outputs
            step_values = flat[start:end]

            values_dict = {
                field: round(float(step_values[i]), 6)
                for i, field in enumerate(output_fields)
            }

            # Calculate window timestamps for this prediction step
            # For tumbling: step_size = window_duration (overlap = 0)
            # For sliding: step_size = window_duration - overlap
            window_start = last_window_end + (step_idx * step_size)
            window_end = window_start + window_duration_seconds

            predictions.append(
                ForecastStepPrediction(
                    step=step_idx + 1,
                    window_start_time=window_start,
                    window_end_time=window_end,
                    values=values_dict
                )
            )

        return predictions
