"""Service for running anomaly detection."""

import asyncio
import json
import logging
from datetime import datetime, timezone

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from src.core.config import settings
from src.models.autoencoder import Autoencoder
from src.schemas.anomaly import (
    AnomalyDetectionResult,
    AnomalyFeatureImportanceResponse,
    AnomalyFeatureImportanceValue,
    AnomalyLocalExplanation,
    WindowScore,
)
from src.services.anomaly_config_service import AnomalyConfigService
from src.services.data_storage_client import DataStorageClient

logger = logging.getLogger(__name__)

try:
    from dlt_client import analytics_from_env, inference_from_env as _dlt_inference_from_env
    _DLT_AVAILABLE = True
except ImportError:
    _DLT_AVAILABLE = False
    logger.debug("dlt-client not installed; anomaly traceability disabled")

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


async def _dlt_trace_anomaly(
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
_background_cache: dict[tuple, np.ndarray] = {}


class AnomalyDetectionService:
    """Service for running anomaly detection on tag-filtered data."""

    def __init__(
        self,
        data_storage_client: DataStorageClient,
        anomaly_config_service: AnomalyConfigService,
    ):
        self.data_storage_client = data_storage_client
        self.anomaly_config_service = anomaly_config_service

    async def detect_for_tags(
        self, tags, model_id: str, lookback_seconds: int = 1800, explain: bool = False
    ) -> AnomalyDetectionResult:
        """Run anomaly detection using tag filters. tags can be Tags schema or plain dict."""
        import time

        from src.schemas.tags import Tags as TagsSchema
        tags_dict = tags.to_filter_dict() if hasattr(tags, "to_filter_dict") else tags
        tags_obj = tags if isinstance(tags, TagsSchema) else TagsSchema(**{k: v for k, v in tags_dict.items() if v is not None})

        if model_id is None:
            event_type = tags_dict.get("event")
            trained = [
                c for c in self.anomaly_config_service.list_all()
                if c.threshold_value is not None and (event_type is None or c.event_type == event_type)
            ]
            if not trained:
                raise ValueError(f"No trained anomaly models for event_type '{event_type}'")
            model_id = trained[0].model_id

        config_db = self.anomaly_config_service.get_config(model_id)
        if not config_db:
            raise ValueError(f"Anomaly model '{model_id}' not found")
        if config_db.threshold_value is None:
            raise ValueError(f"Anomaly model '{model_id}' has not been trained yet (no threshold)")

        config = self.anomaly_config_service.config_from_db(config_db)

        ae, scaler_mean, scaler_std, mlflow_run_id, model_version = await asyncio.to_thread(
            self._load_model_and_scaler, model_id, len(config.input_fields), config.hidden_size
        )

        X_background = None
        if explain:
            try:
                X_background = await asyncio.to_thread(self._load_background, model_id)
            except Exception as e:
                logger.warning("Could not load KernelSHAP background for model %s: %s", model_id, e)

        end_ts = int(time.time())
        start_ts = end_ts - lookback_seconds

        raw_data = await self.data_storage_client.fetch_data(
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            window_duration_seconds=config.window_duration_seconds,
            snssai_sst=tags_dict.get("snssai_sst"),
            dnn=tags_dict.get("dnn"),
            snssai_sd=tags_dict.get("snssai_sd"),
            event=tags_dict.get("event"),
        )

        empty = AnomalyDetectionResult(
            model_id=model_id,
            model_name=config_db.name,
            tags=tags_obj,
            threshold_value=config_db.threshold_value,
            window_duration_seconds=config.window_duration_seconds,
            input_fields=config.input_fields,
            num_windows=0,
            num_anomalies=0,
            scores=[],
        )

        if not raw_data:
            return empty

        raw_data.sort(key=lambda w: w.get("window_start_time", 0))

        features = []
        timestamps = []
        for w in raw_data:
            try:
                row = [float(w.get(f, 0.0)) for f in config.input_fields]
                features.append(row)
                raw_ts = w.get("window_start_time", 0)
                if isinstance(raw_ts, str):
                    raw_ts = int(datetime.fromisoformat(raw_ts).timestamp())
                timestamps.append(int(raw_ts))
            except (ValueError, TypeError):
                continue

        if not features:
            return empty

        X = np.nan_to_num(np.array(features, dtype=np.float32))
        X_scaled = ((X - scaler_mean) / scaler_std).astype(np.float32)
        errors = ae.score(X_scaled)

        scores = []
        num_anomalies = 0
        anomalous_indices = []
        for i, (ts, err) in enumerate(zip(timestamps, errors)):
            is_anomaly = float(err) > config_db.threshold_value
            if is_anomaly:
                num_anomalies += 1
                anomalous_indices.append(i)
            scores.append(WindowScore(
                window_start_time=ts,
                reconstruction_error=float(err),
                is_anomaly=is_anomaly,
            ))

        if explain and X_background is not None and anomalous_indices:
            try:
                explanations = await asyncio.to_thread(
                    self._run_shap_batch,
                    ae, X_background,
                    X_scaled[anomalous_indices],
                    [timestamps[i] for i in anomalous_indices],
                    [float(errors[i]) for i in anomalous_indices],
                    config.input_fields, model_id,
                )
                for idx, expl in zip(anomalous_indices, explanations):
                    scores[idx] = scores[idx].model_copy(update={"explanation": expl})
            except Exception as e:
                logger.warning("KernelSHAP batch failed for model %s: %s", model_id, e)

        result = AnomalyDetectionResult(
            model_id=model_id,
            model_name=config_db.name,
            tags=tags_obj,
            threshold_value=config_db.threshold_value,
            window_duration_seconds=config.window_duration_seconds,
            input_fields=config.input_fields,
            num_windows=len(scores),
            num_anomalies=num_anomalies,
            scores=scores,
        )

        if raw_data:
            max_score = max((s.reconstruction_error for s in scores), default=0.0)
            asyncio.create_task(_dlt_trace_anomaly(
                model_run_id=mlflow_run_id,
                model_name=config_db.name,
                model_version=model_version,
                query_params={**tags_dict, "lookback_seconds": lookback_seconds},
                data_payload=raw_data,
                anomaly_score=max_score,
                decision="ANOMALY" if num_anomalies > 0 else "NORMAL",
                time_range_start=str(start_ts),
                time_range_end=str(end_ts),
            ))

        return result

    def _load_model_and_scaler(
        self, model_id: str, num_features: int, hidden_size: int
    ) -> tuple[Autoencoder, np.ndarray, np.ndarray, str, str]:
        """Load a trained autoencoder and its scaler from MLflow.

        Returns (ae, scaler_mean, scaler_std, run_id, version).
        """
        client = MlflowClient()
        versions = client.get_latest_versions(model_id, stages=["None"])
        if not versions:
            raise ValueError(f"No trained version found for anomaly model '{model_id}'")

        latest = max(versions, key=lambda v: int(v.version))
        model_uri = f"models:/{model_id}/{latest.version}"
        loaded_net = mlflow.pytorch.load_model(model_uri)

        ae = Autoencoder(input_size=num_features, hidden_size=hidden_size)
        ae.model = loaded_net

        scaler_path = mlflow.artifacts.download_artifacts(
            run_id=latest.run_id, artifact_path="scaler/scaler.npz"
        )
        scaler_data = np.load(scaler_path)
        scaler_mean = scaler_data["mean"].astype(np.float32)
        scaler_std = scaler_data["std"].astype(np.float32)

        logger.info(f"Loaded anomaly model {model_id} v{latest.version} with scaler")
        return ae, scaler_mean, scaler_std, latest.run_id or "", str(latest.version)

    def _load_background(self, model_id: str) -> np.ndarray:
        """Load KernelSHAP background from MLflow artifact, with in-memory cache."""
        client = MlflowClient()
        versions = client.get_latest_versions(model_id, stages=["None"])
        if not versions:
            raise ValueError(f"No trained version found for anomaly model '{model_id}'")
        latest = max(versions, key=lambda v: int(v.version))
        cache_key = (model_id, latest.run_id)
        if cache_key in _background_cache:
            return _background_cache[cache_key]
        bg_path = mlflow.artifacts.download_artifacts(
            run_id=latest.run_id, artifact_path="background/background.npz"
        )
        X_background = np.load(bg_path)["X_background"].astype(np.float32)
        _background_cache[cache_key] = X_background
        logger.info("Loaded KernelSHAP background for model %s - %d samples", model_id, len(X_background))
        return X_background

    def _run_shap_batch(
        self,
        ae: Autoencoder,
        X_background: np.ndarray,
        X_anomalous: np.ndarray,
        timestamps: list[int],
        errors: list[float],
        input_fields: list[str],
        model_id: str,
        n_samples: int = 200,
    ) -> list[AnomalyLocalExplanation]:
        """Fit KernelSHAP once, explain all anomalous windows. Runs synchronously in thread pool."""
        from alibi.explainers import KernelShap

        def predict_fn(X: np.ndarray) -> np.ndarray:
            return ae.score(X)

        explainer = KernelShap(predict_fn, feature_names=input_fields)
        explainer.fit(X_background)

        results = []
        for x, ts, err in zip(X_anomalous, timestamps, errors):
            explanation = explainer.explain(x[np.newaxis], nsamples=n_samples)
            shap_values = explanation.data["shap_values"][0][0]
            expected_value = float(explanation.data["expected_value"][0])
            attributions = {
                name: round(float(v), 6)
                for name, v in zip(input_fields, shap_values)
            }
            logger.info(
                "KernelSHAP anomaly model=%s ts=%d baseline=%.4f attributions=%s",
                model_id, ts, expected_value,
                {k: v for k, v in sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)},
            )
            results.append(AnomalyLocalExplanation(
                model_id=model_id,
                window_start_time=ts,
                reconstruction_error=err,
                attributions=attributions,
                baseline=round(expected_value, 6),
                computed_at=datetime.now(timezone.utc),
            ))
        return results

    async def compute_permutation_importance(
        self, model_id: str, n_repeats: int = 2
    ) -> AnomalyFeatureImportanceResponse:
        """Compute permutation importance using training-time background artifact."""
        from alibi.explainers import PermutationImportance

        config_db = self.anomaly_config_service.get_config(model_id)
        if config_db is None or config_db.threshold_value is None:
            raise ValueError(f"Model '{model_id}' not found or not trained")

        ae, *_ = await asyncio.to_thread(
            self._load_model_and_scaler,
            model_id, len(config_db.input_fields), config_db.hidden_size
        )
        X_background = await asyncio.to_thread(self._load_background, model_id)

        def predict_fn(X: np.ndarray) -> np.ndarray:
            return ae.score(X)

        def scorer(y_true, y_pred):
            return -float(np.mean(y_pred))

        scorer.__name__ = "reconstruction_error"
        y_dummy = np.zeros(len(X_background))

        def _run_importance() -> object:
            explainer = PermutationImportance(
                predictor=predict_fn,
                score_fns=scorer,
                feature_names=config_db.input_fields,
            )
            return explainer.explain(X_background, y_dummy, n_repeats=n_repeats, kind="difference")

        explanation = await asyncio.to_thread(_run_importance)

        f_names = explanation.data["feature_names"]
        f_importance = explanation.data["feature_importance"][0]
        importances = {
            f_names[i]: AnomalyFeatureImportanceValue(
                mean=round(float(f_importance[i]["mean"]), 6),
                std=round(float(f_importance[i]["std"]), 6),
            )
            for i in range(len(f_names))
        }

        ranked = sorted(importances.items(), key=lambda x: x[1].mean, reverse=True)
        logger.info("Permutation importance for anomaly model '%s' (n_repeats=%d):", model_id, n_repeats)
        for name, val in ranked:
            logger.info("  %-30s mean=%+.6f  std=%.6f", name, val.mean, val.std)

        computed_at = datetime.now(timezone.utc)
        client = MlflowClient()
        client.set_registered_model_tag(
            model_id, "anomaly_importance",
            json.dumps({k: {"mean": v.mean, "std": v.std} for k, v in importances.items()})
        )
        client.set_registered_model_tag(model_id, "anomaly_importance_at", computed_at.isoformat())

        return AnomalyFeatureImportanceResponse(
            model_id=model_id, importances=importances, computed_at=computed_at
        )

    def get_cached_importance(self, model_id: str) -> AnomalyFeatureImportanceResponse | None:
        """Read cached permutation importance from MLflow tags."""
        try:
            client = MlflowClient()
            rm = client.get_registered_model(model_id)
            raw = rm.tags.get("anomaly_importance")
            at_str = rm.tags.get("anomaly_importance_at")
            if not raw:
                return None
            return AnomalyFeatureImportanceResponse(
                model_id=model_id,
                importances=json.loads(raw),
                computed_at=datetime.fromisoformat(at_str) if at_str else None,
            )
        except Exception as e:
            logger.warning("Failed to read cached importance for model %s: %s", model_id, e)
            return None
