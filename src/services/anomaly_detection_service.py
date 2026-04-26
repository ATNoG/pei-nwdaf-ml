"""Service for running anomaly detection."""

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from src.models.autoencoder import Autoencoder
from src.schemas.anomaly import (
    AnomalyDetectionResult,
    AnomalyFeatureImportanceResponse,
    AnomalyFeatureImportanceValue,
    AnomalyLocalExplanation,
    IPAnomalyResult,
    WindowScore,
)
from src.services.anomaly_config_service import AnomalyConfigService
from src.services.data_storage_client import DataStorageClient

logger = logging.getLogger(__name__)


class AnomalyDetectionService:
    """Service for running per-IP anomaly detection on a cell."""

    def __init__(
        self,
        data_storage_client: DataStorageClient,
        anomaly_config_service: AnomalyConfigService,
    ):
        self.data_storage_client = data_storage_client
        self.anomaly_config_service = anomaly_config_service

    async def detect(
        self, cell_id: int, model_id: str | None = None, lookback_seconds: int = 1800, explain: bool = False
    ) -> AnomalyDetectionResult:
        """
        Run anomaly detection for all IPs in a cell.

        Args:
            cell_id: Cell index to analyse
            model_id: Anomaly model ID (if None, auto-select best model)
            lookback_seconds: How far back to fetch data

        Returns:
            AnomalyDetectionResult with per-IP scores
        """
        if model_id is None:
            model_id = await self._select_best_model(cell_id, lookback_seconds)

        # Load config
        config_db = self.anomaly_config_service.get_config(model_id)
        if not config_db:
            raise ValueError(f"Anomaly model '{model_id}' not found")
        if config_db.threshold_value is None:
            raise ValueError(
                f"Anomaly model '{model_id}' has not been trained yet (no threshold)"
            )

        config = self.anomaly_config_service.config_from_db(config_db)

        # Load trained model and scaler from MLflow
        ae, scaler_mean, scaler_std = self._load_model_and_scaler(
            model_id, len(config.input_fields), config.hidden_size
        )

        # Load background once if explain requested
        X_background = None
        if explain:
            try:
                X_background = self._load_background(model_id)
            except Exception as e:
                logger.warning("Could not load KernelSHAP background for model %s: %s", model_id, e)

        # Fetch data for cell with all IPs
        import time

        end_ts = int(time.time())
        start_ts = end_ts - lookback_seconds

        raw_data = await self.data_storage_client.fetch_cell_data(
            cell_index=cell_id,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            window_duration_seconds=config.window_duration_seconds,
            ip_src="*",
        )

        if not raw_data:
            # Return empty result
            return AnomalyDetectionResult(
                model_id=model_id,
                model_name=config_db.name,
                cell_id=cell_id,
                threshold_value=config_db.threshold_value,
                window_duration_seconds=config.window_duration_seconds,
                input_fields=config.input_fields,
                results=[],
            )

        # Group by ip_src
        ip_groups: dict[str, list[dict]] = defaultdict(list)
        for record in raw_data:
            ip = record.get("ip_src", "unknown")
            ip_groups[ip].append(record)

        # Score each IP
        ip_results: list[IPAnomalyResult] = []
        for ip_src, windows in ip_groups.items():
            windows.sort(key=lambda w: w.get("window_start_time", 0))

            # Extract features
            features = []
            timestamps = []
            for w in windows:
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
                continue

            X = np.nan_to_num(np.array(features, dtype=np.float32))
            X_scaled = ((X - scaler_mean) / scaler_std).astype(np.float32)
            errors = ae.score(X_scaled)

            scores = []
            num_anomalies = 0
            for i, (ts, err) in enumerate(zip(timestamps, errors)):
                is_anomaly = float(err) > config_db.threshold_value
                if is_anomaly:
                    num_anomalies += 1
                expl = None
                if is_anomaly and explain and X_background is not None:
                    try:
                        expl = await self._explain_kernelshap(
                            ae=ae,
                            X_background=X_background,
                            x_instance=X_scaled[i],
                            input_fields=config.input_fields,
                            model_id=model_id,
                            cell_id=cell_id,
                            ip_src=ip_src,
                            window_start_time=ts,
                            reconstruction_error=float(err),
                        )
                    except Exception as e:
                        logger.warning("KernelSHAP failed for ts=%d ip=%s: %s", ts, ip_src, e)
                scores.append(
                    WindowScore(
                        window_start_time=ts,
                        reconstruction_error=float(err),
                        is_anomaly=is_anomaly,
                        explanation=expl,
                    )
                )

            ip_results.append(
                IPAnomalyResult(
                    ip_src=ip_src,
                    num_windows=len(scores),
                    num_anomalies=num_anomalies,
                    scores=scores,
                )
            )

        return AnomalyDetectionResult(
            model_id=model_id,
            model_name=config_db.name,
            cell_id=cell_id,
            threshold_value=config_db.threshold_value,
            window_duration_seconds=config.window_duration_seconds,
            input_fields=config.input_fields,
            results=ip_results,
        )

    async def _select_best_model(self, cell_id: int, lookback_seconds: int) -> str:
        """Select the trained model with lowest training loss whose input_fields are available."""
        import time

        trained = [
            cfg
            for cfg in self.anomaly_config_service.list_all()
            if cfg.threshold_value is not None
        ]
        if not trained:
            raise ValueError("No trained anomaly models available")

        # Fetch one record to discover which fields the cell has
        end_ts = int(time.time())
        start_ts = end_ts - lookback_seconds

        sample = await self.data_storage_client.fetch_cell_data(
            cell_index=cell_id,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            window_duration_seconds=trained[0].window_duration_seconds,
            ip_src="*",
        )
        if not sample:
            raise ValueError(f"No data available for cell {cell_id}")

        logger.info(
            f"Auto-select: fetched {len(sample)} sample records for cell {cell_id}"
        )

        available_fields: set[str] = set()
        for record in sample:
            available_fields.update(record.keys())

        logger.info(
            f"Auto-select: available fields in cell {cell_id}: {sorted(available_fields)}"
        )

        # Filter models whose input_fields are all present
        compatible = []
        for cfg in trained:
            model_fields = set(cfg.input_fields)
            missing = model_fields - available_fields
            if missing:
                logger.info(
                    f"Auto-select: model {cfg.model_id} ({cfg.name}) incompatible - "
                    f"missing fields: {sorted(missing)}"
                )
            else:
                logger.info(
                    f"Auto-select: model {cfg.model_id} ({cfg.name}) is compatible"
                )
                compatible.append(cfg)

        if not compatible:
            raise ValueError(
                "No trained anomaly model has input_fields matching this cell's data"
            )

        # Pick the one with lowest training loss from MLflow
        client = MlflowClient()
        best_model_id: str | None = None
        best_loss = float("inf")

        for cfg in compatible:
            try:
                rm = client.get_registered_model(cfg.model_id)
                if not rm.latest_versions:
                    continue
                mv = rm.latest_versions[0]
                if not mv.run_id:
                    continue
                run = client.get_run(mv.run_id)
                loss = run.data.metrics.get("final_loss")
                if loss is not None and loss < best_loss:
                    best_loss = loss
                    best_model_id = cfg.model_id
            except Exception as e:
                logger.warning(f"Skipping model {cfg.model_id}: {e}")

        if best_model_id is None:
            raise ValueError("No compatible anomaly model has a recorded training loss")

        logger.info(
            f"Auto-selected model {best_model_id} (training_loss={best_loss:.6f})"
        )
        return best_model_id

    def _load_model_and_scaler(
        self, model_id: str, num_features: int, hidden_size: int
    ) -> tuple[Autoencoder, np.ndarray, np.ndarray]:
        """Load a trained autoencoder and its scaler from MLflow."""
        client = MlflowClient()
        versions = client.get_latest_versions(model_id, stages=["None"])
        if not versions:
            raise ValueError(f"No trained version found for anomaly model '{model_id}'")

        latest = max(versions, key=lambda v: int(v.version))
        model_uri = f"models:/{model_id}/{latest.version}"
        loaded_net = mlflow.pytorch.load_model(model_uri)

        ae = Autoencoder(input_size=num_features, hidden_size=hidden_size)
        ae.model = loaded_net

        # Load scaler
        scaler_path = mlflow.artifacts.download_artifacts(
            run_id=latest.run_id, artifact_path="scaler/scaler.npz"
        )
        scaler_data = np.load(scaler_path)
        scaler_mean = scaler_data["mean"].astype(np.float32)
        scaler_std = scaler_data["std"].astype(np.float32)

        logger.info(f"Loaded anomaly model {model_id} v{latest.version} with scaler")
        return ae, scaler_mean, scaler_std

    def _load_background(self, model_id: str) -> np.ndarray:
        """Load training background artifact from the latest model version."""
        client = MlflowClient()
        versions = client.get_latest_versions(model_id, stages=["None"])
        if not versions:
            raise ValueError(f"No trained version found for anomaly model '{model_id}'")
        latest = max(versions, key=lambda v: int(v.version))
        bg_path = mlflow.artifacts.download_artifacts(
            run_id=latest.run_id, artifact_path="background/background.npz"
        )
        return np.load(bg_path)["X_background"].astype(np.float32)

    async def _explain_kernelshap(
        self,
        ae: Autoencoder,
        X_background: np.ndarray,
        x_instance: np.ndarray,
        input_fields: list[str],
        model_id: str,
        cell_id: int,
        ip_src: str,
        window_start_time: int,
        reconstruction_error: float,
        n_samples: int = 200,
    ) -> AnomalyLocalExplanation:
        """Compute KernelSHAP local explanation for a single anomalous window."""
        from alibi.explainers import KernelShap

        def predict_fn(X: np.ndarray) -> np.ndarray:
            return ae.score(X)

        explainer = KernelShap(predict_fn, feature_names=input_fields)
        explainer.fit(X_background)
        explanation = explainer.explain(x_instance[np.newaxis], nsamples=n_samples)

        shap_values = explanation.data["shap_values"][0][0]
        expected_value = float(explanation.data["expected_value"][0])

        attributions = {
            name: round(float(v), 6)
            for name, v in zip(input_fields, shap_values)
        }

        logger.info(
            "KernelSHAP anomaly cell=%d ip=%s ts=%d baseline=%.4f attributions=%s",
            cell_id, ip_src, window_start_time, expected_value,
            {k: v for k, v in sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)},
        )

        return AnomalyLocalExplanation(
            model_id=model_id,
            cell_id=cell_id,
            ip_src=ip_src,
            window_start_time=window_start_time,
            reconstruction_error=reconstruction_error,
            attributions=attributions,
            baseline=round(expected_value, 6),
            computed_at=datetime.now(timezone.utc),
        )

    async def compute_permutation_importance(
        self, model_id: str, n_repeats: int = 2
    ) -> AnomalyFeatureImportanceResponse:
        """Compute permutation importance using training-time background artifact."""
        import asyncio
        from alibi.explainers import PermutationImportance

        config_db = self.anomaly_config_service.get_config(model_id)
        if config_db is None or config_db.threshold_value is None:
            raise ValueError(f"Model '{model_id}' not found or not trained")

        ae, _, _ = await asyncio.to_thread(
            self._load_model_and_scaler,
            model_id, len(config_db.input_fields), config_db.hidden_size
        )
        X_background = await asyncio.to_thread(self._load_background, model_id)

        def predict_fn(X: np.ndarray) -> np.ndarray:
            return ae.score(X)

        # alibi: higher = better → negate reconstruction error so that
        # importance = baseline_score - permuted_score > 0 when feature is important
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
        except Exception:
            return None
