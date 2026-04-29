"""Service for running anomaly detection."""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from src.core.config import settings
from src.models.autoencoder import Autoencoder
from src.schemas.anomaly import (
    AnomalyDetectionResult,
    IPAnomalyResult,
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
        self, cell_id: int, model_id: str | None = None, lookback_seconds: int = 1800
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
        ae, scaler_mean, scaler_std, mlflow_run_id, model_version = self._load_model_and_scaler(
            model_id, len(config.input_fields), config.hidden_size
        )

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
                scores.append(
                    WindowScore(
                        window_start_time=ts,
                        reconstruction_error=float(err),
                        is_anomaly=is_anomaly,
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

        result = AnomalyDetectionResult(
            model_id=model_id,
            model_name=config_db.name,
            cell_id=cell_id,
            threshold_value=config_db.threshold_value,
            window_duration_seconds=config.window_duration_seconds,
            input_fields=config.input_fields,
            results=ip_results,
        )

        if raw_data:
            max_score = max(
                (s.reconstruction_error for r in ip_results for s in r.scores),
                default=0.0,
            )
            any_anomaly = any(s.is_anomaly for r in ip_results for s in r.scores)
            asyncio.create_task(_dlt_trace_anomaly(
                model_run_id=mlflow_run_id,
                model_name=config_db.name,
                model_version=model_version,
                query_params={"cell_id": cell_id, "lookback_seconds": lookback_seconds},
                data_payload=raw_data,
                anomaly_score=max_score,
                decision="ANOMALY" if any_anomaly else "NORMAL",
                time_range_start=str(start_ts),
                time_range_end=str(end_ts),
            ))

        return result

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

        # Load scaler
        scaler_path = mlflow.artifacts.download_artifacts(
            run_id=latest.run_id, artifact_path="scaler/scaler.npz"
        )
        scaler_data = np.load(scaler_path)
        scaler_mean = scaler_data["mean"].astype(np.float32)
        scaler_std = scaler_data["std"].astype(np.float32)

        logger.info(f"Loaded anomaly model {model_id} v{latest.version} with scaler")
        return ae, scaler_mean, scaler_std, latest.run_id or "", str(latest.version)
