"""Service for running anomaly detection."""

import logging
from collections import defaultdict
from datetime import datetime

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from src.models.autoencoder import Autoencoder
from src.schemas.anomaly import (
    AnomalyDetectionResult,
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

    async def detect(self, cell_id: int, model_id: str) -> AnomalyDetectionResult:
        """
        Run anomaly detection for all IPs in a cell.

        Args:
            cell_id: Cell index to analyse
            model_id: Anomaly model ID

        Returns:
            AnomalyDetectionResult with per-IP scores
        """
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

        # Fetch data for cell with all IPs
        import time

        end_ts = int(time.time())
        # Use a recent window — fetch the last hour of data
        start_ts = end_ts - 3600

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

        return AnomalyDetectionResult(
            model_id=model_id,
            model_name=config_db.name,
            cell_id=cell_id,
            threshold_value=config_db.threshold_value,
            window_duration_seconds=config.window_duration_seconds,
            input_fields=config.input_fields,
            results=ip_results,
        )

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
