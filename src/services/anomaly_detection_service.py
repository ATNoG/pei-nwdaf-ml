"""Service for running anomaly detection."""

import logging
from datetime import datetime

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from src.models.autoencoder import Autoencoder
from src.schemas.anomaly import (
    AnomalyDetectionResult,
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

    async def detect_for_tags(
        self, tags, model_id: str, lookback_seconds: int = 1800
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
        ae, scaler_mean, scaler_std = self._load_model_and_scaler(
            model_id, len(config.input_fields), config.hidden_size
        )

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
        for ts, err in zip(timestamps, errors):
            is_anomaly = float(err) > config_db.threshold_value
            if is_anomaly:
                num_anomalies += 1
            scores.append(WindowScore(
                window_start_time=ts,
                reconstruction_error=float(err),
                is_anomaly=is_anomaly,
            ))

        return AnomalyDetectionResult(
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
