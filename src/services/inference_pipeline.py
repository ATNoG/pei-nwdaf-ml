"""Kafka-driven inference pipeline.

Subscribes to processed network data and runs anomaly detection + forecasting
for every unique tag combination (snssai/dnn/event) that receives new data.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone

from src.core.config import settings
from src.schemas.anomaly import AnomalyModelMeta
from src.services.anomaly_detection_service import AnomalyDetectionService
from src.services.inference_service import InferenceService

logger = logging.getLogger(__name__)


class InferencePipeline:
    _MAX_CONCURRENT = 2

    def __init__(
        self,
        bridge,
        anomaly_detection_service: AnomalyDetectionService,
        inference_service: InferenceService,
    ):
        self.bridge = bridge
        self.anomaly_detection_service = anomaly_detection_service
        self.inference_service = inference_service
        self._last_run: dict[tuple, float] = {}
        self._semaphore = asyncio.Semaphore(self._MAX_CONCURRENT)

    def on_message(self, data: dict) -> dict:
        """Kafka bind callback — schedules async processing per tag combo."""
        try:
            content = json.loads(data["content"])
        except (KeyError, json.JSONDecodeError) as e:
            logger.warning("Inference pipeline: bad message: %s", e)
            return data

        tags = content.get("tags")
        if not tags:
            return data

        tag_key = tuple(sorted(tags.items()))
        now = time.monotonic()
        if now - self._last_run.get(tag_key, 0) < settings.KAFKA_DEBOUNCE_SECONDS:
            return data
        self._last_run[tag_key] = now

        loop = asyncio.get_running_loop()
        task = loop.create_task(self._process(tags))
        task.add_done_callback(
            lambda t: (
                logger.error("_process failed: %s", t.exception())
                if t.exception()
                else None
            )
        )
        return data

    async def _process(self, tags: dict):
        async with self._semaphore:
            results: list[dict] = []
            event_type = tags.get("event", "")

            # Anomaly: all trained models matching this event_type
            trained_models = [
                cfg
                for cfg in self.anomaly_detection_service.anomaly_config_service.list_all()
                if cfg.threshold_value is not None and cfg.event_type == event_type
            ]
            models_meta: dict[str, AnomalyModelMeta] = {}
            anomaly_counts: dict[str, str] = {}
            for model_cfg in trained_models:
                try:
                    anomaly = await self.anomaly_detection_service.detect_for_tags(
                        tags=tags, model_id=model_cfg.model_id, lookback_seconds=120
                    )
                    models_meta[anomaly.model_id] = AnomalyModelMeta(
                        name=anomaly.model_name,
                        fields=anomaly.input_fields,
                        threshold=anomaly.threshold_value,
                        window_duration_seconds=anomaly.window_duration_seconds,
                    )
                    anomaly_counts[anomaly.model_id] = (
                        f"{anomaly.num_anomalies}/{anomaly.num_windows}"
                    )
                except Exception as e:
                    logger.warning(
                        "Anomaly skipped for model %s: %s", model_cfg.name, e
                    )

            if models_meta:
                results.append(
                    {
                        "type": "anomaly",
                        "result": {
                            "models": {
                                mid: m.model_dump() for mid, m in models_meta.items()
                            },
                            "anomaly_counts": anomaly_counts,
                        },
                    }
                )

            # Forecast: best model per (event_type, output_field)
            for model_id, field_name in self._get_forecastable_fields(event_type):
                try:
                    forecast = await self.inference_service.predict_for_tags(
                        output_field=field_name, tags=tags, model_id=model_id
                    )
                    results.append({"type": "forecast", "result": forecast})
                except Exception as e:
                    logger.warning(
                        "Forecast %s skipped [%s]: %s", field_name, type(e).__name__, e
                    )

            if results:
                self._publish(tags, results)

    def _get_forecastable_fields(self, event_type: str) -> list[tuple[str, str]]:
        """Return (model_id, field_name) pairs designated best for event_type:field."""
        result: list[tuple[str, str]] = []
        prefix = f"best_for:{event_type}:"
        try:
            for (
                config
            ) in self.inference_service.mlflow_service.ml_config_service.list_all():
                rm = self.inference_service.mlflow_service.client.get_registered_model(
                    config.model_id
                )
                for tag_key, tag_val in rm.tags.items():
                    if tag_key.startswith(prefix) and tag_val == "true":
                        field = tag_key.removeprefix(prefix)
                        if (config.model_id, field) not in result:
                            result.append((config.model_id, field))
        except Exception as e:
            logger.warning("Failed to discover forecastable fields: %s", e)
        return result

    def _publish(self, tags: dict, results: list[dict]):
        tmp = {
            "tags": tags,
            "results": results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        msg = json.dumps(tmp)
        self.bridge.produce(settings.KAFKA_OUTPUT_TOPIC, msg)
        logger.info(
            "Published %d result(s) for tags %s to %s",
            len(results),
            tags,
            settings.KAFKA_OUTPUT_TOPIC,
        )


def setup_inference_pipeline():
    """Start the Kafka inference pipeline in a daemon thread."""
    from threading import Thread

    def _worker():
        try:
            asyncio.run(_start_pipeline())
        except Exception as e:
            logger.error("Kafka inference pipeline crashed: %s", e)

    thread = Thread(target=_worker, daemon=True, name="kafka-inference-thread")
    thread.start()
    logger.info("Kafka inference pipeline thread started")


async def _start_pipeline():
    from mlflow.tracking import MlflowClient
    from utils.kmw import PyKafBridge

    from src.db.database import SessionLocal
    from src.services.anomaly_config_service import AnomalyConfigService
    from src.services.config_service import MLConfigService
    from src.services.data_storage_client import DataStorageClient
    from src.services.mlflow_service import MLflowService

    bridge = PyKafBridge(
        settings.KAFKA_INPUT_TOPIC,
        hostname=settings.KAFKA_HOST,
        port=settings.KAFKA_PORT,
    )

    db = SessionLocal()
    data_storage_client = DataStorageClient()
    anomaly_config_service = AnomalyConfigService(db)
    anomaly_detection_service = AnomalyDetectionService(
        data_storage_client, anomaly_config_service
    )

    ml_config_service = MLConfigService(db)
    mlflow_service = MLflowService(MlflowClient(), ml_config_service)
    inference_service = InferenceService(mlflow_service, data_storage_client)

    pipeline = InferencePipeline(bridge, anomaly_detection_service, inference_service)
    bridge.bind_topic(settings.KAFKA_INPUT_TOPIC, pipeline.on_message)
    await bridge.start_consumer()

    if bridge._consumer_task:
        await bridge._consumer_task
