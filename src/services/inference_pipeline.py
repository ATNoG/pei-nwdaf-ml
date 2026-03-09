"""Kafka-driven inference pipeline.

Subscribes to processed network data and runs anomaly detection
for every cell that receives new data.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone

from src.core.config import settings
from src.schemas.anomaly import AnomalyDetectionResult
from src.services.anomaly_detection_service import AnomalyDetectionService
from src.services.inference_service import InferenceService

logger = logging.getLogger(__name__)


class InferencePipeline:
    # Max concurrent inference calls to data-storage
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
        self._last_run: dict[int, float] = {}
        self._semaphore = asyncio.Semaphore(self._MAX_CONCURRENT)

    def on_message(self, data: dict) -> dict:
        """Kafka bind callback — schedules async processing."""
        try:
            content = json.loads(data["content"])
        except (KeyError, json.JSONDecodeError) as e:
            logger.warning("Inference pipeline: bad message: %s", e)
            return data

        cell_id = content.get("cell_index") or content.get("cell_id")
        if cell_id is None:
            return data

        cell_id = int(cell_id)
        now = time.monotonic()
        if now - self._last_run.get(cell_id, 0) < settings.KAFKA_DEBOUNCE_SECONDS:
            return data
        self._last_run[cell_id] = now

        loop = asyncio.get_event_loop()
        loop.create_task(self._process(cell_id))
        return data

    async def _process(self, cell_id: int):
        async with self._semaphore:
            results: list[dict] = []

            try:
                anomaly = await self.anomaly_detection_service.detect(cell_id)
                results.append(
                    {"type": "anomaly", "result": _serialize_anomaly(anomaly)}
                )
            except Exception as e:
                logger.warning("Anomaly detection skipped for cell %s: %s", cell_id, e)

            for output_field in self._get_forecastable_fields():
                try:
                    forecast = await self.inference_service.predict(
                        output_field=output_field, cell_id=cell_id
                    )
                    results.append({"type": "forecast", "result": _serialize_forecast(forecast)})
                except Exception as e:
                    logger.debug(
                        "Forecast %s skipped for cell %s: %s", output_field, cell_id, e
                    )

            if results:
                self._publish(cell_id, results)

    def _get_forecastable_fields(self) -> list[str]:
        """Get output_fields that have a best model designated in MLflow."""
        fields: list[str] = []
        try:
            for (
                config
            ) in self.inference_service.mlflow_service.ml_config_service.list_all():
                rm = self.inference_service.mlflow_service.client.get_registered_model(
                    config.model_id
                )
                for tag_key, tag_val in rm.tags.items():
                    if tag_key.startswith("best_for:") and tag_val == "true":
                        field = tag_key.removeprefix("best_for:")
                        if field not in fields:
                            fields.append(field)
        except Exception as e:
            logger.warning("Failed to discover forecastable fields: %s", e)
        return fields

    def _publish(self, cell_id: int, results: list[dict]):
        msg = json.dumps(
            {
                "cell_id": cell_id,
                "results": results,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.bridge.produce(settings.KAFKA_OUTPUT_TOPIC, msg)
        logger.info(
            "Published %d result(s) for cell %s to %s",
            len(results),
            cell_id,
            settings.KAFKA_OUTPUT_TOPIC,
        )


def _serialize_anomaly(result: AnomalyDetectionResult) -> dict:
    """Convert AnomalyDetectionResult to a JSON-safe dict."""
    return result.model_dump()


def _serialize_forecast(result: dict) -> dict:
    """Convert forecast result dict (with Pydantic objects) to JSON-safe dict."""
    result = result.copy()
    if "predictions" in result:
        result["predictions"] = [
            p.model_dump() if hasattr(p, "model_dump") else p
            for p in result["predictions"]
        ]
    return result


def setup_inference_pipeline():
    """Start the Kafka inference pipeline in a daemon thread.

    The thread runs its own asyncio event loop so that a Kafka
    failure does not bring down the FastAPI server.
    """
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

    from src.db.database import SessionLocal
    from src.services.anomaly_config_service import AnomalyConfigService
    from src.services.config_service import MLConfigService
    from src.services.data_storage_client import DataStorageClient
    from src.services.mlflow_service import MLflowService
    from utils.kmw import PyKafBridge

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
