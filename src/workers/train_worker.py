import asyncio
import logging
import os

import mlflow
from mlflow import MlflowClient

from src.core.config import settings
from src.db.database import SessionLocal
from src.services.data_storage_client import DataStorageClient

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    job_id = os.environ.get("TRAINING_JOB_ID")
    training_type = os.environ.get("TRAINING_TYPE")

    if not job_id:
        raise RuntimeError("TRAINING_JOB_ID env var not set")
    if training_type not in ("forecast", "anomaly"):
        raise RuntimeError("TRAINING_TYPE must be 'forecast' or 'anomaly'")

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    db = SessionLocal()
    try:
        if training_type == "forecast":
            from src.services.config_service import MLConfigService
            from src.services.mlflow_service import MLflowService
            from src.services.training_service import TrainingService

            config_service = MLConfigService(db)
            service = TrainingService(
                mlflow_service=MLflowService(MlflowClient(), config_service),
                data_storage_client=DataStorageClient(),
                ml_config_service=config_service,
                db=db,
            )
        else:
            from src.services.anomaly_config_service import AnomalyConfigService
            from src.services.anomaly_training_service import AnomalyTrainingService

            service = AnomalyTrainingService(
                data_storage_client=DataStorageClient(),
                anomaly_config_service=AnomalyConfigService(db),
                db=db,
            )

        asyncio.run(service.execute_training(job_id))
    finally:
        db.close()


if __name__ == "__main__":
    main()
