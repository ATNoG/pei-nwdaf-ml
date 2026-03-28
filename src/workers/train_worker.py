import asyncio
import logging
import os

import mlflow
from mlflow import MlflowClient

from src.core.config import settings
from src.db.database import SessionLocal
from src.services.config_service import MLConfigService
from src.services.data_storage_client import DataStorageClient
from src.services.mlflow_service import MLflowService
from src.services.training_service import TrainingService

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    job_id = os.environ.get("TRAINING_JOB_ID")
    if not job_id:
        raise RuntimeError("TRAINING_JOB_ID env var not set")

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    db = SessionLocal()
    try:
        config_service = MLConfigService(db)
        mlflow_service = MLflowService(MlflowClient(), config_service)
        training_service = TrainingService(
            mlflow_service=mlflow_service,
            data_storage_client=DataStorageClient(),
            ml_config_service=config_service,
            db=db,
        )
        asyncio.run(training_service.execute_training(job_id))
    finally:
        db.close()


if __name__ == "__main__":
    main()
