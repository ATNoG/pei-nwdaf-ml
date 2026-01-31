"""FastAPI dependency injection for shared resources."""

import mlflow
from mlflow import MlflowClient

from src.core.config import settings


def get_mlflow_client() -> MlflowClient:
    """
    Get MLflow client instance.

    Sets the tracking URI and returns a configured client.
    """
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    return MlflowClient()
