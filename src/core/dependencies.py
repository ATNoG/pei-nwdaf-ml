"""FastAPI dependency injection for shared resources."""

from typing import Any, Generator, Optional

import mlflow
from fastapi import Request
from mlflow import MlflowClient

from src.core.config import settings
from src.db.database import get_db
from src.services.config_service import MLConfigService
from src.services.mlflow_service import MLflowService


def get_mlflow_client() -> Generator[MLflowService, None, None]:
    """
    Get MLflowService instance with config service and database session.

    Sets the tracking URI and returns a configured service.
    """
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Get database session
    db = next(get_db())
    try:
        # Create config service
        config_service = MLConfigService(db)
        # Create and yield MLflow service
        yield MLflowService(client, config_service)
    finally:
        db.close()


async def get_policy_client(request: Request) -> Optional[Any]:
    """
    Get the app-level PolicyClient if policy is enabled.

    Returns None if policy is disabled or not initialized.
    """
    return getattr(request.app.state, "policy_client", None)
