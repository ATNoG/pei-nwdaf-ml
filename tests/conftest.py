"""Shared test fixtures."""

import os
import pytest
from unittest.mock import Mock, MagicMock
from mlflow import MlflowClient


os.environ.setdefault("DEV_MODE", "true")


def pytest_configure(config):
    """
    Set test environment variables before any imports.
    This runs before pytest collects tests, ensuring env vars are set
    before any module imports Settings.
    """
    test_env = {
        "API_HOST": "0.0.0.0",
        "API_PORT": "8060",
        "LOG_LEVEL": "INFO",
        "MLFLOW_TRACKING_URI": "http://localhost:5000",
        "MLFLOW_S3_ENDPOINT_URL": "http://localhost:9000",
        "AWS_ACCESS_KEY_ID": "minio",
        "AWS_SECRET_ACCESS_KEY": "minio123",
        "AWS_DEFAULT_REGION": "us-east-1",
        "DATA_STORAGE_API_URL": "http://localhost:8000",
        "DATA_STORAGE_EXAMPLE_ENDPOINT": "/api/v1/processed/latency/example",
        "DATA_STORAGE_DATA_ENDPOINT": "/api/v1/processed/latency",
        "DATA_STORAGE_CELL_ENDPOINT": "/api/v1/cell",
        "DATA_STORAGE_EXCLUDED_FIELDS": "",
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
    }

    # Only set if not already set (allows override via actual env vars or .env)
    for key, value in test_env.items():
        if key not in os.environ:
            os.environ[key] = value


@pytest.fixture
def mock_mlflow_client():
    """Mock MLflow client for testing."""
    return Mock(spec=MlflowClient)


@pytest.fixture
def mock_db_session():
    """Mock database session for testing."""
    from unittest.mock import MagicMock
    session = MagicMock()
    # Setup query mock to return itself for chaining
    session.query.return_value = session
    session.filter.return_value = session
    return session


@pytest.fixture
def mock_config_service():
    """Mock MLConfigService for testing."""
    from unittest.mock import MagicMock
    return MagicMock()


@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing."""
    from src.schemas.model import ModelConfig

    return ModelConfig(
        architecture="lstm",
        input_fields=["latency_mean", "rsrp_mean", "sinr_mean"],
        output_fields=["latency_mean"],
        window_duration_seconds=60,
        lookback_steps=30,
        forecast_steps=5,
        hidden_size=32,
    )


@pytest.fixture
def sample_registered_model():
    """Sample MLflow registered model response."""
    model = MagicMock()
    model.name = "test-model-uuid"
    model.creation_timestamp = 1706745600000  # 2024-02-01 00:00:00 UTC
    model.tags = {
        "name": "my_test_model",
        "config:architecture": "lstm",
        "config:input_fields": '["latency_mean", "rsrp_mean", "sinr_mean"]',
        "config:output_fields": '["latency_mean"]',
        "config:window_duration_seconds": "60",
        "config:lookback_steps": "30",
        "config:forecast_steps": "5",
        "config:hidden_size": "32",
        "is_default": "false",
    }
    model.latest_versions = []
    return model
