"""Shared test fixtures."""

import pytest
from unittest.mock import Mock, MagicMock
from mlflow import MlflowClient


@pytest.fixture
def mock_mlflow_client():
    """Mock MLflow client for testing."""
    return Mock(spec=MlflowClient)


@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing."""
    from src.schemas.model import ModelConfig, ArchitectureType

    return ModelConfig(
        architecture=ArchitectureType.LSTM,
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
