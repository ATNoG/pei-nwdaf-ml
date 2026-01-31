"""Unit tests for MLflow service."""

import json
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from mlflow.exceptions import MlflowException

from src.services.mlflow_service import MLflowService
from src.schemas.model import ArchitectureType, ModelConfig


class TestMLflowServiceCreate:
    """Tests for MLflowService.create_model()."""

    def test_create_model_success(self, mock_mlflow_client, sample_model_config):
        """Test successfully creating a model."""
        # Setup mock
        mock_registered_model = MagicMock()
        mock_registered_model.name = "generated-uuid"
        mock_registered_model.creation_timestamp = 1706745600000
        mock_mlflow_client.create_registered_model.return_value = mock_registered_model
        mock_mlflow_client.search_registered_models.return_value = []

        service = MLflowService(mock_mlflow_client)

        # Create model
        result = service.create_model("my_test_model", sample_model_config)

        # Verify MLflow calls
        mock_mlflow_client.search_registered_models.assert_called_once()
        mock_mlflow_client.create_registered_model.assert_called_once()

        # Verify tags were set
        assert mock_mlflow_client.set_registered_model_tag.call_count == 8

        # Verify result
        assert result.name == "my_test_model"
        assert result.config == sample_model_config
        assert result.latest_version is None

    def test_create_model_duplicate_name(self, mock_mlflow_client, sample_model_config):
        """Test creating a model with duplicate name fails."""
        # Setup mock to return existing model
        existing_model = MagicMock()
        existing_model.tags = {"name": "my_test_model"}
        mock_mlflow_client.search_registered_models.return_value = [existing_model]

        service = MLflowService(mock_mlflow_client)

        # Attempt to create duplicate
        with pytest.raises(ValueError, match="already exists"):
            service.create_model("my_test_model", sample_model_config)

        # Verify create was never called
        mock_mlflow_client.create_registered_model.assert_not_called()

    def test_create_model_stores_config_as_tags(self, mock_mlflow_client, sample_model_config):
        """Test that model config is correctly stored as tags."""
        mock_registered_model = MagicMock()
        mock_registered_model.name = "uuid-123"
        mock_registered_model.creation_timestamp = 1706745600000
        mock_mlflow_client.create_registered_model.return_value = mock_registered_model
        mock_mlflow_client.search_registered_models.return_value = []

        service = MLflowService(mock_mlflow_client)
        service.create_model("test_model", sample_model_config)

        # Verify specific tag calls
        tag_calls = mock_mlflow_client.set_registered_model_tag.call_args_list

        # Check that critical tags were set
        tag_dict = {call[0][1]: call[0][2] for call in tag_calls}

        assert tag_dict["name"] == "test_model"
        assert tag_dict["config:architecture"] == "lstm"
        assert tag_dict["config:window_duration_seconds"] == "60"
        assert tag_dict["config:lookback_steps"] == "30"
        assert tag_dict["config:forecast_steps"] == "5"
        assert tag_dict["config:hidden_size"] == "32"

        # Check JSON fields
        assert json.loads(tag_dict["config:input_fields"]) == ["latency_mean", "rsrp_mean", "sinr_mean"]
        assert json.loads(tag_dict["config:output_fields"]) == ["latency_mean"]


class TestMLflowServiceGet:
    """Tests for MLflowService.get_model()."""

    def test_get_model_success(self, mock_mlflow_client, sample_registered_model):
        """Test successfully retrieving a model."""
        mock_mlflow_client.get_registered_model.return_value = sample_registered_model

        service = MLflowService(mock_mlflow_client)
        result = service.get_model("test-model-uuid")

        # Verify MLflow call (called twice: once in get_model, once in _get_model_config)
        assert mock_mlflow_client.get_registered_model.call_count == 2

        # Verify result
        assert result.id == "test-model-uuid"
        assert result.name == "my_test_model"
        assert result.config.architecture == ArchitectureType.LSTM
        assert result.config.input_fields == ["latency_mean", "rsrp_mean", "sinr_mean"]
        assert result.config.output_fields == ["latency_mean"]
        assert result.config.window_duration_seconds == 60
        assert result.config.lookback_steps == 30
        assert result.config.forecast_steps == 5
        assert result.config.hidden_size == 32

    def test_get_model_not_found(self, mock_mlflow_client):
        """Test getting a non-existent model raises ValueError."""
        mock_mlflow_client.get_registered_model.side_effect = MlflowException("Not found")

        service = MLflowService(mock_mlflow_client)

        with pytest.raises(ValueError, match="not found"):
            service.get_model("nonexistent-uuid")

    def test_get_model_with_version_info(self, mock_mlflow_client, sample_registered_model):
        """Test retrieving a model with training version info."""
        # Add version info
        mock_version = MagicMock()
        mock_version.version = "2"
        mock_version.run_id = "run-abc123"
        sample_registered_model.latest_versions = [mock_version]

        # Mock run info
        mock_run = MagicMock()
        mock_run.info.end_time = 1706832000000  # 2024-02-02 00:00:00 UTC
        mock_run.data.metrics = {"loss": 0.042}
        mock_mlflow_client.get_run.return_value = mock_run
        mock_mlflow_client.get_registered_model.return_value = sample_registered_model

        service = MLflowService(mock_mlflow_client)
        result = service.get_model("test-model-uuid")

        # Verify version info
        assert result.latest_version == 2
        assert result.mlflow_run_id == "run-abc123"
        assert result.training_loss == 0.042
        assert result.last_trained_at is not None


class TestMLflowServiceList:
    """Tests for MLflowService.list_models()."""

    def test_list_models_empty(self, mock_mlflow_client):
        """Test listing models when none exist."""
        mock_mlflow_client.search_registered_models.return_value = []

        service = MLflowService(mock_mlflow_client)
        result = service.list_models()

        assert result == []

    def test_list_models_multiple(self, mock_mlflow_client):
        """Test listing multiple models."""
        # Create mock models
        model1 = MagicMock()
        model1.name = "uuid-1"
        model1.tags = {
            "name": "model_one",
            "config:architecture": "ann"
        }
        model1.creation_timestamp = 1706745600000
        model1.latest_versions = []

        model2 = MagicMock()
        model2.name = "uuid-2"
        model2.tags = {
            "name": "model_two",
            "config:architecture": "lstm"
        }
        model2.creation_timestamp = 1706832000000

        mock_version = MagicMock()
        mock_version.version = "3"
        model2.latest_versions = [mock_version]

        mock_mlflow_client.search_registered_models.return_value = [model1, model2]

        service = MLflowService(mock_mlflow_client)
        result = service.list_models()

        # Verify results
        assert len(result) == 2

        assert result[0].id == "uuid-1"
        assert result[0].name == "model_one"
        assert result[0].architecture == ArchitectureType.ANN
        assert result[0].latest_version is None

        assert result[1].id == "uuid-2"
        assert result[1].name == "model_two"
        assert result[1].architecture == ArchitectureType.LSTM
        assert result[1].latest_version == 3


class TestMLflowServiceDelete:
    """Tests for MLflowService.delete_model()."""

    def test_delete_model_success(self, mock_mlflow_client):
        """Test successfully deleting a model."""
        service = MLflowService(mock_mlflow_client)
        service.delete_model("uuid-123")

        mock_mlflow_client.delete_registered_model.assert_called_once_with("uuid-123")

    def test_delete_model_not_found(self, mock_mlflow_client):
        """Test deleting a non-existent model raises ValueError."""
        mock_mlflow_client.delete_registered_model.side_effect = MlflowException("Not found")

        service = MLflowService(mock_mlflow_client)

        with pytest.raises(ValueError, match="not found"):
            service.delete_model("nonexistent-uuid")


class TestMLflowServiceGetConfig:
    """Tests for MLflowService._get_model_config()."""

    def test_get_model_config(self, mock_mlflow_client, sample_registered_model):
        """Test reconstructing ModelConfig from tags."""
        mock_mlflow_client.get_registered_model.return_value = sample_registered_model

        service = MLflowService(mock_mlflow_client)
        config = service._get_model_config("test-model-uuid")

        assert isinstance(config, ModelConfig)
        assert config.architecture == ArchitectureType.LSTM
        assert config.input_fields == ["latency_mean", "rsrp_mean", "sinr_mean"]
        assert config.output_fields == ["latency_mean"]
        assert config.window_duration_seconds == 60
        assert config.lookback_steps == 30
        assert config.forecast_steps == 5
        assert config.hidden_size == 32
