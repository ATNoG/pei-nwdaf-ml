"""Unit tests for MLflow service."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from mlflow.exceptions import MlflowException

from src.services.mlflow_service import MLflowService
from src.schemas.model import ArchitectureType, ModelConfig
from src.db.model_config import ModelConfigDB


class TestMLflowServiceCreate:
    """Tests for MLflowService.create_model()."""

    def test_create_model_success(self, mock_mlflow_client, mock_config_service, sample_model_config):
        """Test successfully creating a model."""
        # Setup mocks
        mock_registered_model = MagicMock()
        mock_registered_model.name = "generated-uuid"
        mock_registered_model.creation_timestamp = 1706745600000
        mock_mlflow_client.create_registered_model.return_value = mock_registered_model

        service = MLflowService(mock_mlflow_client, mock_config_service)

        # Create model
        with patch('src.services.mlflow_service.uuid4') as mock_uuid:
            mock_uuid.return_value = MagicMock(hex="test-uuid-123")
            result = service.create_model("my_test_model", sample_model_config)

        # Verify MLflow calls
        mock_mlflow_client.create_registered_model.assert_called_once()
        mock_mlflow_client.set_registered_model_tag.assert_called_once()  # Only name tag

        # Verify config service calls
        mock_config_service.create.assert_called_once()

        # Verify result
        assert result.name == "my_test_model"
        assert result.config == sample_model_config
        assert result.latest_version is None

    def test_create_model_stores_config_in_db(self, mock_mlflow_client, mock_config_service, sample_model_config):
        """Test that model config is correctly stored in database."""
        mock_registered_model = MagicMock()
        mock_registered_model.creation_timestamp = 1706745600000
        mock_mlflow_client.create_registered_model.return_value = mock_registered_model

        service = MLflowService(mock_mlflow_client, mock_config_service)
        service.create_model("test_model", sample_model_config)

        # Verify config service create was called
        mock_config_service.create.assert_called_once()
        db_config = mock_config_service.create.call_args[0][0]

        # Verify it's a ModelConfigDB instance with correct data
        assert isinstance(db_config, ModelConfigDB)
        assert db_config.name == "test_model"
        assert db_config.architecture == "lstm"
        assert db_config.input_fields == ["latency_mean", "rsrp_mean", "sinr_mean"]
        assert db_config.output_fields == ["latency_mean"]
        assert db_config.window_duration_seconds == 60
        assert db_config.lookback_steps == 30
        assert db_config.forecast_steps == 5
        assert db_config.hidden_size == 32

    def test_create_model_rollback_on_db_error(self, mock_mlflow_client, mock_config_service, sample_model_config):
        """Test that MLflow model is cleaned up if database insert fails."""
        mock_registered_model = MagicMock()
        mock_registered_model.creation_timestamp = 1706745600000
        mock_mlflow_client.create_registered_model.return_value = mock_registered_model

        # Simulate DB error
        mock_config_service.create.side_effect = Exception("DB error")

        service = MLflowService(mock_mlflow_client, mock_config_service)

        with pytest.raises(Exception, match="DB error"):
            service.create_model("test_model", sample_model_config)

        # Verify cleanup
        mock_mlflow_client.delete_registered_model.assert_called_once()


class TestMLflowServiceGet:
    """Tests for MLflowService.get_model()."""

    def test_get_model_success(self, mock_mlflow_client, mock_config_service):
        """Test successfully retrieving a model."""
        # Setup database mock
        mock_db_config = MagicMock(spec=ModelConfigDB)
        mock_db_config.model_id = "test-model-uuid"
        mock_db_config.name = "my_test_model"
        mock_db_config.architecture = "lstm"
        mock_db_config.input_fields = ["latency_mean", "rsrp_mean", "sinr_mean"]
        mock_db_config.output_fields = ["latency_mean"]
        mock_db_config.window_duration_seconds = 60
        mock_db_config.lookback_steps = 30
        mock_db_config.forecast_steps = 5
        mock_db_config.hidden_size = 32
        mock_db_config.created_at = datetime(2024, 2, 1, 0, 0, 0)

        mock_config_service.get_config.return_value = mock_db_config
        mock_config_service.config_from_db.return_value = ModelConfig(
            architecture=ArchitectureType.LSTM,
            input_fields=["latency_mean", "rsrp_mean", "sinr_mean"],
            output_fields=["latency_mean"],
            window_duration_seconds=60,
            lookback_steps=30,
            forecast_steps=5,
            hidden_size=32,
        )

        # Setup MLflow mock
        mock_registered_model = MagicMock()
        mock_registered_model.latest_versions = []
        mock_mlflow_client.get_registered_model.return_value = mock_registered_model

        service = MLflowService(mock_mlflow_client, mock_config_service)
        result = service.get_model("test-model-uuid")

        # Verify config service query
        mock_config_service.get_config.assert_called_once_with("test-model-uuid")

        # Verify result
        assert result.id == "test-model-uuid"
        assert result.name == "my_test_model"
        assert result.config.architecture == ArchitectureType.LSTM

    def test_get_model_not_found_in_db(self, mock_mlflow_client, mock_config_service):
        """Test getting a model that doesn't exist in database."""
        mock_config_service.get_config.return_value = None

        service = MLflowService(mock_mlflow_client, mock_config_service)

        with pytest.raises(ValueError, match="not found"):
            service.get_model("nonexistent-uuid")

    def test_get_model_not_found_in_mlflow(self, mock_mlflow_client, mock_config_service):
        """Test getting a model that exists in DB but not in MLflow."""
        # Setup database mock
        mock_db_config = MagicMock(spec=ModelConfigDB)
        mock_db_config.model_id = "test-model-uuid"
        mock_config_service.get_config.return_value = mock_db_config

        # MLflow returns not found
        mock_mlflow_client.get_registered_model.side_effect = MlflowException("Not found")

        service = MLflowService(mock_mlflow_client, mock_config_service)

        with pytest.raises(ValueError, match="not found in MLflow"):
            service.get_model("test-model-uuid")

    def test_get_model_with_version_info(self, mock_mlflow_client, mock_config_service):
        """Test retrieving a model with training version info."""
        # Setup database mock
        mock_db_config = MagicMock(spec=ModelConfigDB)
        mock_db_config.model_id = "test-model-uuid"
        mock_db_config.name = "my_test_model"
        mock_db_config.created_at = datetime(2024, 2, 1, 0, 0, 0)

        mock_config_service.get_config.return_value = mock_db_config
        mock_config_service.config_from_db.return_value = ModelConfig(
            architecture=ArchitectureType.LSTM,
            input_fields=["latency_mean"],
            output_fields=["latency_mean"],
            window_duration_seconds=60,
            lookback_steps=30,
            forecast_steps=5,
            hidden_size=32,
        )

        # Setup MLflow mock with version info
        mock_version = MagicMock()
        mock_version.version = "2"
        mock_version.run_id = "run-abc123"

        mock_registered_model = MagicMock()
        mock_registered_model.latest_versions = [mock_version]
        mock_mlflow_client.get_registered_model.return_value = mock_registered_model

        # Mock run info
        mock_run = MagicMock()
        mock_run.info.end_time = 1706832000000  # 2024-02-02 00:00:00 UTC
        mock_run.data.metrics = {"final_loss": 0.042}
        mock_mlflow_client.get_run.return_value = mock_run

        service = MLflowService(mock_mlflow_client, mock_config_service)
        result = service.get_model("test-model-uuid")

        # Verify version info
        assert result.latest_version == 2
        assert result.mlflow_run_id == "run-abc123"
        assert result.training_loss == 0.042
        assert result.last_trained_at is not None


class TestMLflowServiceList:
    """Tests for MLflowService.list_models()."""

    def test_list_models_empty(self, mock_mlflow_client, mock_config_service):
        """Test listing models when none exist."""
        mock_config_service.list_all.return_value = []

        service = MLflowService(mock_mlflow_client, mock_config_service)
        result = service.list_models()

        assert result == []

    def test_list_models_multiple(self, mock_mlflow_client, mock_config_service):
        """Test listing multiple models."""
        # Create mock database configs
        mock_config1 = MagicMock(spec=ModelConfigDB)
        mock_config1.model_id = "uuid-1"
        mock_config1.name = "model_one"
        mock_config1.architecture = "ann"
        mock_config1.created_at = datetime(2024, 2, 1, 0, 0, 0)

        mock_config2 = MagicMock(spec=ModelConfigDB)
        mock_config2.model_id = "uuid-2"
        mock_config2.name = "model_two"
        mock_config2.architecture = "lstm"
        mock_config2.created_at = datetime(2024, 2, 2, 0, 0, 0)

        mock_config_service.list_all.return_value = [mock_config1, mock_config2]

        # Setup MLflow mocks for version info
        mock_rm1 = MagicMock()
        mock_rm1.latest_versions = []

        mock_version = MagicMock()
        mock_version.version = "3"
        mock_rm2 = MagicMock()
        mock_rm2.latest_versions = [mock_version]

        def get_model_side_effect(model_id):
            if model_id == "uuid-1":
                return mock_rm1
            elif model_id == "uuid-2":
                return mock_rm2
            raise MlflowException("Not found")

        mock_mlflow_client.get_registered_model.side_effect = get_model_side_effect

        service = MLflowService(mock_mlflow_client, mock_config_service)
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

    def test_delete_model_success(self, mock_mlflow_client, mock_config_service):
        """Test successfully deleting a model."""
        service = MLflowService(mock_mlflow_client, mock_config_service)
        service.delete_model("uuid-123")

        # Verify config service delete
        mock_config_service.delete_model.assert_called_once_with("uuid-123")

        # Verify MLflow delete
        mock_mlflow_client.delete_registered_model.assert_called_once_with("uuid-123")

    def test_delete_model_not_found(self, mock_mlflow_client, mock_config_service):
        """Test deleting a non-existent model raises ValueError."""
        mock_config_service.delete_model.side_effect = ValueError("Model 'nonexistent-uuid' not found")

        service = MLflowService(mock_mlflow_client, mock_config_service)

        with pytest.raises(ValueError, match="not found"):
            service.delete_model("nonexistent-uuid")

        # Verify MLflow delete was not called
        mock_mlflow_client.delete_registered_model.assert_not_called()

    def test_delete_model_handles_mlflow_missing(self, mock_mlflow_client, mock_config_service):
        """Test deleting when model exists in DB but not in MLflow."""
        # MLflow raises exception
        mock_mlflow_client.delete_registered_model.side_effect = MlflowException("Not found")

        service = MLflowService(mock_mlflow_client, mock_config_service)

        # Should not raise - gracefully handles MLflow missing
        service.delete_model("uuid-123")

        # DB delete should still happen
        mock_config_service.delete_model.assert_called_once()
