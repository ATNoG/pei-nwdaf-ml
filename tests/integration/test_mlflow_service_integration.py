"""Integration tests for MLflow service with real MLflow."""

import pytest

from src.services.mlflow_service import MLflowService
from src.schemas.model import ArchitectureType, ModelConfig

pytestmark = pytest.mark.integration


@pytest.fixture
def sample_config():
    """Sample model config for integration tests."""
    return ModelConfig(
        architecture=ArchitectureType.LSTM,
        input_fields=["latency_mean", "rsrp_mean"],
        output_fields=["latency_mean"],
        window_duration_seconds=60,
        lookback_steps=10,
        forecast_steps=5,
        hidden_size=32,
    )


@pytest.fixture
def mlflow_service(mlflow_client, db_session):
    """MLflow service instance with real client and database session."""
    from src.services.config_service import MLConfigService

    config_service = MLConfigService(db_session)
    return MLflowService(mlflow_client, config_service)


class TestMLflowServiceIntegration:
    """Integration tests with real MLflow."""

    def test_create_and_get_model(self, mlflow_service, sample_config):
        """Test creating a model and retrieving it."""
        # Create model
        result = mlflow_service.create_model("test_model", sample_config)

        assert result.name == "test_model"
        assert result.config == sample_config
        assert result.id is not None
        assert result.latest_version is None

        # Get model by ID
        retrieved = mlflow_service.get_model(result.id)

        assert retrieved.id == result.id
        assert retrieved.name == "test_model"
        assert retrieved.config.architecture == ArchitectureType.LSTM
        assert retrieved.config.input_fields == ["latency_mean", "rsrp_mean"]
        assert retrieved.config.output_fields == ["latency_mean"]
        assert retrieved.config.window_duration_seconds == 60
        assert retrieved.config.lookback_steps == 10
        assert retrieved.config.forecast_steps == 5
        assert retrieved.config.hidden_size == 32


    def test_list_models(self, mlflow_service, sample_config):
        """Test listing models."""
        # Initially empty
        models = mlflow_service.list_models()
        initial_count = len(models)

        # Create two models
        config1 = sample_config
        config2 = ModelConfig(
            architecture=ArchitectureType.ANN,
            input_fields=["sinr_mean"],
            output_fields=["latency_mean"],
            window_duration_seconds=300,
            lookback_steps=20,
            forecast_steps=3,
            hidden_size=64,
        )

        mlflow_service.create_model("model_one", config1)
        mlflow_service.create_model("model_two", config2)

        # List models
        models = mlflow_service.list_models()

        assert len(models) == initial_count + 2

        # Find our models
        model_names = [m.name for m in models]
        assert "model_one" in model_names
        assert "model_two" in model_names

        # Check architectures
        model_one = next(m for m in models if m.name == "model_one")
        model_two = next(m for m in models if m.name == "model_two")

        assert model_one.architecture == ArchitectureType.LSTM
        assert model_two.architecture == ArchitectureType.ANN

    def test_delete_model(self, mlflow_service, sample_config):
        """Test deleting a model."""
        # Create model
        result = mlflow_service.create_model("to_delete", sample_config)
        model_id = result.id

        # Verify it exists
        retrieved = mlflow_service.get_model(model_id)
        assert retrieved.name == "to_delete"

        # Delete it
        mlflow_service.delete_model(model_id)

        # Verify it's gone
        with pytest.raises(ValueError, match="not found"):
            mlflow_service.get_model(model_id)

    def test_delete_nonexistent_model_fails(self, mlflow_service):
        """Test deleting a non-existent model fails."""
        with pytest.raises(ValueError, match="not found"):
            mlflow_service.delete_model("nonexistent-uuid-12345")

    def test_get_nonexistent_model_fails(self, mlflow_service):
        """Test getting a non-existent model fails."""
        with pytest.raises(ValueError, match="not found"):
            mlflow_service.get_model("nonexistent-uuid-12345")

    def test_config_reconstruction(self, mlflow_service):
        """Test that complex configs are correctly stored and reconstructed."""
        config = ModelConfig(
            architecture=ArchitectureType.ANN,
            input_fields=[
                "latency_mean",
                "latency_max",
                "rsrp_mean",
                "rsrp_std",
                "sinr_mean",
            ],
            output_fields=["latency_mean", "latency_max"],
            window_duration_seconds=300,
            lookback_steps=50,
            forecast_steps=10,
            hidden_size=128,
        )

        # Create model
        result = mlflow_service.create_model("complex_config", config)

        # Retrieve and verify
        retrieved = mlflow_service.get_model(result.id)

        assert retrieved.config.architecture == config.architecture
        assert retrieved.config.input_fields == config.input_fields
        assert retrieved.config.output_fields == config.output_fields
        assert retrieved.config.window_duration_seconds == config.window_duration_seconds
        assert retrieved.config.lookback_steps == config.lookback_steps
        assert retrieved.config.forecast_steps == config.forecast_steps
        assert retrieved.config.hidden_size == config.hidden_size

    def test_multiple_models_different_configs(self, mlflow_service):
        """Test creating multiple models with different configurations."""
        configs = [
            ModelConfig(
                architecture=ArchitectureType.ANN,
                input_fields=["latency_mean"],
                output_fields=["latency_mean"],
                window_duration_seconds=60,
                lookback_steps=5,
                forecast_steps=1,
                hidden_size=16,
            ),
            ModelConfig(
                architecture=ArchitectureType.LSTM,
                input_fields=["latency_mean", "rsrp_mean", "sinr_mean"],
                output_fields=["latency_mean"],
                window_duration_seconds=300,
                lookback_steps=30,
                forecast_steps=7,
                hidden_size=64,
            ),
        ]

        created_ids = []
        for i, config in enumerate(configs):
            result = mlflow_service.create_model(f"multi_model_{i}", config)
            created_ids.append(result.id)

        # Verify each model has correct config
        for i, (model_id, expected_config) in enumerate(zip(created_ids, configs)):
            retrieved = mlflow_service.get_model(model_id)
            assert retrieved.name == f"multi_model_{i}"
            assert retrieved.config == expected_config

    def test_get_model_by_id_full_details(self, mlflow_service, sample_config):
        """Test getting model by ID returns full details."""
        # Create a model
        created = mlflow_service.create_model("detailed_model", sample_config)
        model_id = created.id

        # Get the model by ID
        retrieved = mlflow_service.get_model(model_id)

        # Verify all details match
        assert retrieved.id == model_id
        assert retrieved.name == "detailed_model"
        assert retrieved.config.architecture == sample_config.architecture
        assert retrieved.config.input_fields == sample_config.input_fields
        assert retrieved.config.output_fields == sample_config.output_fields
        assert retrieved.config.window_duration_seconds == sample_config.window_duration_seconds
        assert retrieved.config.lookback_steps == sample_config.lookback_steps
        assert retrieved.config.forecast_steps == sample_config.forecast_steps
        assert retrieved.config.hidden_size == sample_config.hidden_size
        assert retrieved.created_at is not None
        assert retrieved.latest_version is None  # No training yet

    def test_delete_and_verify_cascade(self, mlflow_service, sample_config):
        """Test that deletion removes model from both DB and MLflow."""
        # Create a model
        created = mlflow_service.create_model("cascade_delete_model", sample_config)
        model_id = created.id

        # Verify it exists
        retrieved = mlflow_service.get_model(model_id)
        assert retrieved.id == model_id

        # Delete it
        mlflow_service.delete_model(model_id)

        # Verify it's gone from both systems
        with pytest.raises(ValueError, match="not found"):
            mlflow_service.get_model(model_id)

        # Verify it's not in the list
        models = mlflow_service.list_models()
        model_ids = [m.id for m in models]
        assert model_id not in model_ids

    def test_duplicate_names_allowed(self, mlflow_service, sample_config):
        """Test that models with duplicate names are allowed."""
        # Create first model
        model1 = mlflow_service.create_model("duplicate_name", sample_config)

        # Create second model with same name but different config
        different_config = ModelConfig(
            architecture=ArchitectureType.ANN,
            input_fields=["rsrp_mean"],
            output_fields=["rsrp_mean"],
            window_duration_seconds=120,
            lookback_steps=10,
            forecast_steps=3,
            hidden_size=32,
        )
        model2 = mlflow_service.create_model("duplicate_name", different_config)

        # Verify both exist with different IDs
        assert model1.id != model2.id
        assert model1.name == model2.name == "duplicate_name"

        # Verify both can be retrieved
        retrieved1 = mlflow_service.get_model(model1.id)
        retrieved2 = mlflow_service.get_model(model2.id)

        assert retrieved1.id == model1.id
        assert retrieved2.id == model2.id
        assert retrieved1.config.architecture == ArchitectureType.LSTM
        assert retrieved2.config.architecture == ArchitectureType.ANN

        # Both should appear in list
        models = mlflow_service.list_models()
        duplicate_models = [m for m in models if m.name == "duplicate_name"]
        assert len(duplicate_models) >= 2
