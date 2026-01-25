"""
Integration tests for model config storage and retrieval.
Tests the full flow: create model with config → train model → verify stored config is used.
"""
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch
import numpy as np

from src.services.model_service import ModelService
from src.services.training_service import TrainingService
from src.config.model_config import ModelConfig, TrainingConfig, ArchitectureConfig, SequenceConfig


@pytest.fixture(scope="module")
def temp_mlflow_dir():
    """Create temporary directory for MLflow tracking"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="module")
def mlflow_setup(temp_mlflow_dir):
    """Setup MLflow with temporary tracking URI"""
    import mlflow
    tracking_uri = f"file://{temp_mlflow_dir}"
    mlflow.set_tracking_uri(tracking_uri)

    # Create and set default experiment
    experiment_name = "test_experiment"
    experiment_id = mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    yield tracking_uri

    # Cleanup
    mlflow.set_tracking_uri(None)


@pytest.fixture
def mock_ml_interface():
    """Create mock ML interface for testing"""
    mock = Mock()
    mock.request_data_from_storage.return_value = [{
        "cell_index": 1,
        "latency_mean": 50.0,
        "throughput_mean": 1000.0,
        "num_ues": 10,
    }]
    mock.fetch_known_cells.return_value = [1, 2]
    mock.fetch_training_data_for_cells.return_value = [
        {
            "cell_index": 1,
            "latency_mean": 50.0,
            "throughput_mean": 1000.0,
            "num_ues": 10,
        }
    ]
    return mock


@pytest.mark.integration
class TestConfigStorageAndRetrieval:
    """Test model config is stored on creation and retrieved during training"""

    def test_create_model_stores_config(self, mlflow_setup, mock_ml_interface):
        """Test that model creation stores config in MLflow"""
        from mlflow.tracking import MlflowClient

        # Create custom config with specific values
        custom_config = ModelConfig(
            training=TrainingConfig(
                learning_rate=0.005,
                max_epochs=15,
                batch_size=32
            ),
            architecture=ArchitectureConfig(
                hidden_size=128,
                num_layers=3,
                dropout=0.3
            ),
            sequence=SequenceConfig(
                sequence_length=1
            )
        )

        # Create model with custom config
        service = ModelService(mock_ml_interface)
        model_name = "test_config_model_ann"

        service.create_model_instance(
            horizon=60,
            analytics_type="latency",
            model_type="ann",
            model_config=custom_config,
            name=model_name
        )

        # Verify model exists and has config artifact
        client = MlflowClient()
        model = client.get_registered_model(model_name)
        assert model.name == model_name

        # Get version 1 (creation version)
        version_1 = client.get_model_version(model_name, "1")

        # Verify config artifact exists
        import mlflow
        artifact_path = client.download_artifacts(version_1.run_id, "config/model_config.json")

        import json
        with open(artifact_path, 'r') as f:
            stored_config = json.load(f)

        # Verify stored config matches what we provided
        assert stored_config["training"]["learning_rate"] == 0.005
        assert stored_config["training"]["max_epochs"] == 15
        assert stored_config["training"]["batch_size"] == 32
        assert stored_config["architecture"]["hidden_size"] == 128
        assert stored_config["architecture"]["num_layers"] == 3
        assert stored_config["architecture"]["dropout"] == 0.3

    def test_training_loads_stored_config(self, mlflow_setup, mock_ml_interface):
        """Test that training loads and uses the stored config"""
        # Create model with specific max_epochs
        custom_config = ModelConfig(
            training=TrainingConfig(
                learning_rate=0.01,
                max_epochs=20  # Custom value to verify it's used
            ),
            sequence=SequenceConfig(
                sequence_length=1
            )
        )

        model_name = "test_training_config_ann"

        model_service = ModelService(mock_ml_interface)
        model_service.create_model_instance(
            horizon=60,
            analytics_type="latency",
            model_type="ann",
            model_config=custom_config,
            name=model_name
        )

        # Now test that training service loads the config
        training_service = TrainingService(mock_ml_interface)

        # Load config using the private method
        loaded_config = training_service._load_model_config(model_name)

        # Verify loaded config matches stored config
        assert loaded_config.training.learning_rate == 0.01
        assert loaded_config.training.max_epochs == 20
        assert loaded_config.sequence.sequence_length == 1

    def test_full_training_flow_with_stored_config(self, mlflow_setup, mock_ml_interface):
        """Test complete flow: create → train → verify stored config was used"""
        # Create model with very specific config
        custom_config = ModelConfig(
            training=TrainingConfig(
                learning_rate=0.002,
                max_epochs=5  # Low number for fast test
            ),
            architecture=ArchitectureConfig(
                hidden_size=16  # Small for fast test
            ),
            sequence=SequenceConfig(
                sequence_length=1
            )
        )

        model_name = "test_full_flow_ann"

        # Step 1: Create model
        model_service = ModelService(mock_ml_interface)
        model_service.create_model_instance(
            horizon=60,
            analytics_type="latency",
            model_type="ann",
            model_config=custom_config,
            name=model_name
        )

        # Step 2: Mock training data
        training_service = TrainingService(mock_ml_interface)

        # Track status callbacks to verify max_epochs is used
        callback_epochs = []
        def status_callback(current_epoch: int, total_epochs: int, loss: float = None):
            callback_epochs.append((current_epoch, total_epochs))

        # Step 3: Train model (should load and use stored config)
        with patch.object(training_service.ml_interface, 'fetch_known_cells', return_value=[1]):
            with patch.object(training_service.ml_interface, 'fetch_training_data_for_cells') as mock_fetch:
                # Create enough mock data
                mock_data = []
                for i in range(50):
                    mock_data.append({
                        "cell_index": 1,
                        "latency_mean": 50.0 + i * 0.5,
                        "throughput_mean": 1000.0,
                        "num_ues": 10,
                        "rsrp_mean": -80.0,
                        "rsrq_mean": -10.0,
                    })
                mock_fetch.return_value = mock_data

                result = training_service.train_model_by_name(
                    model_name=model_name,
                    data_limit_per_cell=100,
                    status_callback=status_callback
                )

        # Step 4: Verify training used the stored config
        assert result["status"] == "success"
        assert result["model_name"] == model_name

        # Verify that max_epochs from stored config was used
        # The callback should have been called with total_epochs=5
        if callback_epochs:
            # At least one callback should have total_epochs=5
            total_epochs_values = [epochs[1] for epochs in callback_epochs]
            assert 5 in total_epochs_values, f"Expected max_epochs=5 to be used, got {set(total_epochs_values)}"

    def test_config_persists_across_versions(self, mlflow_setup, mock_ml_interface):
        """Test that config from version 1 is always loaded, even after retraining"""
        # Create model with specific config
        original_config = ModelConfig(
            training=TrainingConfig(
                learning_rate=0.003,
                max_epochs=8
            ),
            sequence=SequenceConfig(
                sequence_length=1
            )
        )

        model_name = "test_persist_config_ann"

        model_service = ModelService(mock_ml_interface)
        model_service.create_model_instance(
            horizon=60,
            analytics_type="latency",
            model_type="ann",
            model_config=original_config,
            name=model_name
        )

        # Train model (creates version 2)
        training_service = TrainingService(mock_ml_interface)

        with patch.object(training_service.ml_interface, 'fetch_known_cells', return_value=[1]):
            with patch.object(training_service.ml_interface, 'fetch_training_data_for_cells') as mock_fetch:
                mock_data = []
                for i in range(30):
                    mock_data.append({
                        "cell_index": 1,
                        "latency_mean": 50.0,
                        "throughput_mean": 1000.0,
                        "num_ues": 10,
                        "rsrp_mean": -80.0,
                    })
                mock_fetch.return_value = mock_data

                training_service.train_model_by_name(model_name=model_name)

        # Load config again - should still get original from version 1
        loaded_config = training_service._load_model_config(model_name)

        assert loaded_config.training.learning_rate == 0.003
        assert loaded_config.training.max_epochs == 8
