"""
Tests for v1 API routers (model and training).
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.routers.v1 import model_router, training_router


@pytest.fixture
def mock_ml_interface():
    """Create a mock ML interface"""
    mock_interface = Mock()
    mock_interface.request_data_from_storage = Mock(return_value=[{"test": "data"}])
    return mock_interface


@pytest.fixture
def app_with_model_router(mock_ml_interface):
    """Create FastAPI app with model router"""
    app = FastAPI()
    app.include_router(model_router.router, prefix="/api/v1/model", tags=["model"])
    app.state.ml_interface = mock_ml_interface
    return app


@pytest.fixture
def app_with_training_router(mock_ml_interface):
    """Create FastAPI app with training router"""
    app = FastAPI()
    app.include_router(training_router.router, prefix="/api/v1/training", tags=["training"])
    app.state.ml_interface = mock_ml_interface
    return app


class TestModelRouter:
    """Tests for model router endpoints"""

    def test_create_model_instance_success(self, app_with_model_router):
        """Test successful model instance creation"""
        client = TestClient(app_with_model_router)

        with patch("src.routers.v1.model_router.ModelService") as mock_service_class:
            mock_service = Mock()
            mock_service.create_model_instance.return_value = "latency_ann_60"
            mock_service_class.return_value = mock_service

            response = client.post(
                "/api/v1/model/instance",
                json={
                    "analytics_type": "latency",
                    "horizon": 60,
                    "model_type": "ann",
                    "name": "latency_ann_60"
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "created"
            assert data["model_name"] == "latency_ann_60"

            mock_service.create_model_instance.assert_called_once()

    def test_create_model_instance_with_config(self, app_with_model_router):
        """Test model instance creation with custom config"""
        client = TestClient(app_with_model_router)

        with patch("src.routers.v1.model_router.ModelService") as mock_service_class:
            mock_service = Mock()
            mock_service.create_model_instance.return_value = "latency_lstm_60"
            mock_service_class.return_value = mock_service

            response = client.post(
                "/api/v1/model/instance",
                json={
                    "analytics_type": "latency",
                    "horizon": 60,
                    "model_type": "lstm",
                    "config": {
                        "training": {"learning_rate": 0.01, "max_epochs": 200},
                        "architecture": {"hidden_size": 128, "dropout": 0.3},
                    },
                    "name": "latency_lstm_60"
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "created"

            # Verify config was passed to model creation
            call_kwargs = mock_service.create_model_instance.call_args.kwargs
            assert call_kwargs["model_config"] is not None
            assert call_kwargs["model_config"].training.learning_rate == 0.01
            assert call_kwargs["model_config"].architecture.hidden_size == 128

    def test_create_model_instance_with_custom_name(self, app_with_model_router):
        """Test model instance creation with custom name"""
        client = TestClient(app_with_model_router)

        with patch("src.routers.v1.model_router.ModelService") as mock_service_class:
            mock_service = Mock()
            mock_service.create_model_instance.return_value = "my_custom_model"
            mock_service_class.return_value = mock_service

            response = client.post(
                "/api/v1/model/instance",
                json={
                    "analytics_type": "latency",
                    "horizon": 60,
                    "model_type": "ann",
                    "name": "my_custom_model",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["model_name"] == "my_custom_model"

            # Verify custom name was passed
            call_kwargs = mock_service.create_model_instance.call_args.kwargs
            assert call_kwargs["name"] == "my_custom_model"

    def test_create_model_instance_invalid_type(self, app_with_model_router):
        """Test model instance creation with invalid model type"""
        client = TestClient(app_with_model_router)

        with patch("src.routers.v1.model_router.ModelService") as mock_service_class:
            mock_service = Mock()
            mock_service.create_model_instance.side_effect = ValueError(
                "Model [invalid] not found"
            )
            mock_service_class.return_value = mock_service

            response = client.post(
                "/api/v1/model/instance",
                json={
                    "analytics_type": "latency",
                    "horizon": 60,
                    "model_type": "invalid",
                    "name": "invalid_model"
                },
            )

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_create_model_instance_already_exists(self, app_with_model_router):
        """Test model instance creation when model already exists"""
        client = TestClient(app_with_model_router)

        with patch("src.routers.v1.model_router.ModelService") as mock_service_class:
            mock_service = Mock()
            mock_service.create_model_instance.side_effect = ValueError(
                "Model instance 'latency_ann_60' already exists"
            )
            mock_service_class.return_value = mock_service

            response = client.post(
                "/api/v1/model/instance",
                json={
                    "analytics_type": "latency",
                    "horizon": 60,
                    "model_type": "ann",
                    "name": "latency_ann_60"
                },
            )

            assert response.status_code == 400
            assert "already exists" in response.json()["detail"].lower()

    def test_delete_model_instance_success(self, app_with_model_router):
        """Test successful model instance deletion"""
        client = TestClient(app_with_model_router)

        with patch("src.routers.v1.model_router.ModelService") as mock_service_class:
            mock_service = Mock()
            mock_service.delete_model_instance.return_value = True
            mock_service_class.return_value = mock_service

            response = client.delete("/api/v1/model/instance/latency_ann_60")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "deleted"
            assert data["model_name"] == "latency_ann_60"


class TestTrainingRouter:
    """Tests for training router endpoints"""

    def test_start_training_success(self, app_with_training_router):
        """Test successful training start"""
        client = TestClient(app_with_training_router)

        with patch("src.routers.v1.training_router.TrainingService") as mock_service_class:
            mock_service = Mock()
            mock_service.get_model_metadata.return_value = {
                "analytics_type": "latency",
                "model_type": "ann",
                "horizon": 60
            }
            mock_service_class.return_value = mock_service

            response = client.post(
                "/api/v1/training",
                json={
                    "model_name": "latency_ann_60",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "training_started"
            assert data["model_name"] == "latency_ann_60"
            mock_service.get_model_metadata.assert_called_once_with("latency_ann_60")


    def test_start_training_invalid_model_name(self, app_with_training_router):
        """Test training start with invalid model name"""
        client = TestClient(app_with_training_router)

        with patch("src.routers.v1.training_router.TrainingService") as mock_service_class:
            mock_service = Mock()
            mock_service.get_model_metadata.side_effect = ValueError(
                "Model 'invalid_model' not found in registry"
            )
            mock_service_class.return_value = mock_service

            response = client.post(
                "/api/v1/training",
                json={
                    "model_name": "invalid_model",
                },
            )

            assert response.status_code == 404

    def test_get_training_info_success(self, app_with_training_router):
        """Test getting training info"""
        client = TestClient(app_with_training_router)

        with patch("src.routers.v1.training_router.TrainingService") as mock_service_class:
            mock_service = Mock()
            mock_service.get_model_info_by_name.return_value = {
                "model_name": "latency_ann_60",
                "model_version": "1",
                "last_training_time": 1234567890.0,
                "training_loss": 0.05,
                "samples_used": 1000,
                "features_used": 22,
                "run_id": "abc123",
            }
            mock_service_class.return_value = mock_service

            response = client.get("/api/v1/training/latency_ann_60")

            assert response.status_code == 200
            data = response.json()
            assert data["model_name"] == "latency_ann_60"
            assert data["model_version"] == "1"
            assert data["training_loss"] == 0.05
            mock_service.get_model_info_by_name.assert_called_once_with("latency_ann_60")
