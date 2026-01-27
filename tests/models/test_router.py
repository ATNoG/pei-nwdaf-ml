"""
Router tests for model and training endpoints
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock

from src.routers.v1 import model_router, training_router


@pytest.fixture
def app():
    """FastAPI app with routers"""
    app = FastAPI()
    app.include_router(model_router.router, prefix="/api/v1/model", tags=["model"])
    app.include_router(training_router.router, prefix="/api/v1/training", tags=["training"])
    return app


@pytest.fixture
def client(app):
    """Test client with mocked dependencies"""
    mock_interface = Mock()
    mock_interface.is_mlflow_connected = Mock(return_value=True)
    app.state.ml_interface = mock_interface
    yield TestClient(app)


class TestModelRouter:
    """Test model router endpoints"""

    def test_create_model(self, client):
        """Test POST /api/v1/model/instance"""
        with patch("src.routers.v1.model_router.ModelService") as mock_service_class:
            mock_service = Mock()
            mock_service.create_model_instance.return_value = None
            mock_service_class.return_value = mock_service

            response = client.post(
                "/api/v1/model/instance",
                json={
                    "analytics_type": "latency",
                    "horizon": 60,
                    "model_type": "ann",
                    "name": "test_ann"
                }
            )

            assert response.status_code == 200
            assert response.json()["model_name"] == "test_ann"
            mock_service.create_model_instance.assert_called_once()

    def test_get_model_details(self, client):
        """Test GET /api/v1/model/instance/{model_name}"""
        with patch("src.services.model_service.MlflowClient") as mock_client_class:
            mock_client = MagicMock()
            mock_model = Mock()
            mock_model.tags = {"model_type": "ann", "analytics_type": "latency", "horizon": "60"}
            mock_client.get_registered_model.return_value = mock_model

            mock_version = Mock()
            mock_version.version = "1"
            mock_version.current_stage = "Production"
            mock_version.run_id = "run123"
            mock_version.creation_timestamp = 1640000000000
            mock_client.search_model_versions.return_value = [mock_version]
            mock_client_class.return_value = mock_client

            with patch("src.services.model_service.mlflow.get_run") as mock_get_run:
                mock_run = Mock()
                mock_run.data.metrics = {"training_loss": 0.05}
                mock_get_run.return_value = mock_run

                response = client.get("/api/v1/model/instance/test_ann")
                assert response.status_code == 200

    def test_delete_model(self, client):
        """Test DELETE /api/v1/model/instance/{model_name}"""
        with patch("src.services.model_service.MlflowClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_registered_model.return_value = Mock()
            mock_client_class.return_value = mock_client

            response = client.delete("/api/v1/model/instance/test_ann")
            assert response.status_code == 200
            assert response.json()["status"] == "deleted"


class TestTrainingRouter:
    """Test training router endpoints"""

    def test_start_training(self, client):
        """Test POST /api/v1/training"""
        with patch("src.services.training_service.MlflowClient") as mock_client_class:
            mock_client = MagicMock()
            mock_model = Mock()
            mock_model.tags = {"model_type": "ann", "analytics_type": "latency", "horizon": "60"}
            mock_client.get_registered_model.return_value = mock_model
            mock_client_class.return_value = mock_client

            response = client.post(
                "/api/v1/training",
                json={"model_name": "test_ann"}
            )

            assert response.status_code == 200
            assert response.json()["status"] == "training_started"

    def test_get_training_info(self, client):
        """Test GET /api/v1/training/{model_name}"""
        with patch("src.services.training_service.MlflowClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_registered_model.return_value = Mock()

            mock_version = Mock()
            mock_version.version = "1"
            mock_version.run_id = "run456"
            mock_version.creation_timestamp = 1640000000000
            mock_client.search_model_versions.return_value = [mock_version]
            mock_client_class.return_value = mock_client

            with patch("src.services.training_service.mlflow.get_run") as mock_get_run:
                mock_run = Mock()
                mock_run.data.metrics = {"training_loss": 0.03}
                mock_run.data.params = {"n_samples": "100", "n_features": "10"}
                mock_get_run.return_value = mock_run

                response = client.get("/api/v1/training/test_ann")
                assert response.status_code == 200
