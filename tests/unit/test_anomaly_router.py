"""Unit tests for anomaly router endpoints."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
from datetime import datetime

from src.schemas.anomaly import (
    AnomalyModelConfig,
    AnomalyModelDetail,
    AnomalyModelSummary,
)


@pytest.fixture
def client():
    """FastAPI test client with mocked dependencies."""
    with patch("src.db.database.init_db"):
        from main import app

        test_client = TestClient(app)
        yield test_client


@pytest.fixture
def mock_anomaly_config_service():
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_anomaly_training_service():
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_anomaly_detection_service():
    mock = MagicMock()
    mock.detect = AsyncMock()
    return mock


@pytest.fixture
def mock_data_storage_client():
    mock = MagicMock()
    mock.validate_fields = AsyncMock(return_value=(True, []))
    mock.get_available_fields = AsyncMock(return_value=["latency_mean", "rsrp_mean"])
    return mock


class TestAnomalyModelEndpoints:
    """Tests for anomaly model CRUD endpoints."""

    def test_list_models_empty(self, client, mock_anomaly_config_service):
        """Test GET /v1/anomaly/models returns empty list."""
        from main import app
        from src.routers.v1.anomaly_router import get_anomaly_config_service

        mock_anomaly_config_service.list_all.return_value = []

        app.dependency_overrides[get_anomaly_config_service] = (
            lambda: mock_anomaly_config_service
        )

        try:
            response = client.get("/v1/anomaly/models")
            assert response.status_code == 200
            assert response.json() == []
        finally:
            app.dependency_overrides.clear()

    def test_create_model(self, client, mock_data_storage_client):
        """Test POST /v1/anomaly/models creates a model."""
        from main import app
        from src.routers.v1.anomaly_router import get_anomaly_config_service
        from src.services.data_storage_client import DataStorageClient

        mock_config_svc = MagicMock()

        # Mock MLflow client (imported inside endpoint function body)
        with patch("mlflow.MlflowClient") as MockMlflow:
            mock_mlflow_instance = MagicMock()
            MockMlflow.return_value = mock_mlflow_instance

            app.dependency_overrides[get_anomaly_config_service] = (
                lambda: mock_config_svc
            )
            app.dependency_overrides[DataStorageClient] = (
                lambda: mock_data_storage_client
            )

            try:
                payload = {
                    "name": "test-anomaly",
                    "config": {
                        "input_fields": ["latency_mean"],
                        "window_duration_seconds": 60,
                        "hidden_size": 32,
                        "threshold_percentile": 95.0,
                    },
                }
                response = client.post("/v1/anomaly/models", json=payload)

                assert response.status_code == 201
                data = response.json()
                assert data["name"] == "test-anomaly"
                assert data["threshold_value"] is None
                assert data["config"]["input_fields"] == ["latency_mean"]
                mock_config_svc.create.assert_called_once()
            finally:
                app.dependency_overrides.clear()

    def test_create_model_invalid_fields(self, client, mock_data_storage_client):
        """Test POST /v1/anomaly/models with invalid fields returns 400."""
        from main import app
        from src.routers.v1.anomaly_router import get_anomaly_config_service
        from src.services.data_storage_client import DataStorageClient

        mock_data_storage_client.validate_fields = AsyncMock(
            return_value=(False, ["bad_field"])
        )

        app.dependency_overrides[get_anomaly_config_service] = lambda: MagicMock()
        app.dependency_overrides[DataStorageClient] = lambda: mock_data_storage_client

        try:
            payload = {
                "name": "test-anomaly",
                "config": {
                    "input_fields": ["bad_field"],
                    "window_duration_seconds": 60,
                },
            }
            response = client.post("/v1/anomaly/models", json=payload)

            assert response.status_code == 400
            assert "invalid_fields" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_get_model_not_found(self, client):
        """Test GET /v1/anomaly/models/{id} with non-existent model."""
        from main import app
        from src.routers.v1.anomaly_router import get_anomaly_config_service

        mock_svc = MagicMock()
        mock_svc.get_config.return_value = None

        app.dependency_overrides[get_anomaly_config_service] = lambda: mock_svc

        try:
            response = client.get("/v1/anomaly/models/nonexistent")
            assert response.status_code == 404
        finally:
            app.dependency_overrides.clear()

    def test_delete_model(self, client):
        """Test DELETE /v1/anomaly/models/{id}."""
        from main import app
        from src.routers.v1.anomaly_router import get_anomaly_config_service

        mock_svc = MagicMock()

        with patch("mlflow.MlflowClient"):
            app.dependency_overrides[get_anomaly_config_service] = lambda: mock_svc

            try:
                response = client.delete("/v1/anomaly/models/uuid-1")
                assert response.status_code == 204
                mock_svc.delete_model.assert_called_once_with("uuid-1")
            finally:
                app.dependency_overrides.clear()

    def test_delete_model_not_found(self, client):
        """Test DELETE /v1/anomaly/models/{id} when model doesn't exist."""
        from main import app
        from src.routers.v1.anomaly_router import get_anomaly_config_service

        mock_svc = MagicMock()
        mock_svc.delete_model.side_effect = ValueError("not found")

        app.dependency_overrides[get_anomaly_config_service] = lambda: mock_svc

        try:
            response = client.delete("/v1/anomaly/models/nonexistent")
            assert response.status_code == 404
        finally:
            app.dependency_overrides.clear()


class TestAnomalyTrainingEndpoints:
    """Tests for anomaly training endpoints."""

    def test_train_model(self, client, mock_anomaly_training_service):
        """Test POST /v1/anomaly/training/train."""
        from main import app
        from src.routers.v1.anomaly_router import get_anomaly_training_service

        mock_anomaly_training_service.create_training_job.return_value = {
            "job_id": "job-1",
            "model_id": "model-1",
            "status": "queued",
            "created_at": datetime(2024, 1, 1),
        }

        app.dependency_overrides[get_anomaly_training_service] = (
            lambda: mock_anomaly_training_service
        )

        try:
            payload = {"model_id": "model-1", "lookback_seconds": 86400}
            response = client.post("/v1/anomaly/training/train", json=payload)

            assert response.status_code == 202
            data = response.json()
            assert data["job_id"] == "job-1"
            assert data["status"] == "queued"
        finally:
            app.dependency_overrides.clear()

    def test_list_training_jobs_empty(self, client, mock_anomaly_training_service):
        """Test GET /v1/anomaly/training/jobs returns empty list."""
        from main import app
        from src.routers.v1.anomaly_router import get_anomaly_training_service

        mock_anomaly_training_service.list_jobs.return_value = []

        app.dependency_overrides[get_anomaly_training_service] = (
            lambda: mock_anomaly_training_service
        )

        try:
            response = client.get("/v1/anomaly/training/jobs")
            assert response.status_code == 200
            assert response.json() == []
        finally:
            app.dependency_overrides.clear()

    def test_get_training_job_not_found(self, client, mock_anomaly_training_service):
        """Test GET /v1/anomaly/training/jobs/{id} with non-existent job."""
        from main import app
        from src.routers.v1.anomaly_router import get_anomaly_training_service

        mock_anomaly_training_service.get_job.return_value = None

        app.dependency_overrides[get_anomaly_training_service] = (
            lambda: mock_anomaly_training_service
        )

        try:
            response = client.get("/v1/anomaly/training/jobs/nonexistent")
            assert response.status_code == 404
        finally:
            app.dependency_overrides.clear()

    def test_cancel_training_job(self, client, mock_anomaly_training_service):
        """Test DELETE /v1/anomaly/training/jobs/{id}."""
        from main import app
        from src.routers.v1.anomaly_router import get_anomaly_training_service

        app.dependency_overrides[get_anomaly_training_service] = (
            lambda: mock_anomaly_training_service
        )

        try:
            response = client.delete("/v1/anomaly/training/jobs/job-1")
            assert response.status_code == 204
            mock_anomaly_training_service.cancel_job.assert_called_once_with("job-1")
        finally:
            app.dependency_overrides.clear()

    def test_cancel_job_not_found(self, client, mock_anomaly_training_service):
        """Test DELETE /v1/anomaly/training/jobs/{id} when not found."""
        from main import app
        from src.routers.v1.anomaly_router import get_anomaly_training_service

        mock_anomaly_training_service.cancel_job.side_effect = ValueError("not found")

        app.dependency_overrides[get_anomaly_training_service] = (
            lambda: mock_anomaly_training_service
        )

        try:
            response = client.delete("/v1/anomaly/training/jobs/nonexistent")
            assert response.status_code == 404
        finally:
            app.dependency_overrides.clear()


class TestAnomalyDetectionEndpoint:
    """Tests for anomaly detection endpoint."""

    def test_detect(self, client, mock_anomaly_detection_service):
        """Test POST /v1/anomaly/detect."""
        from main import app
        from src.routers.v1.anomaly_router import get_anomaly_detection_service
        from src.schemas.anomaly import AnomalyDetectionResult

        mock_anomaly_detection_service.detect.return_value = AnomalyDetectionResult(
            model_id="model-1",
            model_name="test",
            cell_id=5,
            threshold_value=0.5,
            window_duration_seconds=60,
            input_fields=["latency_mean"],
            results=[],
        )

        app.dependency_overrides[get_anomaly_detection_service] = (
            lambda: mock_anomaly_detection_service
        )

        try:
            payload = {"cell_id": 5, "model_id": "model-1"}
            response = client.post("/v1/anomaly/detect", json=payload)

            assert response.status_code == 200
            data = response.json()
            assert data["model_id"] == "model-1"
            assert data["cell_id"] == 5
            assert data["results"] == []
        finally:
            app.dependency_overrides.clear()

    def test_detect_model_not_found(self, client, mock_anomaly_detection_service):
        """Test POST /v1/anomaly/detect with non-existent model."""
        from main import app
        from src.routers.v1.anomaly_router import get_anomaly_detection_service

        mock_anomaly_detection_service.detect.side_effect = ValueError("not found")

        app.dependency_overrides[get_anomaly_detection_service] = (
            lambda: mock_anomaly_detection_service
        )

        try:
            payload = {"cell_id": 5, "model_id": "nonexistent"}
            response = client.post("/v1/anomaly/detect", json=payload)

            assert response.status_code == 404
        finally:
            app.dependency_overrides.clear()
