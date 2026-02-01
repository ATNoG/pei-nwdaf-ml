"""Unit tests for model router endpoints."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from datetime import datetime

from src.schemas.model import ArchitectureType, ModelConfig, ModelDetail, ModelSummary


@pytest.fixture
def client():
    """FastAPI test client with mocked dependencies."""
    # Mock the database initialization to prevent hanging
    with patch("src.db.database.init_db"):
        from main import app

        # Create test client
        test_client = TestClient(app)
        yield test_client


@pytest.fixture
def mock_mlflow_service():
    """Mock MLflowService for dependency injection."""
    return MagicMock()


class TestModelRouterEndpoints:
    """Tests for model router endpoints."""

    def test_list_models(self, client, mock_mlflow_service):
        """Test GET /v1/models endpoint."""
        # Setup mock
        mock_mlflow_service.list_models.return_value = [
            ModelSummary(
                id="uuid-1",
                name="model_one",
                architecture=ArchitectureType.LSTM,
                created_at=datetime(2024, 2, 1),
                latest_version=1,
            )
        ]

        # Override dependency
        from src.core.dependencies import get_mlflow_client
        from main import app

        def override_get_mlflow_client():
            yield mock_mlflow_service

        app.dependency_overrides[get_mlflow_client] = override_get_mlflow_client

        try:
            # Make request
            response = client.get("/v1/models")

            # Verify
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["name"] == "model_one"
            assert data[0]["architecture"] == "lstm"
            mock_mlflow_service.list_models.assert_called_once()
        finally:
            app.dependency_overrides.clear()

    def test_create_model(self, client, mock_mlflow_service):
        """Test POST /v1/models endpoint."""
        # Setup mock
        mock_mlflow_service.create_model.return_value = ModelDetail(
            id="uuid-123",
            name="test_model",
            config=ModelConfig(
                architecture=ArchitectureType.LSTM,
                input_fields=["latency_mean"],
                output_fields=["latency_mean"],
                window_duration_seconds=60,
                lookback_steps=30,
                forecast_steps=5,
                hidden_size=64,
            ),
            created_at=datetime(2024, 2, 1),
            latest_version=None,
            last_trained_at=None,
            mlflow_run_id=None,
            training_loss=None,
        )

        # Override dependency
        from src.core.dependencies import get_mlflow_client
        from main import app

        def override_get_mlflow_client():
            yield mock_mlflow_service

        app.dependency_overrides[get_mlflow_client] = override_get_mlflow_client

        try:
            # Make request
            payload = {
                "name": "test_model",
                "config": {
                    "architecture": "lstm",
                    "input_fields": ["latency_mean"],
                    "output_fields": ["latency_mean"],
                    "window_duration_seconds": 60,
                    "lookback_steps": 30,
                    "forecast_steps": 5,
                    "hidden_size": 64,
                }
            }
            response = client.post("/v1/models", json=payload)

            # Verify
            assert response.status_code == 201
            data = response.json()
            assert data["id"] == "uuid-123"
            assert data["name"] == "test_model"
            mock_mlflow_service.create_model.assert_called_once()
        finally:
            app.dependency_overrides.clear()

    def test_create_model_error(self, client, mock_mlflow_service):
        """Test POST /v1/models with error."""
        # Setup mock to raise ValueError
        mock_mlflow_service.create_model.side_effect = ValueError("Model already exists")

        # Override dependency
        from src.core.dependencies import get_mlflow_client
        from main import app

        def override_get_mlflow_client():
            yield mock_mlflow_service

        app.dependency_overrides[get_mlflow_client] = override_get_mlflow_client

        try:
            # Make request
            payload = {
                "name": "test_model",
                "config": {
                    "architecture": "lstm",
                    "input_fields": ["latency_mean"],
                    "output_fields": ["latency_mean"],
                    "window_duration_seconds": 60,
                    "lookback_steps": 30,
                    "forecast_steps": 5,
                    "hidden_size": 64,
                }
            }
            response = client.post("/v1/models", json=payload)

            # Verify
            assert response.status_code == 400
            assert "already exists" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_get_model_by_id(self, client, mock_mlflow_service):
        """Test GET /v1/models/{model_id} endpoint."""
        # Setup mock
        mock_mlflow_service.get_model.return_value = ModelDetail(
            id="uuid-123",
            name="test_model",
            config=ModelConfig(
                architecture=ArchitectureType.LSTM,
                input_fields=["latency_mean"],
                output_fields=["latency_mean"],
                window_duration_seconds=60,
                lookback_steps=30,
                forecast_steps=5,
                hidden_size=64,
            ),
            created_at=datetime(2024, 2, 1),
            latest_version=None,
            last_trained_at=None,
            mlflow_run_id=None,
            training_loss=None,
        )

        # Override dependency
        from src.core.dependencies import get_mlflow_client
        from main import app

        def override_get_mlflow_client():
            yield mock_mlflow_service

        app.dependency_overrides[get_mlflow_client] = override_get_mlflow_client

        try:
            # Make request
            response = client.get("/v1/models/uuid-123")

            # Verify
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "uuid-123"
            assert data["name"] == "test_model"
            mock_mlflow_service.get_model.assert_called_once_with("uuid-123")
        finally:
            app.dependency_overrides.clear()

    def test_get_model_not_found(self, client, mock_mlflow_service):
        """Test GET /v1/models/{model_id} with non-existent model."""
        # Setup mock to raise ValueError
        mock_mlflow_service.get_model.side_effect = ValueError("Model not found")

        # Override dependency
        from src.core.dependencies import get_mlflow_client
        from main import app

        def override_get_mlflow_client():
            yield mock_mlflow_service

        app.dependency_overrides[get_mlflow_client] = override_get_mlflow_client

        try:
            # Make request
            response = client.get("/v1/models/nonexistent-uuid")

            # Verify
            assert response.status_code == 404
            assert "not found" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_delete_model(self, client, mock_mlflow_service):
        """Test DELETE /v1/models/{model_id} endpoint."""
        # Setup mock (delete returns None)
        mock_mlflow_service.delete_model.return_value = None

        # Override dependency
        from src.core.dependencies import get_mlflow_client
        from main import app

        def override_get_mlflow_client():
            yield mock_mlflow_service

        app.dependency_overrides[get_mlflow_client] = override_get_mlflow_client

        try:
            # Make request
            response = client.delete("/v1/models/uuid-123")

            # Verify
            assert response.status_code == 204
            assert response.content == b""  # No content
            mock_mlflow_service.delete_model.assert_called_once_with("uuid-123")
        finally:
            app.dependency_overrides.clear()

    def test_delete_model_not_found(self, client, mock_mlflow_service):
        """Test DELETE /v1/models/{model_id} with non-existent model."""
        # Setup mock to raise ValueError
        mock_mlflow_service.delete_model.side_effect = ValueError("Model not found")

        # Override dependency
        from src.core.dependencies import get_mlflow_client
        from main import app

        def override_get_mlflow_client():
            yield mock_mlflow_service

        app.dependency_overrides[get_mlflow_client] = override_get_mlflow_client

        try:
            # Make request
            response = client.delete("/v1/models/nonexistent-uuid")

            # Verify
            assert response.status_code == 404
            assert "not found" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()
