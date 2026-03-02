"""Unit tests for inference router endpoints."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

from src.schemas.model import ArchitectureType
from src.schemas.inference import ForecastStepPrediction


@pytest.fixture
def client():
    """FastAPI test client with mocked database init."""
    with patch("src.db.database.init_db"):
        from main import app

        test_client = TestClient(app)
        yield test_client


@pytest.fixture
def mock_inference_service():
    """Mock InferenceService for dependency injection."""
    mock = MagicMock()
    mock.predict = AsyncMock()
    return mock


def _make_success_result():
    """Helper to create a valid inference result dict."""
    return {
        "model_id": "test-uuid",
        "model_name": "test_model",
        "model_version": 2,
        "architecture": ArchitectureType.LSTM,
        "cell_id": 5,
        "lookback_steps": 30,
        "forecast_steps": 3,
        "window_duration_seconds": 60,
        "input_fields": ["rsrp_mean", "sinr_mean"],
        "output_fields": ["latency_mean"],
        "predictions": [
            ForecastStepPrediction(step=1, values={"latency_mean": 10.0}),
            ForecastStepPrediction(step=2, values={"latency_mean": 11.0}),
            ForecastStepPrediction(step=3, values={"latency_mean": 12.0}),
        ],
    }


class TestInferenceRouter:
    """Tests for POST /v1/inference endpoint."""

    def test_success_200(self, client, mock_inference_service):
        """Test successful inference returns 200 with full result."""
        mock_inference_service.predict.return_value = _make_success_result()

        from src.routers.v1.inference_router import get_inference_service
        from main import app

        app.dependency_overrides[get_inference_service] = (
            lambda: mock_inference_service
        )

        try:
            response = client.post(
                "/v1/inference",
                json={"output_field": "latency_mean", "model_id": "test-uuid", "cell_id": 5},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["model_id"] == "test-uuid"
            assert data["model_name"] == "test_model"
            assert data["model_version"] == 2
            assert data["architecture"] == "lstm"
            assert data["cell_id"] == 5
            assert len(data["predictions"]) == 3
            assert data["predictions"][0]["step"] == 1
            assert data["predictions"][0]["values"]["latency_mean"] == 10.0
            mock_inference_service.predict.assert_called_once_with(
                output_field="latency_mean", cell_id=5, model_id="test-uuid"
            )
        finally:
            app.dependency_overrides.clear()

    def test_model_not_found_422(self, client, mock_inference_service):
        """Test that ValueError from service maps to 422."""
        mock_inference_service.predict.side_effect = ValueError("Model not found")

        from src.routers.v1.inference_router import get_inference_service
        from main import app

        app.dependency_overrides[get_inference_service] = (
            lambda: mock_inference_service
        )

        try:
            response = client.post(
                "/v1/inference",
                json={"output_field": "latency_mean", "model_id": "bad-uuid", "cell_id": 0},
            )

            assert response.status_code == 422
            assert "Model not found" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_model_not_trained_422(self, client, mock_inference_service):
        """Test that untrained model returns 422."""
        mock_inference_service.predict.side_effect = ValueError(
            "no trained versions"
        )

        from src.routers.v1.inference_router import get_inference_service
        from main import app

        app.dependency_overrides[get_inference_service] = (
            lambda: mock_inference_service
        )

        try:
            response = client.post(
                "/v1/inference",
                json={"output_field": "latency_mean", "model_id": "test-uuid", "cell_id": 0},
            )

            assert response.status_code == 422
            assert "no trained versions" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_prediction_failure_500(self, client, mock_inference_service):
        """Test that RuntimeError from service maps to 500."""
        mock_inference_service.predict.side_effect = RuntimeError(
            "Prediction failed: CUDA error"
        )

        from src.routers.v1.inference_router import get_inference_service
        from main import app

        app.dependency_overrides[get_inference_service] = (
            lambda: mock_inference_service
        )

        try:
            response = client.post(
                "/v1/inference",
                json={"output_field": "latency_mean", "model_id": "test-uuid", "cell_id": 5},
            )

            assert response.status_code == 500
            assert "Prediction failed" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_unexpected_error_500(self, client, mock_inference_service):
        """Test that unexpected Exception maps to 500."""
        mock_inference_service.predict.side_effect = Exception("unexpected error")

        from src.routers.v1.inference_router import get_inference_service
        from main import app

        app.dependency_overrides[get_inference_service] = (
            lambda: mock_inference_service
        )

        try:
            response = client.post(
                "/v1/inference",
                json={"output_field": "latency_mean", "model_id": "test-uuid", "cell_id": 5},
            )

            assert response.status_code == 500
            assert "Inference failed" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_model_id_optional_accepted(self, client, mock_inference_service):
        """Test that omitting model_id is valid — service selects best model."""
        mock_inference_service.predict.return_value = _make_success_result()

        from src.routers.v1.inference_router import get_inference_service
        from main import app

        app.dependency_overrides[get_inference_service] = lambda: mock_inference_service

        try:
            response = client.post(
                "/v1/inference",
                json={"output_field": "latency_mean", "cell_id": 5},
            )

            assert response.status_code == 200
            mock_inference_service.predict.assert_called_once_with(
                output_field="latency_mean", cell_id=5, model_id=None
            )
        finally:
            app.dependency_overrides.clear()

    def test_negative_cell_id_422(self, client):
        """Test that negative cell_id fails FastAPI validation."""
        response = client.post(
            "/v1/inference",
            json={"output_field": "latency_mean", "model_id": "test-uuid", "cell_id": -1},
        )

        assert response.status_code == 422

    def test_missing_cell_id_422(self, client):
        """Test that missing cell_id fails FastAPI validation."""
        response = client.post(
            "/v1/inference",
            json={"output_field": "latency_mean", "model_id": "test-uuid"},
        )

        assert response.status_code == 422

    def test_missing_output_field_422(self, client):
        """Test that missing output_field fails FastAPI validation."""
        response = client.post(
            "/v1/inference",
            json={"model_id": "test-uuid", "cell_id": 5},
        )

        assert response.status_code == 422
