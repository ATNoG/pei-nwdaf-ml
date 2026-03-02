"""Unit tests for performance router endpoints."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

from src.schemas.model import ArchitectureType
from src.schemas.performance import FieldEvaluationResponse, ModelPerformance


@pytest.fixture
def client():
    """FastAPI test client with mocked database init."""
    with patch("src.db.database.init_db"):
        from main import app

        test_client = TestClient(app)
        yield test_client


@pytest.fixture
def mock_performance_service():
    """Mock PerformanceService for dependency injection."""
    mock = MagicMock()
    mock.evaluate_field = AsyncMock()
    mock.monitor_best_model = AsyncMock()
    return mock


def _make_model_performance(
    model_id: str,
    model_name: str,
    score: float | None = None,
    metric: str = "rmse",
    is_best: bool = False,
    latest_version: int | None = 1,
    evaluated_at: datetime | None = None,
) -> ModelPerformance:
    """Helper to build a ModelPerformance object."""
    return ModelPerformance(
        model_id=model_id,
        model_name=model_name,
        architecture=ArchitectureType.LSTM,
        latest_version=latest_version,
        training_loss=0.05,
        score=score,
        metric=metric,
        is_best=is_best,
        last_trained_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
        evaluated_at=evaluated_at or (
            datetime(2024, 2, 2, tzinfo=timezone.utc) if score is not None else None
        ),
    )


def _make_field_response(
    field_name: str = "latency_mean",
    models: list[ModelPerformance] | None = None,
    best_model_id: str | None = None,
) -> FieldEvaluationResponse:
    """Helper to build a FieldEvaluationResponse."""
    return FieldEvaluationResponse(
        field_name=field_name,
        models=models or [],
        best_model_id=best_model_id,
        last_evaluated_at=datetime(2024, 2, 2, tzinfo=timezone.utc) if models else None,
    )


class TestEvaluateField:
    """Tests for POST /v1/performance/{field_name}/evaluate."""

    def test_evaluate_multiple_models_returns_ranked_list(
        self, client, mock_performance_service
    ):
        """Models are sorted by score ascending; best_model_id is the winner."""
        best = _make_model_performance("uuid-best", "model_best", score=0.05, is_best=True)
        worse = _make_model_performance("uuid-worse", "model_worse", score=0.20, is_best=False)

        mock_performance_service.evaluate_field.return_value = _make_field_response(
            models=[best, worse], best_model_id="uuid-best"
        )

        from src.routers.v1.performance_router import get_performance_service
        from main import app

        app.dependency_overrides[get_performance_service] = lambda: mock_performance_service

        try:
            response = client.post("/v1/performance/latency_mean/evaluate")

            assert response.status_code == 200
            data = response.json()
            assert data["field_name"] == "latency_mean"
            assert data["best_model_id"] == "uuid-best"
            assert len(data["models"]) == 2
            assert data["models"][0]["model_id"] == "uuid-best"
            assert data["models"][0]["is_best"] is True
            assert data["models"][0]["score"] == pytest.approx(0.05)
            assert data["models"][1]["model_id"] == "uuid-worse"
            assert data["models"][1]["is_best"] is False
            # Router passes metric.value (default "rmse") to the service
            mock_performance_service.evaluate_field.assert_called_once_with("latency_mean", "rmse")
        finally:
            app.dependency_overrides.clear()

    def test_evaluate_with_explicit_metric(self, client, mock_performance_service):
        """metric query param is forwarded to the service."""
        best = _make_model_performance("uuid-best", "model_best", score=0.05, metric="mae", is_best=True)

        mock_performance_service.evaluate_field.return_value = _make_field_response(
            models=[best], best_model_id="uuid-best"
        )

        from src.routers.v1.performance_router import get_performance_service
        from main import app

        app.dependency_overrides[get_performance_service] = lambda: mock_performance_service

        try:
            response = client.post("/v1/performance/latency_mean/evaluate?metric=mae")

            assert response.status_code == 200
            mock_performance_service.evaluate_field.assert_called_once_with("latency_mean", "mae")
        finally:
            app.dependency_overrides.clear()

    def test_evaluate_mix_of_trained_and_untrained(
        self, client, mock_performance_service
    ):
        """Untrained models appear last with score=None and is_best=False."""
        scored = _make_model_performance("uuid-scored", "model_a", score=0.10, is_best=True)
        unscored = _make_model_performance(
            "uuid-unscored", "model_b", score=None, is_best=False, latest_version=None
        )

        mock_performance_service.evaluate_field.return_value = _make_field_response(
            models=[scored, unscored], best_model_id="uuid-scored"
        )

        from src.routers.v1.performance_router import get_performance_service
        from main import app

        app.dependency_overrides[get_performance_service] = lambda: mock_performance_service

        try:
            response = client.post("/v1/performance/latency_mean/evaluate")

            assert response.status_code == 200
            data = response.json()
            assert data["models"][0]["score"] == pytest.approx(0.10)
            assert data["models"][1]["score"] is None
            assert data["models"][1]["is_best"] is False
        finally:
            app.dependency_overrides.clear()

    def test_evaluate_no_matching_models_returns_empty(
        self, client, mock_performance_service
    ):
        """When no model predicts the field, returns an empty models list."""
        mock_performance_service.evaluate_field.return_value = _make_field_response(
            field_name="unknown_field"
        )

        from src.routers.v1.performance_router import get_performance_service
        from main import app

        app.dependency_overrides[get_performance_service] = lambda: mock_performance_service

        try:
            response = client.post("/v1/performance/unknown_field/evaluate")

            assert response.status_code == 200
            data = response.json()
            assert data["models"] == []
            assert data["best_model_id"] is None
        finally:
            app.dependency_overrides.clear()

    def test_evaluate_all_models_unscored_returns_200_no_best(
        self, client, mock_performance_service
    ):
        """When no valid cells exist, all models get score=None and best_model_id is null."""
        unscored_a = _make_model_performance("uuid-a", "model_a", score=None, is_best=False)
        unscored_b = _make_model_performance("uuid-b", "model_b", score=None, is_best=False)

        mock_performance_service.evaluate_field.return_value = _make_field_response(
            models=[unscored_a, unscored_b], best_model_id=None
        )

        from src.routers.v1.performance_router import get_performance_service
        from main import app

        app.dependency_overrides[get_performance_service] = lambda: mock_performance_service

        try:
            response = client.post("/v1/performance/latency_mean/evaluate")

            assert response.status_code == 200
            data = response.json()
            assert data["best_model_id"] is None
            assert all(m["score"] is None for m in data["models"])
            assert all(not m["is_best"] for m in data["models"])
        finally:
            app.dependency_overrides.clear()

    def test_evaluate_service_error_returns_500(
        self, client, mock_performance_service
    ):
        """Unexpected service exceptions map to HTTP 500."""
        mock_performance_service.evaluate_field.side_effect = Exception("MLflow unreachable")

        from src.routers.v1.performance_router import get_performance_service
        from main import app

        app.dependency_overrides[get_performance_service] = lambda: mock_performance_service

        try:
            response = client.post("/v1/performance/latency_mean/evaluate")

            assert response.status_code == 500
            assert "MLflow unreachable" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()


class TestGetCachedEvaluation:
    """Tests for GET /v1/performance/{field_name}."""

    def test_get_cached_returns_stored_scores(self, client, mock_performance_service):
        """GET returns last evaluation result without triggering evaluate_field."""
        best = _make_model_performance("uuid-best", "model_best", score=0.07, is_best=True)
        mock_performance_service.get_cached_evaluation.return_value = _make_field_response(
            models=[best], best_model_id="uuid-best"
        )

        from src.routers.v1.performance_router import get_performance_service
        from main import app

        app.dependency_overrides[get_performance_service] = lambda: mock_performance_service

        try:
            response = client.get("/v1/performance/latency_mean")

            assert response.status_code == 200
            data = response.json()
            assert data["field_name"] == "latency_mean"
            assert data["best_model_id"] == "uuid-best"
            mock_performance_service.evaluate_field.assert_not_called()
            mock_performance_service.get_cached_evaluation.assert_called_once_with("latency_mean")
        finally:
            app.dependency_overrides.clear()

    def test_get_cached_service_error_returns_500(self, client, mock_performance_service):
        """Service exceptions map to HTTP 500."""
        mock_performance_service.get_cached_evaluation.side_effect = Exception("DB error")

        from src.routers.v1.performance_router import get_performance_service
        from main import app

        app.dependency_overrides[get_performance_service] = lambda: mock_performance_service

        try:
            response = client.get("/v1/performance/latency_mean")

            assert response.status_code == 500
            assert "DB error" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()


class TestGetBestModel:
    """Tests for GET /v1/performance/{field_name}/best."""

    def test_get_best_returns_model(self, client, mock_performance_service):
        """Returns the ModelPerformance of the tagged best model."""
        best = _make_model_performance("uuid-best", "model_best", score=0.03, is_best=True)
        mock_performance_service.get_best_model.return_value = best

        from src.routers.v1.performance_router import get_performance_service
        from main import app

        app.dependency_overrides[get_performance_service] = lambda: mock_performance_service

        try:
            response = client.get("/v1/performance/latency_mean/best")

            assert response.status_code == 200
            data = response.json()
            assert data["model_id"] == "uuid-best"
            assert data["is_best"] is True
            assert data["score"] == pytest.approx(0.03)
            mock_performance_service.get_best_model.assert_called_once_with("latency_mean")
        finally:
            app.dependency_overrides.clear()

    def test_get_best_no_evaluation_returns_404(self, client, mock_performance_service):
        """Returns 404 when no best model has been designated."""
        mock_performance_service.get_best_model.return_value = None

        from src.routers.v1.performance_router import get_performance_service
        from main import app

        app.dependency_overrides[get_performance_service] = lambda: mock_performance_service

        try:
            response = client.get("/v1/performance/latency_mean/best")

            assert response.status_code == 404
            assert "latency_mean" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()


class TestMonitorBestModel:
    """Tests for POST /v1/performance/{field_name}/monitor."""

    def test_monitor_returns_updated_performance(
        self, client, mock_performance_service
    ):
        """Returns updated ModelPerformance with refreshed score."""
        updated = _make_model_performance(
            "uuid-best",
            "model_best",
            score=0.04,
            is_best=True,
            evaluated_at=datetime(2024, 2, 3, tzinfo=timezone.utc),
        )
        mock_performance_service.monitor_best_model.return_value = updated

        from src.routers.v1.performance_router import get_performance_service
        from main import app

        app.dependency_overrides[get_performance_service] = lambda: mock_performance_service

        try:
            response = client.post("/v1/performance/latency_mean/monitor")

            assert response.status_code == 200
            data = response.json()
            assert data["model_id"] == "uuid-best"
            assert data["is_best"] is True
            assert data["score"] == pytest.approx(0.04)
            # Router always passes trigger="monitor" to distinguish from auto_monitor
            mock_performance_service.monitor_best_model.assert_called_once_with(
                "latency_mean", trigger="monitor"
            )
            mock_performance_service.evaluate_field.assert_not_called()
        finally:
            app.dependency_overrides.clear()

    def test_monitor_no_best_model_returns_422(
        self, client, mock_performance_service
    ):
        """Returns 422 when no best model has been designated yet."""
        mock_performance_service.monitor_best_model.side_effect = ValueError(
            "No best model designated for field 'latency_mean'. "
            "Run POST /performance/latency_mean/evaluate first."
        )

        from src.routers.v1.performance_router import get_performance_service
        from main import app

        app.dependency_overrides[get_performance_service] = lambda: mock_performance_service

        try:
            response = client.post("/v1/performance/latency_mean/monitor")

            assert response.status_code == 422
            assert "No best model designated" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_monitor_service_error_returns_500(
        self, client, mock_performance_service
    ):
        """Unexpected service exceptions map to HTTP 500."""
        mock_performance_service.monitor_best_model.side_effect = Exception(
            "MLflow connection failed"
        )

        from src.routers.v1.performance_router import get_performance_service
        from main import app

        app.dependency_overrides[get_performance_service] = lambda: mock_performance_service

        try:
            response = client.post("/v1/performance/latency_mean/monitor")

            assert response.status_code == 500
            assert "MLflow connection failed" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()


class TestGetModelsWithOutputField:
    """Tests for GET /v1/models?output_field={field}."""

    def test_filter_by_output_field_returns_matching_models(
        self, client
    ):
        """Only models with the queried field in output_fields are returned."""
        from src.schemas.model import ModelSummary
        from src.core.dependencies import get_mlflow_client
        from main import app

        mock_mlflow_service = MagicMock()
        mock_mlflow_service.list_models.return_value = [
            ModelSummary(
                id="uuid-1",
                name="model_a",
                architecture=ArchitectureType.LSTM,
                created_at=datetime(2024, 2, 1),
                latest_version=1,
            ),
            ModelSummary(
                id="uuid-2",
                name="model_b",
                architecture=ArchitectureType.ANN,
                created_at=datetime(2024, 2, 1),
                latest_version=None,
            ),
        ]

        from src.db.model_config import ModelConfigDB

        config_1 = MagicMock(spec=ModelConfigDB)
        config_1.model_id = "uuid-1"
        config_1.output_fields = ["latency_mean", "jitter"]

        config_2 = MagicMock(spec=ModelConfigDB)
        config_2.model_id = "uuid-2"
        config_2.output_fields = ["jitter"]

        mock_mlflow_service.ml_config_service.list_all.return_value = [config_1, config_2]

        def override_get_mlflow_client():
            yield mock_mlflow_service

        app.dependency_overrides[get_mlflow_client] = override_get_mlflow_client

        try:
            response = client.get("/v1/models?output_field=latency_mean")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["id"] == "uuid-1"
            assert data[0]["name"] == "model_a"
        finally:
            app.dependency_overrides.clear()

    def test_no_output_field_filter_returns_all_models(self, client):
        """Without output_field param, all models are returned as before."""
        from src.schemas.model import ModelSummary
        from src.core.dependencies import get_mlflow_client
        from main import app

        mock_mlflow_service = MagicMock()
        mock_mlflow_service.list_models.return_value = [
            ModelSummary(
                id="uuid-1",
                name="model_a",
                architecture=ArchitectureType.LSTM,
                created_at=datetime(2024, 2, 1),
                latest_version=1,
            ),
            ModelSummary(
                id="uuid-2",
                name="model_b",
                architecture=ArchitectureType.ANN,
                created_at=datetime(2024, 2, 1),
                latest_version=None,
            ),
        ]

        def override_get_mlflow_client():
            yield mock_mlflow_service

        app.dependency_overrides[get_mlflow_client] = override_get_mlflow_client

        try:
            response = client.get("/v1/models")

            assert response.status_code == 200
            assert len(response.json()) == 2
            # list_all should NOT have been called (no filter applied)
            mock_mlflow_service.ml_config_service.list_all.assert_not_called()
        finally:
            app.dependency_overrides.clear()

    def test_output_field_no_matches_returns_empty(self, client):
        """Filter that matches no models returns an empty list."""
        from src.schemas.model import ModelSummary
        from src.core.dependencies import get_mlflow_client
        from main import app

        mock_mlflow_service = MagicMock()
        mock_mlflow_service.list_models.return_value = [
            ModelSummary(
                id="uuid-1",
                name="model_a",
                architecture=ArchitectureType.LSTM,
                created_at=datetime(2024, 2, 1),
                latest_version=1,
            ),
        ]

        from src.db.model_config import ModelConfigDB

        config_1 = MagicMock(spec=ModelConfigDB)
        config_1.model_id = "uuid-1"
        config_1.output_fields = ["jitter"]

        mock_mlflow_service.ml_config_service.list_all.return_value = [config_1]

        def override_get_mlflow_client():
            yield mock_mlflow_service

        app.dependency_overrides[get_mlflow_client] = override_get_mlflow_client

        try:
            response = client.get("/v1/models?output_field=latency_mean")

            assert response.status_code == 200
            assert response.json() == []
        finally:
            app.dependency_overrides.clear()
