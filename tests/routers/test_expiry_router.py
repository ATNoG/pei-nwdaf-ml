"""
Tests for the expiry router endpoints.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.routers.v1 import expiry_router
from src.schemas.expiry import ExpiryPolicy, ExpiryPolicyUpdate


@pytest.fixture
def mock_expiry_service():
    service = Mock()
    service.get_policy.return_value = ExpiryPolicy(
        base_ttl_days=30,
        tolerance_per_use_days=1,
        max_tolerance_days=90,
        sweep_interval_seconds=3600,
    )
    service.run_sweep.return_value = 0
    return service


@pytest.fixture
def app_with_expiry_router(mock_expiry_service):
    app = FastAPI()
    app.include_router(expiry_router.router, prefix="/api/v1/expiry", tags=["expiry"])
    app.state.expiry_service = mock_expiry_service
    return app


class TestExpiryRouter:

    def test_get_policy_returns_defaults(self, app_with_expiry_router, mock_expiry_service):
        client = TestClient(app_with_expiry_router)
        response = client.get("/api/v1/expiry/policy")

        assert response.status_code == 200
        data = response.json()
        assert data["base_ttl_days"] == 30
        assert data["tolerance_per_use_days"] == 1
        assert data["max_tolerance_days"] == 90
        assert data["sweep_interval_seconds"] == 3600

    def test_patch_policy_partial_update(self, app_with_expiry_router, mock_expiry_service):
        updated_policy = ExpiryPolicy(
            base_ttl_days=14,
            tolerance_per_use_days=1,
            max_tolerance_days=90,
            sweep_interval_seconds=3600,
        )
        mock_expiry_service.update_policy.return_value = updated_policy

        client = TestClient(app_with_expiry_router)
        response = client.patch(
            "/api/v1/expiry/policy",
            json={"base_ttl_days": 14},
        )

        assert response.status_code == 200
        assert response.json()["base_ttl_days"] == 14
        mock_expiry_service.update_policy.assert_called_once()

    def test_put_policy_full_replace(self, app_with_expiry_router, mock_expiry_service):
        new_policy = ExpiryPolicy(
            base_ttl_days=7,
            tolerance_per_use_days=2,
            max_tolerance_days=60,
            sweep_interval_seconds=1800,
        )
        mock_expiry_service.replace_policy.return_value = new_policy

        client = TestClient(app_with_expiry_router)
        response = client.put(
            "/api/v1/expiry/policy",
            json={
                "base_ttl_days": 7,
                "tolerance_per_use_days": 2,
                "max_tolerance_days": 60,
                "sweep_interval_seconds": 1800,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["base_ttl_days"] == 7
        assert data["tolerance_per_use_days"] == 2
        mock_expiry_service.replace_policy.assert_called_once()

    def test_trigger_sweep_returns_count(self, app_with_expiry_router, mock_expiry_service):
        mock_expiry_service.run_sweep.return_value = 3

        client = TestClient(app_with_expiry_router)
        response = client.post("/api/v1/expiry/sweep")

        assert response.status_code == 200
        assert response.json()["deleted"] == 3
        mock_expiry_service.run_sweep.assert_called_once()

    def test_trigger_sweep_service_error(self, app_with_expiry_router, mock_expiry_service):
        mock_expiry_service.run_sweep.side_effect = RuntimeError("MLflow unavailable")

        client = TestClient(app_with_expiry_router)
        response = client.post("/api/v1/expiry/sweep")

        assert response.status_code == 500

    def test_no_expiry_service_returns_503(self):
        app = FastAPI()
        app.include_router(expiry_router.router, prefix="/api/v1/expiry", tags=["expiry"])
        # intentionally no app.state.expiry_service

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/expiry/policy")
        assert response.status_code == 503
