"""Unit tests for expiry router endpoints."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.db.expiry_audit import ExpiryAuditDB
from src.schemas.expiry import ExpiryPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_expiry_service():
    """Mock ExpiryService with a default policy pre-configured."""
    svc = MagicMock()
    svc.get_policy.return_value = ExpiryPolicy(
        base_ttl_days=30,
        tolerance_per_use_days=1,
        max_tolerance_days=90,
        sweep_interval_seconds=3600,
    )
    return svc


@pytest.fixture
def client(mock_expiry_service):
    """TestClient with DB init mocked and expiry service injected into app.state."""
    with patch("src.db.database.init_db"):
        from main import app
        app.state.expiry_service = mock_expiry_service
        yield TestClient(app)


def _make_audit_row(model_id="uuid-1", model_name="old_model"):
    """Build a mock ExpiryAuditDB row with sensible defaults."""
    row = MagicMock()
    row.model_id = model_id
    row.model_name = model_name
    row.expired_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    row.last_used_at = datetime(2023, 12, 1, tzinfo=timezone.utc)
    row.effective_ttl_days = 30
    row.policy_snapshot = {"base_ttl_days": 30}
    return row


# ---------------------------------------------------------------------------
# Policy endpoints
# ---------------------------------------------------------------------------

class TestExpiryRouterPolicy:
    """Tests for GET/PATCH/PUT /v1/expiry/policy."""

    def test_get_policy_returns_current_policy(self, client, mock_expiry_service):
        """GET /v1/expiry/policy returns the active policy."""
        response = client.get("/v1/expiry/policy")

        assert response.status_code == 200
        data = response.json()
        assert data["base_ttl_days"] == 30
        assert data["tolerance_per_use_days"] == 1
        assert data["max_tolerance_days"] == 90
        assert data["sweep_interval_seconds"] == 3600
        mock_expiry_service.get_policy.assert_called_once()

    def test_patch_policy_applies_partial_update(self, client, mock_expiry_service):
        """PATCH /v1/expiry/policy calls update_policy with the supplied fields."""
        updated = ExpiryPolicy(
            base_ttl_days=7,
            tolerance_per_use_days=1,
            max_tolerance_days=90,
            sweep_interval_seconds=3600,
        )
        mock_expiry_service.update_policy.return_value = updated

        response = client.patch("/v1/expiry/policy", json={"base_ttl_days": 7})

        assert response.status_code == 200
        assert response.json()["base_ttl_days"] == 7
        mock_expiry_service.update_policy.assert_called_once()

    def test_patch_policy_rejects_base_ttl_below_minimum(self, client):
        """PATCH /v1/expiry/policy returns 422 when base_ttl_days < 1."""
        response = client.patch("/v1/expiry/policy", json={"base_ttl_days": 0})

        assert response.status_code == 422

    def test_patch_policy_rejects_sweep_interval_below_minimum(self, client):
        """PATCH /v1/expiry/policy returns 422 when sweep_interval_seconds < 60."""
        response = client.patch("/v1/expiry/policy", json={"sweep_interval_seconds": 10})

        assert response.status_code == 422

    def test_patch_policy_empty_body_is_accepted(self, client, mock_expiry_service):
        """PATCH with an empty body is valid (no-op update)."""
        mock_expiry_service.update_policy.return_value = mock_expiry_service.get_policy.return_value

        response = client.patch("/v1/expiry/policy", json={})

        assert response.status_code == 200

    def test_put_policy_replaces_all_fields(self, client, mock_expiry_service):
        """PUT /v1/expiry/policy calls replace_policy with the full body."""
        new_policy = ExpiryPolicy(
            base_ttl_days=14,
            tolerance_per_use_days=2,
            max_tolerance_days=60,
            sweep_interval_seconds=7200,
        )
        mock_expiry_service.replace_policy.return_value = new_policy

        response = client.put(
            "/v1/expiry/policy",
            json={
                "base_ttl_days": 14,
                "tolerance_per_use_days": 2,
                "max_tolerance_days": 60,
                "sweep_interval_seconds": 7200,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["base_ttl_days"] == 14
        assert data["sweep_interval_seconds"] == 7200
        mock_expiry_service.replace_policy.assert_called_once()

    def test_put_policy_rejects_invalid_field_value(self, client):
        """PUT /v1/expiry/policy returns 422 when a field violates its constraint."""
        response = client.put("/v1/expiry/policy", json={"base_ttl_days": 0})

        assert response.status_code == 422

    def test_policy_endpoint_returns_503_when_service_unavailable(self, client):
        """GET /v1/expiry/policy returns 503 when expiry_service is not on app.state."""
        from main import app

        original = app.state.expiry_service
        del app.state.expiry_service
        try:
            response = client.get("/v1/expiry/policy")
            assert response.status_code == 503
        finally:
            app.state.expiry_service = original


# ---------------------------------------------------------------------------
# Sweep endpoint
# ---------------------------------------------------------------------------

class TestExpiryRouterSweep:
    """Tests for POST /v1/expiry/sweep."""

    def test_sweep_returns_deleted_count(self, client, mock_expiry_service):
        """POST /v1/expiry/sweep returns the number of models the sweep deleted."""
        mock_expiry_service.run_sweep.return_value = 3

        response = client.post("/v1/expiry/sweep")

        assert response.status_code == 200
        assert response.json() == {"deleted": 3}
        mock_expiry_service.run_sweep.assert_called_once()

    def test_sweep_returns_zero_when_nothing_expired(self, client, mock_expiry_service):
        """POST /v1/expiry/sweep returns 0 when no models are expired."""
        mock_expiry_service.run_sweep.return_value = 0

        response = client.post("/v1/expiry/sweep")

        assert response.status_code == 200
        assert response.json()["deleted"] == 0

    def test_sweep_returns_500_on_service_error(self, client, mock_expiry_service):
        """POST /v1/expiry/sweep returns 500 when the sweep raises an exception."""
        mock_expiry_service.run_sweep.side_effect = RuntimeError("DB connection lost")

        response = client.post("/v1/expiry/sweep")

        assert response.status_code == 500
        assert "DB connection lost" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Audit endpoint
# ---------------------------------------------------------------------------

class TestExpiryRouterAudit:
    """Tests for GET /v1/expiry/audit."""

    def test_get_audit_returns_empty_list(self, client):
        """GET /v1/expiry/audit returns [] when no entries exist."""
        from src.db.database import get_db
        from main import app

        mock_db = MagicMock()
        mock_db.query.return_value.order_by.return_value \
            .offset.return_value.limit.return_value.all.return_value = []

        app.dependency_overrides[get_db] = lambda: mock_db
        try:
            response = client.get("/v1/expiry/audit")
            assert response.status_code == 200
            assert response.json() == []
        finally:
            app.dependency_overrides.clear()

    def test_get_audit_returns_entries(self, client):
        """GET /v1/expiry/audit returns all audit rows."""
        from src.db.database import get_db
        from main import app

        rows = [_make_audit_row("uuid-1", "model_alpha"), _make_audit_row("uuid-2", "model_beta")]
        mock_db = MagicMock()
        mock_db.query.return_value.order_by.return_value \
            .offset.return_value.limit.return_value.all.return_value = rows

        app.dependency_overrides[get_db] = lambda: mock_db
        try:
            response = client.get("/v1/expiry/audit")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["model_id"] == "uuid-1"
            assert data[0]["model_name"] == "model_alpha"
            assert data[1]["model_id"] == "uuid-2"
        finally:
            app.dependency_overrides.clear()

    def test_get_audit_applies_limit_and_offset(self, client):
        """GET /v1/expiry/audit passes limit and offset query params to the DB query."""
        from src.db.database import get_db
        from main import app

        mock_db = MagicMock()
        chain = mock_db.query.return_value.order_by.return_value
        chain.offset.return_value.limit.return_value.all.return_value = []

        app.dependency_overrides[get_db] = lambda: mock_db
        try:
            client.get("/v1/expiry/audit?limit=10&offset=20")
            chain.offset.assert_called_once_with(20)
            chain.offset.return_value.limit.assert_called_once_with(10)
        finally:
            app.dependency_overrides.clear()

    def test_get_audit_default_limit_is_50(self, client):
        """GET /v1/expiry/audit uses limit=50 when not specified."""
        from src.db.database import get_db
        from main import app

        mock_db = MagicMock()
        chain = mock_db.query.return_value.order_by.return_value
        chain.offset.return_value.limit.return_value.all.return_value = []

        app.dependency_overrides[get_db] = lambda: mock_db
        try:
            client.get("/v1/expiry/audit")
            chain.offset.return_value.limit.assert_called_once_with(50)
        finally:
            app.dependency_overrides.clear()

    def test_get_audit_entry_fields(self, client):
        """GET /v1/expiry/audit response includes all AuditEntry fields."""
        from src.db.database import get_db
        from main import app

        rows = [_make_audit_row("uuid-99", "expiry_test_model")]
        mock_db = MagicMock()
        mock_db.query.return_value.order_by.return_value \
            .offset.return_value.limit.return_value.all.return_value = rows

        app.dependency_overrides[get_db] = lambda: mock_db
        try:
            response = client.get("/v1/expiry/audit")
            entry = response.json()[0]
            assert entry["model_id"] == "uuid-99"
            assert entry["model_name"] == "expiry_test_model"
            assert entry["effective_ttl_days"] == 30
            assert "expired_at" in entry
            assert "last_used_at" in entry
            assert "policy_snapshot" in entry
        finally:
            app.dependency_overrides.clear()
