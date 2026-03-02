"""Unit tests for ExpiryService."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.db.expiry_audit import ExpiryAuditDB
from src.db.model_config import ModelConfigDB
from src.schemas.expiry import ExpiryPolicy, ExpiryPolicyUpdate
from src.services.expiry_service import ExpiryService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_config(model_id="uuid-1", name="test_model", created_days_ago=0):
    """Return a ModelConfigDB mock with created_at offset from now."""
    cfg = MagicMock(spec=ModelConfigDB)
    cfg.model_id = model_id
    cfg.name = name
    cfg.created_at = datetime.now(timezone.utc) - timedelta(days=created_days_ago)
    return cfg


def _make_mlflow_client(last_run_end_time=None):
    """Return a MlflowClient mock, optionally with a latest run end time."""
    client = MagicMock()
    rm = MagicMock()
    if last_run_end_time is not None:
        mv = MagicMock()
        mv.run_id = "run-123"
        rm.latest_versions = [mv]
        run = MagicMock()
        run.info.end_time = int(last_run_end_time.timestamp() * 1000)
        client.get_run.return_value = run
    else:
        rm.latest_versions = []
    client.get_registered_model.return_value = rm
    return client


def _mock_db_context(mock_db):
    """Return a get_db_context patch that yields mock_db."""
    @contextmanager
    def _ctx():
        yield mock_db
    return _ctx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def expiry_service():
    """ExpiryService with no policy in DB → seeds from env defaults."""
    with patch("src.services.expiry_service._load_policy_from_db", return_value=None), \
         patch("src.services.expiry_service._save_policy_to_db"):
        return ExpiryService()


# ---------------------------------------------------------------------------
# Policy management
# ---------------------------------------------------------------------------

class TestExpiryServicePolicy:
    """Tests for get_policy / update_policy / replace_policy."""

    def test_get_policy_returns_defaults_when_no_db_entry(self):
        """Service seeds from env defaults when no policy row exists in DB."""
        with patch("src.services.expiry_service._load_policy_from_db", return_value=None), \
             patch("src.services.expiry_service._save_policy_to_db"):
            svc = ExpiryService()

        policy = svc.get_policy()
        assert policy.base_ttl_days == 30
        assert policy.tolerance_per_use_days == 1
        assert policy.max_tolerance_days == 90
        assert policy.sweep_interval_seconds == 3600

    def test_get_policy_loads_from_db_when_row_exists(self):
        """Service uses persisted policy when a DB row is found."""
        stored = ExpiryPolicy(
            base_ttl_days=7,
            tolerance_per_use_days=2,
            max_tolerance_days=30,
            sweep_interval_seconds=120,
        )
        with patch("src.services.expiry_service._load_policy_from_db", return_value=stored), \
             patch("src.services.expiry_service._save_policy_to_db"):
            svc = ExpiryService()

        policy = svc.get_policy()
        assert policy.base_ttl_days == 7
        assert policy.sweep_interval_seconds == 120

    def test_seeds_default_policy_to_db_when_no_row(self):
        """Default policy is persisted to DB on first startup."""
        with patch("src.services.expiry_service._load_policy_from_db", return_value=None), \
             patch("src.services.expiry_service._save_policy_to_db") as mock_save:
            ExpiryService()

        mock_save.assert_called_once()

    def test_update_policy_changes_specified_field_only(self, expiry_service):
        """PATCH: only the supplied field changes; others are preserved."""
        with patch("src.services.expiry_service._save_policy_to_db"):
            result = expiry_service.update_policy(ExpiryPolicyUpdate(base_ttl_days=5))

        assert result.base_ttl_days == 5
        assert result.tolerance_per_use_days == 1    # unchanged
        assert result.max_tolerance_days == 90       # unchanged
        assert result.sweep_interval_seconds == 3600  # unchanged

    def test_update_policy_persists_to_db(self, expiry_service):
        """PATCH: updated policy is saved to the database."""
        with patch("src.services.expiry_service._save_policy_to_db") as mock_save:
            expiry_service.update_policy(ExpiryPolicyUpdate(base_ttl_days=10))

        mock_save.assert_called_once()

    def test_update_policy_reflects_in_get_policy(self, expiry_service):
        """PATCH: subsequent get_policy returns the updated value."""
        with patch("src.services.expiry_service._save_policy_to_db"):
            expiry_service.update_policy(ExpiryPolicyUpdate(base_ttl_days=5))

        assert expiry_service.get_policy().base_ttl_days == 5

    def test_replace_policy_sets_all_fields(self, expiry_service):
        """PUT: all fields are replaced with the supplied policy."""
        new_policy = ExpiryPolicy(
            base_ttl_days=14,
            tolerance_per_use_days=3,
            max_tolerance_days=60,
            sweep_interval_seconds=7200,
        )
        with patch("src.services.expiry_service._save_policy_to_db"):
            result = expiry_service.replace_policy(new_policy)

        assert result.base_ttl_days == 14
        assert result.tolerance_per_use_days == 3
        assert result.max_tolerance_days == 60
        assert result.sweep_interval_seconds == 7200

    def test_replace_policy_persists_to_db(self, expiry_service):
        """PUT: replaced policy is saved to the database."""
        new_policy = ExpiryPolicy(
            base_ttl_days=14,
            tolerance_per_use_days=3,
            max_tolerance_days=60,
            sweep_interval_seconds=7200,
        )
        with patch("src.services.expiry_service._save_policy_to_db") as mock_save:
            expiry_service.replace_policy(new_policy)

        mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# Sweep logic
# ---------------------------------------------------------------------------

class TestExpiryServiceSweep:
    """Tests for run_sweep()."""

    def test_sweep_no_models_returns_zero(self, expiry_service):
        """Sweep with no model configs deletes nothing and returns 0."""
        mock_db = MagicMock()
        mock_db.query.return_value.all.return_value = []

        with patch("src.services.expiry_service.get_db_context", _mock_db_context(mock_db)), \
             patch("src.services.expiry_service.MlflowClient"):
            result = expiry_service.run_sweep()

        assert result == 0

    def test_sweep_fresh_model_not_deleted(self, expiry_service):
        """Model created today is not expired under the default TTL of 30 days."""
        cfg = _make_model_config(created_days_ago=1)
        mock_db = MagicMock()
        mock_db.query.return_value.all.return_value = [cfg]

        with patch("src.services.expiry_service.get_db_context", _mock_db_context(mock_db)), \
             patch("src.services.expiry_service.MlflowClient", return_value=_make_mlflow_client()):
            result = expiry_service.run_sweep()

        assert result == 0
        mock_db.delete.assert_not_called()

    def test_sweep_expired_model_deleted(self, expiry_service):
        """Model created 31 days ago is expired and removed under TTL=30."""
        cfg = _make_model_config(created_days_ago=31)
        mock_db = MagicMock()
        mock_db.query.return_value.all.return_value = [cfg]
        mlflow_client = _make_mlflow_client()

        with patch("src.services.expiry_service.get_db_context", _mock_db_context(mock_db)), \
             patch("src.services.expiry_service.MlflowClient", return_value=mlflow_client):
            result = expiry_service.run_sweep()

        assert result == 1
        mlflow_client.delete_registered_model.assert_called_once_with(cfg.model_id)
        mock_db.delete.assert_called_once_with(cfg)
        mock_db.commit.assert_called_once()

    def test_sweep_writes_audit_record_for_deleted_model(self, expiry_service):
        """An ExpiryAuditDB row is written for every expired model."""
        cfg = _make_model_config(created_days_ago=31)
        mock_db = MagicMock()
        mock_db.query.return_value.all.return_value = [cfg]

        with patch("src.services.expiry_service.get_db_context", _mock_db_context(mock_db)), \
             patch("src.services.expiry_service.MlflowClient", return_value=_make_mlflow_client()):
            expiry_service.run_sweep()

        mock_db.add.assert_called_once()
        audit_arg = mock_db.add.call_args[0][0]
        assert isinstance(audit_arg, ExpiryAuditDB)
        assert audit_arg.model_id == cfg.model_id
        assert audit_arg.model_name == cfg.name

    def test_sweep_uses_last_run_time_over_created_at(self, expiry_service):
        """last_activity comes from the latest MLflow run, not created_at."""
        # created 60 days ago but trained 1 day ago → should NOT expire with TTL=30
        cfg = _make_model_config(created_days_ago=60)
        mock_db = MagicMock()
        mock_db.query.return_value.all.return_value = [cfg]

        recent = datetime.now(timezone.utc) - timedelta(days=1)
        mlflow_client = _make_mlflow_client(last_run_end_time=recent)

        with patch("src.services.expiry_service.get_db_context", _mock_db_context(mock_db)), \
             patch("src.services.expiry_service.MlflowClient", return_value=mlflow_client):
            result = expiry_service.run_sweep()

        assert result == 0
        mlflow_client.delete_registered_model.assert_not_called()

    def test_sweep_falls_back_to_created_at_when_no_run(self, expiry_service):
        """When no MLflow run exists, created_at is used as last_activity."""
        # No run, created 31 days ago → expires with TTL=30
        cfg = _make_model_config(created_days_ago=31)
        mock_db = MagicMock()
        mock_db.query.return_value.all.return_value = [cfg]

        # MlflowClient returns a model with no versions
        mlflow_client = _make_mlflow_client(last_run_end_time=None)

        with patch("src.services.expiry_service.get_db_context", _mock_db_context(mock_db)), \
             patch("src.services.expiry_service.MlflowClient", return_value=mlflow_client):
            result = expiry_service.run_sweep()

        assert result == 1

    def test_sweep_handles_mlflow_delete_failure(self, expiry_service):
        """MLflow delete failure does not abort the sweep; DB record is still removed."""
        cfg = _make_model_config(created_days_ago=31)
        mock_db = MagicMock()
        mock_db.query.return_value.all.return_value = [cfg]

        mlflow_client = _make_mlflow_client()
        mlflow_client.delete_registered_model.side_effect = Exception("MLflow unavailable")

        with patch("src.services.expiry_service.get_db_context", _mock_db_context(mock_db)), \
             patch("src.services.expiry_service.MlflowClient", return_value=mlflow_client):
            result = expiry_service.run_sweep()

        assert result == 1
        mock_db.delete.assert_called_once_with(cfg)

    def test_sweep_counts_only_expired_models(self, expiry_service):
        """Sweep returns the count of actually expired models, not total models."""
        expired1 = _make_model_config("id-1", "old_1", created_days_ago=35)
        expired2 = _make_model_config("id-2", "old_2", created_days_ago=40)
        fresh = _make_model_config("id-3", "fresh", created_days_ago=5)

        mock_db = MagicMock()
        mock_db.query.return_value.all.return_value = [expired1, expired2, fresh]

        with patch("src.services.expiry_service.get_db_context", _mock_db_context(mock_db)), \
             patch("src.services.expiry_service.MlflowClient", return_value=_make_mlflow_client()):
            result = expiry_service.run_sweep()

        assert result == 2

    def test_sweep_audit_record_contains_policy_snapshot(self, expiry_service):
        """Audit record captures a snapshot of the policy at sweep time."""
        cfg = _make_model_config(created_days_ago=31)
        mock_db = MagicMock()
        mock_db.query.return_value.all.return_value = [cfg]

        with patch("src.services.expiry_service.get_db_context", _mock_db_context(mock_db)), \
             patch("src.services.expiry_service.MlflowClient", return_value=_make_mlflow_client()):
            expiry_service.run_sweep()

        audit_arg = mock_db.add.call_args[0][0]
        assert "base_ttl_days" in audit_arg.policy_snapshot
        assert audit_arg.policy_snapshot["base_ttl_days"] == 30
