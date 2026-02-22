"""
Tests for ExpiryService sweep logic.
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from src.services.expiry_service import ExpiryService
from src.schemas.expiry import ExpiryPolicy, ExpiryPolicyUpdate


def _mock_model(name, last_used_at=None, use_count="0", protected=False, creation_ts=None):
    """Helper to build a mock registered model."""
    rm = Mock()
    rm.name = name
    rm.tags = {"use_count": use_count}
    if last_used_at is not None:
        rm.tags["last_used_at"] = str(last_used_at)
    if protected:
        rm.tags["expiry_protected"] = "true"
    rm.creation_timestamp = (creation_ts or 0) * 1000
    return rm


@pytest.fixture
def mock_ml_interface():
    ml = Mock()
    ml.is_mlflow_connected.return_value = True
    return ml


@pytest.fixture
def expiry_service(mock_ml_interface):
    return ExpiryService(mock_ml_interface)


class TestExpiryPolicy:

    def test_default_policy_values(self, expiry_service):
        policy = expiry_service.get_policy()
        assert policy.base_ttl_days == 30
        assert policy.tolerance_per_use_days == 1
        assert policy.max_tolerance_days == 90

    def test_partial_update_changes_only_supplied_fields(self, expiry_service):
        update = ExpiryPolicyUpdate(base_ttl_days=7)
        updated = expiry_service.update_policy(update)

        assert updated.base_ttl_days == 7
        assert updated.tolerance_per_use_days == 1  # unchanged
        assert updated.max_tolerance_days == 90     # unchanged

    def test_replace_policy_replaces_all_fields(self, expiry_service):
        new_policy = ExpiryPolicy(
            base_ttl_days=5,
            tolerance_per_use_days=2,
            max_tolerance_days=20,
            sweep_interval_seconds=600,
        )
        result = expiry_service.replace_policy(new_policy)

        assert result.base_ttl_days == 5
        assert result.tolerance_per_use_days == 2
        assert result.max_tolerance_days == 20


class TestExpirySweep:

    def test_skips_protected_models(self, expiry_service):
        old_ts = 0.0  # epoch — definitely expired
        protected = _mock_model("protected_model", last_used_at=old_ts, protected=True)

        with patch("src.services.expiry_service.MlflowClient") as MockClient, \
             patch("src.services.expiry_service.ModelService") as MockModelService:

            MockClient.return_value.search_registered_models.return_value = [protected]

            deleted = expiry_service.run_sweep()

        assert deleted == 0
        MockModelService.return_value.delete_model_instance.assert_not_called()

    def test_deletes_expired_model(self, expiry_service):
        old_ts = 0.0  # epoch — way past any TTL
        expired = _mock_model("old_cell_model", last_used_at=old_ts)

        with patch("src.services.expiry_service.MlflowClient") as MockClient, \
             patch("src.services.expiry_service.ModelService") as MockModelService:

            MockClient.return_value.search_registered_models.return_value = [expired]
            mock_svc = MockModelService.return_value

            deleted = expiry_service.run_sweep()

        assert deleted == 1
        mock_svc.delete_model_instance.assert_called_once_with("old_cell_model")

    def test_keeps_recently_used_model(self, expiry_service):
        now = datetime.now(timezone.utc).timestamp()
        recent = _mock_model("active_model", last_used_at=now)

        with patch("src.services.expiry_service.MlflowClient") as MockClient, \
             patch("src.services.expiry_service.ModelService") as MockModelService:

            MockClient.return_value.search_registered_models.return_value = [recent]

            deleted = expiry_service.run_sweep()

        assert deleted == 0
        MockModelService.return_value.delete_model_instance.assert_not_called()

    def test_tolerance_extends_ttl_per_use(self, expiry_service):
        # Model last used 35 days ago — expired under base TTL of 30
        # but use_count=10 adds 10 * 1 = 10 days → effective TTL 40 → still alive
        days_ago_35 = datetime.now(timezone.utc).timestamp() - (35 * 86400)
        model = _mock_model("busy_model", last_used_at=days_ago_35, use_count="10")

        with patch("src.services.expiry_service.MlflowClient") as MockClient, \
             patch("src.services.expiry_service.ModelService") as MockModelService:

            MockClient.return_value.search_registered_models.return_value = [model]

            deleted = expiry_service.run_sweep()

        assert deleted == 0

    def test_tolerance_is_capped_by_max_tolerance(self, expiry_service):
        # Model last used 130 days ago
        # use_count=1000 → tolerance would be 1000d but max is 90d → effective TTL = 120d
        # 130 > 120 → should be deleted
        days_ago_130 = datetime.now(timezone.utc).timestamp() - (130 * 86400)
        model = _mock_model("high_use_old_model", last_used_at=days_ago_130, use_count="1000")

        with patch("src.services.expiry_service.MlflowClient") as MockClient, \
             patch("src.services.expiry_service.ModelService") as MockModelService:

            MockClient.return_value.search_registered_models.return_value = [model]
            mock_svc = MockModelService.return_value

            deleted = expiry_service.run_sweep()

        assert deleted == 1
        mock_svc.delete_model_instance.assert_called_once_with("high_use_old_model")

    def test_falls_back_to_creation_timestamp_when_no_tags(self, expiry_service):
        # No last_used_at tag — falls back to creation_timestamp (epoch = very old)
        model = _mock_model("untagged_model", creation_ts=0.0)

        with patch("src.services.expiry_service.MlflowClient") as MockClient, \
             patch("src.services.expiry_service.ModelService") as MockModelService:

            MockClient.return_value.search_registered_models.return_value = [model]
            mock_svc = MockModelService.return_value

            deleted = expiry_service.run_sweep()

        assert deleted == 1
        mock_svc.delete_model_instance.assert_called_once_with("untagged_model")

    def test_mlflow_not_connected_skips_sweep(self, expiry_service, mock_ml_interface):
        mock_ml_interface.is_mlflow_connected.return_value = False

        with patch("src.services.expiry_service.MlflowClient") as MockClient:
            deleted = expiry_service.run_sweep()

        assert deleted == 0
        MockClient.assert_not_called()

    def test_individual_model_error_does_not_abort_sweep(self, expiry_service):
        old_ts = 0.0
        model_a = _mock_model("model_a", last_used_at=old_ts)
        model_b = _mock_model("model_b", last_used_at=old_ts)

        with patch("src.services.expiry_service.MlflowClient") as MockClient, \
             patch("src.services.expiry_service.ModelService") as MockModelService:

            MockClient.return_value.search_registered_models.return_value = [model_a, model_b]
            mock_svc = MockModelService.return_value
            # First deletion raises, second should still proceed
            mock_svc.delete_model_instance.side_effect = [RuntimeError("oops"), None]

            deleted = expiry_service.run_sweep()

        assert deleted == 1  # only the second one counted
