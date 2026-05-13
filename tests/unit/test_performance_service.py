"""Unit tests for PerformanceService pure and near-pure methods."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch
from mlflow.exceptions import MlflowException

from src.schemas.model import ModelConfig
from src.schemas.performance import ModelPerformance
from src.services.performance_service import PerformanceService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mlflow_service():
    mock = MagicMock()
    mock.client = MagicMock()
    return mock


@pytest.fixture
def mock_ml_config_service():
    return MagicMock()


@pytest.fixture
def mock_data_storage_client():
    return MagicMock()


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.query.return_value = db
    db.filter.return_value = db
    db.order_by.return_value = db
    db.all.return_value = []
    return db


@pytest.fixture
def service(mock_mlflow_service, mock_ml_config_service, mock_data_storage_client, mock_db):
    return PerformanceService(
        mlflow_service=mock_mlflow_service,
        ml_config_service=mock_ml_config_service,
        data_storage_client=mock_data_storage_client,
        db=mock_db,
    )


def _make_perf(model_id: str, score: float | None, metric: str = "rmse") -> ModelPerformance:
    return ModelPerformance(
        model_id=model_id,
        model_name=f"model_{model_id}",
        architecture="lstm",
        score=score,
        metric=metric,
    )


# ---------------------------------------------------------------------------
# _score_is_worse - lower-is-better metrics (rmse, mae, mape)
# ---------------------------------------------------------------------------


class TestScoreIsWorseLowerIsBetter:
    """For rmse/mae/mape: worse when current > baseline * factor."""

    def test_clearly_worse(self, service):
        assert service._score_is_worse(2.0, 1.0, "rmse", 1.5) is True

    def test_clearly_better(self, service):
        assert service._score_is_worse(0.5, 1.0, "rmse", 1.5) is False

    def test_exactly_at_threshold_is_not_worse(self, service):
        # current == baseline * factor → at boundary → not worse
        assert service._score_is_worse(1.5, 1.0, "rmse", 1.5) is False

    def test_same_as_baseline_is_not_worse(self, service):
        assert service._score_is_worse(1.0, 1.0, "rmse", 1.5) is False

    def test_just_above_threshold_is_worse(self, service):
        assert service._score_is_worse(1.51, 1.0, "rmse", 1.5) is True

    @pytest.mark.parametrize("metric", ["mae", "mape"])
    def test_same_direction_as_rmse(self, service, metric):
        """mae and mape hit the same code branch as rmse."""
        assert service._score_is_worse(2.0, 1.0, metric, 1.5) is True
        assert service._score_is_worse(0.5, 1.0, metric, 1.5) is False


# ---------------------------------------------------------------------------
# _score_is_worse - higher-is-better metric (r2)
# ---------------------------------------------------------------------------


class TestScoreIsWorseR2:
    """For r2: worse when current < baseline / factor."""

    def test_clearly_worse(self, service):
        # baseline=0.9, factor=1.2 → threshold=0.75; current=0.5 → worse
        assert service._score_is_worse(0.5, 0.9, "r2", 1.2) is True

    def test_clearly_better(self, service):
        assert service._score_is_worse(0.85, 0.9, "r2", 1.2) is False

    def test_exactly_at_threshold_is_not_worse(self, service):
        # current == baseline / factor → at boundary → not worse
        assert service._score_is_worse(0.75, 0.9, "r2", 1.2) is False

    def test_same_as_baseline_is_not_worse(self, service):
        assert service._score_is_worse(0.9, 0.9, "r2", 1.2) is False

    def test_just_below_threshold_is_worse(self, service):
        assert service._score_is_worse(0.74, 0.9, "r2", 1.2) is True


# ---------------------------------------------------------------------------
# _compute_score - RMSE
# ---------------------------------------------------------------------------


class TestComputeScoreRmse:
    def test_perfect_prediction(self, service):
        assert service._compute_score([1.0, 2.0], [1.0, 2.0], "rmse") == pytest.approx(0.0)

    def test_unit_error(self, service):
        # errors = [1.0] → sqrt(mean([1.0])) = 1.0
        assert service._compute_score([2.0], [1.0], "rmse") == pytest.approx(1.0)

    def test_multiple_errors(self, service):
        # errors = [2, 4] → sqrt(mean([4, 16])) = sqrt(10)
        assert service._compute_score([3.0, 5.0], [1.0, 1.0], "rmse") == pytest.approx(
            10.0 ** 0.5
        )

    def test_symmetric_errors(self, service):
        # errors = [1, -1] → sqrt(mean([1, 1])) = 1.0
        assert service._compute_score([2.0, 0.0], [1.0, 1.0], "rmse") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _compute_score - MAE
# ---------------------------------------------------------------------------


class TestComputeScoreMae:
    def test_perfect_prediction(self, service):
        assert service._compute_score([1.0, 2.0], [1.0, 2.0], "mae") == pytest.approx(0.0)

    def test_unit_error(self, service):
        assert service._compute_score([2.0], [1.0], "mae") == pytest.approx(1.0)

    def test_mixed_signs(self, service):
        # |+2| + |-2| = 4 → mean = 2.0
        assert service._compute_score([2.0, -2.0], [0.0, 0.0], "mae") == pytest.approx(2.0)

    def test_multiple_errors(self, service):
        # |1|+|2|+|3| = 6 → mean = 2.0
        assert service._compute_score([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], "mae") == pytest.approx(
            2.0
        )


# ---------------------------------------------------------------------------
# _compute_score - MAPE
# ---------------------------------------------------------------------------


class TestComputeScoreMape:
    def test_ten_percent_error(self, service):
        # |0.1/1.0| * 100 = 10.0
        assert service._compute_score([1.1], [1.0], "mape") == pytest.approx(10.0)

    def test_fifty_percent_error(self, service):
        assert service._compute_score([1.5], [1.0], "mape") == pytest.approx(50.0)

    def test_perfect_prediction(self, service):
        assert service._compute_score([2.0], [2.0], "mape") == pytest.approx(0.0)

    def test_multiple_errors_averaged(self, service):
        # |(10-10)/10| = 0%, |(12-10)/10| = 20% → mean = 10%
        assert service._compute_score([10.0, 12.0], [10.0, 10.0], "mape") == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# _compute_score - R²
# ---------------------------------------------------------------------------


class TestComputeScoreR2:
    def test_perfect_prediction(self, service):
        assert service._compute_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], "r2") == pytest.approx(
            1.0
        )

    def test_all_truth_same_returns_zero(self, service):
        # ss_tot = 0 → guard returns 0.0
        assert service._compute_score([1.0, 2.0], [5.0, 5.0], "r2") == pytest.approx(0.0)

    def test_mean_predictor_gives_zero(self, service):
        # Predicting the mean of truth always gives R²=0
        truth = [1.0, 3.0]
        mean_t = sum(truth) / len(truth)
        assert service._compute_score([mean_t, mean_t], truth, "r2") == pytest.approx(0.0)

    def test_negative_r2(self, service):
        # Very bad predictions can yield negative R²
        result = service._compute_score([100.0, 100.0], [1.0, 2.0], "r2")
        assert result < 0.0


# ---------------------------------------------------------------------------
# _compute_score - unknown metric
# ---------------------------------------------------------------------------


class TestComputeScoreUnknown:
    def test_unknown_metric_raises(self, service):
        with pytest.raises(ValueError, match="Unknown metric"):
            service._compute_score([1.0], [1.0], "f1")


# ---------------------------------------------------------------------------
# _sort_by_score - lower-is-better (rmse/mae/mape)
# ---------------------------------------------------------------------------


class TestSortByScoreLowerIsBetter:
    def test_sorted_ascending_unscored_last(self, service):
        perfs = [
            _make_perf("c", 0.5, "rmse"),
            _make_perf("a", 0.1, "rmse"),
            _make_perf("d", None, "rmse"),
            _make_perf("b", 0.3, "rmse"),
        ]
        result = service._sort_by_score(perfs, "rmse")
        assert [p.score for p in result] == [0.1, 0.3, 0.5, None]

    def test_all_unscored_preserved(self, service):
        perfs = [_make_perf("a", None, "rmse"), _make_perf("b", None, "rmse")]
        result = service._sort_by_score(perfs, "rmse")
        assert all(p.score is None for p in result)

    def test_empty_list(self, service):
        assert service._sort_by_score([], "rmse") == []


# ---------------------------------------------------------------------------
# _sort_by_score - higher-is-better (r2)
# ---------------------------------------------------------------------------


class TestSortByScoreR2:
    def test_sorted_descending_unscored_last(self, service):
        perfs = [
            _make_perf("c", 0.5, "r2"),
            _make_perf("a", 0.9, "r2"),
            _make_perf("d", None, "r2"),
            _make_perf("b", 0.7, "r2"),
        ]
        result = service._sort_by_score(perfs, "r2")
        scores = [p.score for p in result]
        assert scores == [0.9, 0.7, 0.5, None]

    def test_negative_r2_values_sorted(self, service):
        perfs = [
            _make_perf("a", -0.5, "r2"),
            _make_perf("b", 0.8, "r2"),
            _make_perf("c", -1.0, "r2"),
        ]
        result = service._sort_by_score(perfs, "r2")
        scores = [p.score for p in result]
        assert scores == [0.8, -0.5, -1.0]


# ---------------------------------------------------------------------------
# _get_registered_model_tags
# ---------------------------------------------------------------------------


class TestGetRegisteredModelTags:
    def test_returns_tag_dict(self, service):
        mock_rm = MagicMock()
        mock_rm.tags = {"best_for:PERF_DATA:latency_mean": "true", "score_for:latency_mean": "0.05"}
        service.client.get_registered_model.return_value = mock_rm

        result = service._get_registered_model_tags("some-model-id")

        assert result == {"best_for:PERF_DATA:latency_mean": "true", "score_for:latency_mean": "0.05"}

    def test_returns_empty_dict_when_tags_none(self, service):
        mock_rm = MagicMock()
        mock_rm.tags = None
        service.client.get_registered_model.return_value = mock_rm

        result = service._get_registered_model_tags("some-model-id")

        assert result == {}

    def test_returns_empty_dict_on_mlflow_exception(self, service):
        service.client.get_registered_model.side_effect = MlflowException("not found")

        result = service._get_registered_model_tags("bad-id")

        assert result == {}


# ---------------------------------------------------------------------------
# _write_history_row
# ---------------------------------------------------------------------------


class TestWriteHistoryRow:
    def test_adds_and_commits(self, service, mock_db):
        from src.db.score_history import ScoreHistoryDB

        service._write_history_row("model-1", "latency_mean", 0.05, "rmse", "evaluate")

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

        added = mock_db.add.call_args[0][0]
        assert isinstance(added, ScoreHistoryDB)
        assert added.model_id == "model-1"
        assert added.field_name == "latency_mean"
        assert added.score == 0.05
        assert added.metric == "rmse"
        assert added.trigger == "evaluate"

    def test_measured_at_is_utc(self, service, mock_db):
        service._write_history_row("m", "f", 1.0, "mae", "monitor")
        added = mock_db.add.call_args[0][0]
        assert added.measured_at.tzinfo is not None

    @pytest.mark.parametrize("trigger", ["evaluate", "monitor", "auto_monitor"])
    def test_accepts_all_trigger_types(self, service, mock_db, trigger):
        service._write_history_row("m", "f", 0.1, "rmse", trigger)
        added = mock_db.add.call_args[0][0]
        assert added.trigger == trigger


# ---------------------------------------------------------------------------
# _update_tags
# ---------------------------------------------------------------------------


class TestUpdateTags:
    def test_sets_score_and_eval_at_on_all_scored(self, service):
        scored = [("model-a", 0.1), ("model-b", 0.3)]
        service._update_tags("PERF_DATA", "latency_mean", scored, "model-a", "2024-01-01T00:00:00", "rmse")

        set_calls = service.client.set_registered_model_tag.call_args_list
        set_tag_keys = [(c.args[0], c.args[1]) for c in set_calls]

        assert ("model-a", "score_for:latency_mean") in set_tag_keys
        assert ("model-a", "eval_at:latency_mean") in set_tag_keys
        assert ("model-b", "score_for:latency_mean") in set_tag_keys
        assert ("model-b", "eval_at:latency_mean") in set_tag_keys

    def test_sets_best_tags_only_on_winner(self, service):
        scored = [("model-a", 0.1), ("model-b", 0.3)]
        service._update_tags("PERF_DATA", "latency_mean", scored, "model-a", "2024-01-01T00:00:00", "rmse")

        set_calls = service.client.set_registered_model_tag.call_args_list
        set_tag_keys = [(c.args[0], c.args[1]) for c in set_calls]

        # Winner gets best_for, eval_metric, baseline_score
        assert ("model-a", "best_for:PERF_DATA:latency_mean") in set_tag_keys
        assert ("model-a", "eval_metric:latency_mean") in set_tag_keys
        assert ("model-a", "baseline_score:latency_mean") in set_tag_keys

        # Loser does NOT get best_for set
        assert ("model-b", "best_for:PERF_DATA:latency_mean") not in set_tag_keys
        assert ("model-b", "eval_metric:latency_mean") not in set_tag_keys

    def test_deletes_best_for_on_non_winner(self, service):
        scored = [("model-a", 0.1), ("model-b", 0.3)]
        service._update_tags("PERF_DATA", "latency_mean", scored, "model-a", "2024-01-01T00:00:00", "rmse")

        service.client.delete_registered_model_tag.assert_called_once_with(
            "model-b", "best_for:PERF_DATA:latency_mean"
        )

    def test_delete_exception_silenced(self, service):
        service.client.delete_registered_model_tag.side_effect = MlflowException("tag not found")
        scored = [("model-a", 0.1), ("model-b", 0.3)]

        # Should not raise
        service._update_tags("PERF_DATA", "latency_mean", scored, "model-a", "2024-01-01T00:00:00", "rmse")

    def test_no_delete_when_only_one_model(self, service):
        scored = [("model-a", 0.1)]
        service._update_tags("PERF_DATA", "latency_mean", scored, "model-a", "2024-01-01T00:00:00", "rmse")

        service.client.delete_registered_model_tag.assert_not_called()

    def test_baseline_score_value_is_winner_score(self, service):
        scored = [("model-a", 0.07)]
        service._update_tags("PERF_DATA", "latency_mean", scored, "model-a", "2024-01-01T00:00:00", "rmse")

        set_calls = {c.args[1]: c.args[2] for c in service.client.set_registered_model_tag.call_args_list}
        assert set_calls["baseline_score:latency_mean"] == "0.07"


# ---------------------------------------------------------------------------
# get_monitored_fields
# ---------------------------------------------------------------------------


class TestGetMonitoredFields:
    def test_returns_fields_from_best_for_tags(self, service, mock_ml_config_service):
        cfg_a = MagicMock()
        cfg_a.model_id = "model-a"
        cfg_b = MagicMock()
        cfg_b.model_id = "model-b"
        mock_ml_config_service.list_all.return_value = [cfg_a, cfg_b]

        rm_a = MagicMock()
        rm_a.tags = {"best_for:PERF_DATA:latency_mean": "true", "score_for:latency_mean": "0.05"}
        rm_b = MagicMock()
        rm_b.tags = {"best_for:UE_MOBILITY:jitter": "true"}
        service.client.get_registered_model.side_effect = lambda mid: (
            rm_a if mid == "model-a" else rm_b
        )

        result = service.get_monitored_fields()

        assert set(result) == {"PERF_DATA:latency_mean", "UE_MOBILITY:jitter"}

    def test_deduplicates_fields(self, service, mock_ml_config_service):
        cfg_a = MagicMock()
        cfg_a.model_id = "model-a"
        cfg_b = MagicMock()
        cfg_b.model_id = "model-b"
        mock_ml_config_service.list_all.return_value = [cfg_a, cfg_b]

        # Both models tagged as best for the same field
        rm = MagicMock()
        rm.tags = {"best_for:PERF_DATA:latency_mean": "true"}
        service.client.get_registered_model.return_value = rm

        result = service.get_monitored_fields()

        assert result.count("PERF_DATA:latency_mean") == 1

    def test_returns_empty_when_no_best_for_tags(self, service, mock_ml_config_service):
        cfg = MagicMock()
        cfg.model_id = "model-a"
        mock_ml_config_service.list_all.return_value = [cfg]

        rm = MagicMock()
        rm.tags = {"score_for:latency_mean": "0.1"}  # no best_for tag
        service.client.get_registered_model.return_value = rm

        assert service.get_monitored_fields() == []

    def test_returns_empty_when_no_models(self, service, mock_ml_config_service):
        mock_ml_config_service.list_all.return_value = []
        assert service.get_monitored_fields() == []


# ---------------------------------------------------------------------------
# get_baseline_score
# ---------------------------------------------------------------------------


class TestGetBaselineScore:
    def test_returns_none_when_no_best_model(self, service, mock_ml_config_service):
        mock_ml_config_service.list_all.return_value = []
        assert service.get_baseline_score("latency_mean") is None

    def test_returns_float_from_tag(self, service, mock_ml_config_service):
        cfg = MagicMock()
        cfg.model_id = "model-a"
        cfg.name = "model_a"
        cfg.architecture = "lstm"
        cfg.output_fields = ["latency_mean"]
        cfg.event_type = "PERF_DATA"
        mock_ml_config_service.list_all.return_value = [cfg]

        # get_best_model reads best_for tag → model detail
        service.mlflow_service.get_model.return_value = MagicMock(
            latest_version=1,
            training_loss=0.01,
            last_trained_at=None,
        )

        rm = MagicMock()
        rm.tags = {
            "best_for:PERF_DATA:latency_mean": "true",
            "score_for:latency_mean": "0.05",
            "baseline_score:latency_mean": "0.04",
            "eval_metric:latency_mean": "rmse",
            "eval_at:latency_mean": "2024-01-01T00:00:00+00:00",
        }
        service.client.get_registered_model.return_value = rm

        result = service.get_baseline_score("latency_mean")

        assert result == pytest.approx(0.04)

    def test_returns_none_when_baseline_tag_missing(self, service, mock_ml_config_service):
        cfg = MagicMock()
        cfg.model_id = "model-a"
        cfg.name = "model_a"
        cfg.architecture = "lstm"
        cfg.output_fields = ["latency_mean"]
        mock_ml_config_service.list_all.return_value = [cfg]

        service.mlflow_service.get_model.return_value = MagicMock(
            latest_version=1,
            training_loss=None,
            last_trained_at=None,
        )

        rm = MagicMock()
        rm.tags = {
            "best_for:PERF_DATA:latency_mean": "true",
            "score_for:latency_mean": "0.05",
            # baseline_score tag intentionally absent
        }
        service.client.get_registered_model.return_value = rm

        assert service.get_baseline_score("latency_mean") is None
