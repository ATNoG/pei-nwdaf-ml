"""Unit tests for anomaly detection Pydantic schemas."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.schemas.anomaly import (
    AnomalyModelConfig,
    AnomalyModelCreate,
    AnomalyModelSummary,
    AnomalyModelDetail,
    AnomalyTrainingRequest,
    AnomalyTrainingResponse,
    AnomalyTrainingJobDetail,
    AnomalyTrainingJobSummary,
    AnomalyDetectionRequest,
    WindowScore,
    AnomalyDetectionResult,
)
from src.schemas.tags import Tags


class TestAnomalyModelConfig:
    """Tests for AnomalyModelConfig schema."""

    def test_valid_config(self):
        config = AnomalyModelConfig(
            input_fields=["latency_mean", "rsrp_mean"],
            window_duration_seconds=60,
            hidden_size=32,
            threshold_percentile=95.0,
        )
        assert config.input_fields == ["latency_mean", "rsrp_mean"]
        assert config.window_duration_seconds == 60
        assert config.hidden_size == 32
        assert config.threshold_percentile == 95.0

    def test_default_values(self):
        config = AnomalyModelConfig(
            input_fields=["latency_mean"],
            window_duration_seconds=60,
        )
        assert config.hidden_size == 32
        assert config.threshold_percentile == 95.0

    def test_empty_input_fields_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            AnomalyModelConfig(
                input_fields=[],
                window_duration_seconds=60,
            )
        assert "at least 1 item" in str(exc_info.value).lower()

    def test_zero_window_duration_invalid(self):
        with pytest.raises(ValidationError):
            AnomalyModelConfig(
                input_fields=["latency_mean"],
                window_duration_seconds=0,
            )

    def test_small_hidden_size_invalid(self):
        with pytest.raises(ValidationError):
            AnomalyModelConfig(
                input_fields=["latency_mean"],
                window_duration_seconds=60,
                hidden_size=3,
            )

    def test_threshold_percentile_bounds(self):
        # Valid boundaries
        AnomalyModelConfig(
            input_fields=["f"], window_duration_seconds=60, threshold_percentile=0.0
        )
        AnomalyModelConfig(
            input_fields=["f"], window_duration_seconds=60, threshold_percentile=100.0
        )

        # Invalid
        with pytest.raises(ValidationError):
            AnomalyModelConfig(
                input_fields=["f"], window_duration_seconds=60, threshold_percentile=-1.0
            )
        with pytest.raises(ValidationError):
            AnomalyModelConfig(
                input_fields=["f"], window_duration_seconds=60, threshold_percentile=100.1
            )


class TestAnomalyModelCreate:
    """Tests for AnomalyModelCreate schema."""

    def test_valid_create(self):
        mc = AnomalyModelCreate(
            name="test-anomaly",
            config=AnomalyModelConfig(
                input_fields=["latency_mean"],
                window_duration_seconds=60,
            ),
        )
        assert mc.name == "test-anomaly"

    def test_invalid_name_with_spaces(self):
        with pytest.raises(ValidationError):
            AnomalyModelCreate(
                name="bad name",
                config=AnomalyModelConfig(
                    input_fields=["f"], window_duration_seconds=60
                ),
            )

    def test_empty_name_invalid(self):
        with pytest.raises(ValidationError):
            AnomalyModelCreate(
                name="",
                config=AnomalyModelConfig(
                    input_fields=["f"], window_duration_seconds=60
                ),
            )

    def test_long_name_invalid(self):
        with pytest.raises(ValidationError):
            AnomalyModelCreate(
                name="a" * 129,
                config=AnomalyModelConfig(
                    input_fields=["f"], window_duration_seconds=60
                ),
            )


class TestAnomalyModelSummary:
    """Tests for AnomalyModelSummary schema."""

    def test_creation(self):
        s = AnomalyModelSummary(
            id="uuid-1", name="m1", event_type="PERF_DATA", created_at=datetime(2024, 1, 1), latest_version=2
        )
        assert s.id == "uuid-1"
        assert s.latest_version == 2

    def test_optional_fields(self):
        s = AnomalyModelSummary(id="uuid-1", name="m1", event_type="PERF_DATA")
        assert s.created_at is None
        assert s.latest_version is None


class TestAnomalyModelDetail:
    """Tests for AnomalyModelDetail schema."""

    def test_full_detail(self):
        d = AnomalyModelDetail(
            id="uuid-1",
            name="m1",
            event_type="PERF_DATA",
            config=AnomalyModelConfig(
                input_fields=["f1"], window_duration_seconds=60
            ),
            threshold_value=0.05,
            created_at=datetime(2024, 1, 1),
            latest_version=1,
            last_trained_at=datetime(2024, 1, 2),
            mlflow_run_id="run-1",
            training_loss=0.01,
        )
        assert d.threshold_value == 0.05
        assert d.training_loss == 0.01

    def test_optional_fields(self):
        d = AnomalyModelDetail(
            id="uuid-1",
            name="m1",
            event_type="PERF_DATA",
            config=AnomalyModelConfig(
                input_fields=["f1"], window_duration_seconds=60
            ),
        )
        assert d.threshold_value is None
        assert d.latest_version is None
        assert d.last_trained_at is None
        assert d.mlflow_run_id is None
        assert d.training_loss is None


class TestAnomalyTrainingSchemas:
    """Tests for anomaly training schemas."""

    def test_training_request(self):
        r = AnomalyTrainingRequest(model_id="uuid-1", lookback_seconds=86400)
        assert r.model_id == "uuid-1"
        assert r.lookback_seconds == 86400

    def test_training_request_zero_lookback_invalid(self):
        with pytest.raises(ValidationError):
            AnomalyTrainingRequest(model_id="uuid-1", lookback_seconds=0)

    def test_training_response(self):
        r = AnomalyTrainingResponse(
            job_id="j1",
            model_id="m1",
            status="queued",
            message="ok",
            created_at=datetime(2024, 1, 1),
        )
        assert r.job_id == "j1"

    def test_training_job_detail(self):
        d = AnomalyTrainingJobDetail(
            job_id="j1",
            model_id="m1",
            status="completed",
            mlflow_run_id="run-1",
            created_at=datetime(2024, 1, 1),
            started_at=datetime(2024, 1, 1, 0, 1),
            completed_at=datetime(2024, 1, 1, 0, 10),
            error_message=None,
        )
        assert d.status == "completed"
        assert d.error_message is None

    def test_training_job_summary(self):
        s = AnomalyTrainingJobSummary(
            job_id="j1", model_id="m1", status="queued", created_at=datetime(2024, 1, 1)
        )
        assert s.started_at is None


_TAGS = Tags(snssai_sst="1", snssai_sd="000001", dnn="internet", event="PERF_DATA")


class TestAnomalyDetectionSchemas:
    """Tests for anomaly detection schemas."""

    def test_detection_request(self):
        r = AnomalyDetectionRequest(tags=_TAGS, model_id="uuid-1")
        assert r.tags.snssai_sst == "1"
        assert r.model_id == "uuid-1"
        assert r.lookback_seconds == 1800

    def test_detection_request_defaults(self):
        r = AnomalyDetectionRequest(tags=_TAGS)
        assert r.model_id is None
        assert r.lookback_seconds == 1800

    def test_detection_request_custom_lookback(self):
        r = AnomalyDetectionRequest(tags=_TAGS, lookback_seconds=7200)
        assert r.lookback_seconds == 7200

    def test_detection_request_zero_lookback_invalid(self):
        with pytest.raises(ValidationError):
            AnomalyDetectionRequest(tags=_TAGS, lookback_seconds=0)

    def test_window_score(self):
        ws = WindowScore(
            window_start_time=1000, reconstruction_error=0.123, is_anomaly=False
        )
        assert ws.window_start_time == 1000
        assert not ws.is_anomaly

    def test_detection_result(self):
        r = AnomalyDetectionResult(
            model_id="uuid-1",
            model_name="test",
            tags=_TAGS,
            threshold_value=0.5,
            window_duration_seconds=60,
            input_fields=["f1"],
        )
        assert r.model_id == "uuid-1"
        assert r.tags.snssai_sst == "1"
