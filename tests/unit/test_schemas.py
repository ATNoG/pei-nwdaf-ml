"""Unit tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from src.schemas.model import (
    ArchitectureType,
    ModelConfig,
    ModelCreate,
    ModelSummary,
    ModelDetail,
)
from src.schemas.inference import (
    InferenceRequest,
    ForecastStepPrediction,
    InferenceResult,
)


class TestModelConfig:
    """Tests for ModelConfig schema."""

    def test_valid_config(self):
        """Test creating a valid model config."""
        config = ModelConfig(
            architecture=ArchitectureType.ANN,
            input_fields=["latency_mean", "rsrp_mean"],
            output_fields=["latency_mean"],
            window_duration_seconds=60,
            lookback_steps=10,
            forecast_steps=5,
            hidden_size=64,
        )

        assert config.architecture == ArchitectureType.ANN
        assert config.input_fields == ["latency_mean", "rsrp_mean"]
        assert config.output_fields == ["latency_mean"]
        assert config.window_duration_seconds == 60
        assert config.lookback_steps == 10
        assert config.forecast_steps == 5
        assert config.hidden_size == 64

    def test_default_hidden_size(self):
        """Test that hidden_size defaults to 32."""
        config = ModelConfig(
            architecture=ArchitectureType.LSTM,
            input_fields=["latency_mean"],
            output_fields=["latency_mean"],
            window_duration_seconds=300,
            lookback_steps=20,
            forecast_steps=10,
        )

        assert config.hidden_size == 32

    def test_empty_input_fields_invalid(self):
        """Test that empty input_fields is invalid."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                architecture=ArchitectureType.ANN,
                input_fields=[],
                output_fields=["latency_mean"],
                window_duration_seconds=60,
                lookback_steps=10,
                forecast_steps=5,
            )

        assert "at least 1 item" in str(exc_info.value).lower()

    def test_empty_output_fields_invalid(self):
        """Test that empty output_fields is invalid."""
        with pytest.raises(ValidationError):
            ModelConfig(
                architecture=ArchitectureType.ANN,
                input_fields=["latency_mean"],
                output_fields=[],
                window_duration_seconds=60,
                lookback_steps=10,
                forecast_steps=5,
            )

    def test_negative_window_duration_invalid(self):
        """Test that negative window_duration_seconds is invalid."""
        with pytest.raises(ValidationError):
            ModelConfig(
                architecture=ArchitectureType.ANN,
                input_fields=["latency_mean"],
                output_fields=["latency_mean"],
                window_duration_seconds=0,
                lookback_steps=10,
                forecast_steps=5,
            )

    def test_zero_lookback_steps_invalid(self):
        """Test that zero lookback_steps is invalid."""
        with pytest.raises(ValidationError):
            ModelConfig(
                architecture=ArchitectureType.ANN,
                input_fields=["latency_mean"],
                output_fields=["latency_mean"],
                window_duration_seconds=60,
                lookback_steps=0,
                forecast_steps=5,
            )

    def test_zero_forecast_steps_invalid(self):
        """Test that zero forecast_steps is invalid."""
        with pytest.raises(ValidationError):
            ModelConfig(
                architecture=ArchitectureType.ANN,
                input_fields=["latency_mean"],
                output_fields=["latency_mean"],
                window_duration_seconds=60,
                lookback_steps=10,
                forecast_steps=0,
            )

    def test_small_hidden_size_invalid(self):
        """Test that hidden_size < 4 is invalid."""
        with pytest.raises(ValidationError):
            ModelConfig(
                architecture=ArchitectureType.ANN,
                input_fields=["latency_mean"],
                output_fields=["latency_mean"],
                window_duration_seconds=60,
                lookback_steps=10,
                forecast_steps=5,
                hidden_size=3,
            )


class TestModelCreate:
    """Tests for ModelCreate schema."""

    def test_valid_model_create(self, sample_model_config):
        """Test creating a valid model creation request."""
        model_create = ModelCreate(
            name="my_lstm_model",
            config=sample_model_config,
        )

        assert model_create.name == "my_lstm_model"
        assert model_create.config == sample_model_config

    def test_invalid_name_with_spaces(self, sample_model_config):
        """Test that names with spaces are invalid."""
        with pytest.raises(ValidationError):
            ModelCreate(
                name="my model",
                config=sample_model_config,
            )

    def test_invalid_name_with_special_chars(self, sample_model_config):
        """Test that names with special characters are invalid."""
        with pytest.raises(ValidationError):
            ModelCreate(
                name="my@model!",
                config=sample_model_config,
            )

    def test_valid_name_with_underscore(self, sample_model_config):
        """Test that names with underscores are valid."""
        model_create = ModelCreate(
            name="my_lstm_model",
            config=sample_model_config,
        )

        assert model_create.name == "my_lstm_model"

    def test_valid_name_with_hyphen(self, sample_model_config):
        """Test that names with hyphens are valid."""
        model_create = ModelCreate(
            name="my-lstm-model",
            config=sample_model_config,
        )

        assert model_create.name == "my-lstm-model"

    def test_empty_name_invalid(self, sample_model_config):
        """Test that empty names are invalid."""
        with pytest.raises(ValidationError):
            ModelCreate(
                name="",
                config=sample_model_config,
            )

    def test_long_name_invalid(self, sample_model_config):
        """Test that very long names are invalid."""
        with pytest.raises(ValidationError):
            ModelCreate(
                name="a" * 129,  # max is 128
                config=sample_model_config,
            )


class TestModelSummary:
    """Tests for ModelSummary schema."""

    def test_model_summary_creation(self):
        """Test creating a model summary."""
        from datetime import datetime

        summary = ModelSummary(
            id="uuid-123",
            name="my_model",
            architecture=ArchitectureType.LSTM,
            created_at=datetime(2024, 1, 1),
            latest_version=1,
        )

        assert summary.id == "uuid-123"
        assert summary.name == "my_model"
        assert summary.architecture == ArchitectureType.LSTM
        assert summary.latest_version == 1

    def test_model_summary_optional_fields(self):
        """Test that created_at and latest_version are optional."""
        summary = ModelSummary(
            id="uuid-123",
            name="my_model",
            architecture=ArchitectureType.ANN
        )

        assert summary.created_at is None
        assert summary.latest_version is None


class TestModelDetail:
    """Tests for ModelDetail schema."""

    def test_model_detail_creation(self, sample_model_config):
        """Test creating a detailed model response."""
        from datetime import datetime

        detail = ModelDetail(
            id="uuid-123",
            name="my_model",
            config=sample_model_config,
            created_at=datetime(2024, 1, 1),
            latest_version=2,
            last_trained_at=datetime(2024, 1, 15),
            mlflow_run_id="run-abc",
            training_loss=0.025,
        )

        assert detail.id == "uuid-123"
        assert detail.name == "my_model"
        assert detail.config == sample_model_config
        assert detail.latest_version == 2
        assert detail.training_loss == 0.025

    def test_model_detail_optional_fields(self, sample_model_config):
        """Test that training-related fields are optional."""
        detail = ModelDetail(
            id="uuid-123",
            name="my_model",
            config=sample_model_config        )

        assert detail.created_at is None
        assert detail.latest_version is None
        assert detail.last_trained_at is None
        assert detail.mlflow_run_id is None
        assert detail.training_loss is None


class TestInferenceRequest:
    """Tests for InferenceRequest schema."""

    def test_valid_request(self):
        """Test creating a valid inference request with all fields."""
        req = InferenceRequest(output_field="latency_mean", model_id="uuid-123", cell_id=5)

        assert req.output_field == "latency_mean"
        assert req.model_id == "uuid-123"
        assert req.cell_id == 5

    def test_cell_id_zero_valid(self):
        """Test that cell_id=0 is valid (boundary)."""
        req = InferenceRequest(output_field="latency_mean", model_id="uuid-123", cell_id=0)

        assert req.cell_id == 0

    def test_model_id_optional(self):
        """Test that model_id is optional (defaults to None — uses best model)."""
        req = InferenceRequest(output_field="latency_mean", cell_id=5)

        assert req.model_id is None
        assert req.cell_id == 5

    def test_negative_cell_id_invalid(self):
        """Test that negative cell_id is rejected."""
        with pytest.raises(ValidationError):
            InferenceRequest(output_field="latency_mean", model_id="uuid-123", cell_id=-1)

    def test_missing_output_field_invalid(self):
        """Test that output_field is required."""
        with pytest.raises(ValidationError):
            InferenceRequest(cell_id=5)

    def test_missing_cell_id_invalid(self):
        """Test that cell_id is required."""
        with pytest.raises(ValidationError):
            InferenceRequest(output_field="latency_mean", model_id="uuid-123")


class TestForecastStepPrediction:
    """Tests for ForecastStepPrediction schema."""

    def test_valid_prediction(self):
        """Test creating a valid prediction."""
        pred = ForecastStepPrediction(
            step=1,
            window_start_time=1000,
            window_end_time=1060,
            values={"latency_mean": 12.345}
        )

        assert pred.step == 1
        assert pred.window_start_time == 1000
        assert pred.window_end_time == 1060
        assert pred.values == {"latency_mean": 12.345}

    def test_step_zero_invalid(self):
        """Test that step=0 is rejected (ge=1)."""
        with pytest.raises(ValidationError):
            ForecastStepPrediction(
                step=0,
                window_start_time=1000,
                window_end_time=1060,
                values={"a": 1.0}
            )

    def test_negative_step_invalid(self):
        """Test that negative step is rejected."""
        with pytest.raises(ValidationError):
            ForecastStepPrediction(
                step=-1,
                window_start_time=1000,
                window_end_time=1060,
                values={"a": 1.0}
            )

    def test_empty_values_valid(self):
        """Test that empty dict values is valid."""
        pred = ForecastStepPrediction(
            step=1,
            window_start_time=1000,
            window_end_time=1060,
            values={}
        )

        assert pred.values == {}

    def test_multiple_values(self):
        """Test multiple output fields in values."""
        pred = ForecastStepPrediction(
            step=1,
            window_start_time=1000,
            window_end_time=1060,
            values={"latency_mean": 1.0, "throughput_mean": 2.0, "sinr_mean": 3.0},
        )

        assert len(pred.values) == 3


class TestInferenceResult:
    """Tests for InferenceResult schema."""

    def test_valid_full_result(self):
        """Test creating a full valid inference result."""
        result = InferenceResult(
            model_id="uuid-123",
            model_name="test_model",
            model_version=2,
            architecture=ArchitectureType.LSTM,
            cell_id=5,
            lookback_steps=30,
            forecast_steps=5,
            window_duration_seconds=60,
            input_data_start=1000,
            input_data_end=2800,
            input_fields=["rsrp_mean", "sinr_mean"],
            output_fields=["latency_mean"],
            predictions=[
                ForecastStepPrediction(
                    step=1,
                    window_start_time=2800,
                    window_end_time=2860,
                    values={"latency_mean": 10.0}
                ),
                ForecastStepPrediction(
                    step=2,
                    window_start_time=2860,
                    window_end_time=2920,
                    values={"latency_mean": 11.0}
                ),
            ],
        )

        assert result.model_id == "uuid-123"
        assert result.model_version == 2
        assert result.architecture == ArchitectureType.LSTM
        assert result.input_data_start == 1000
        assert result.input_data_end == 2800
        assert len(result.predictions) == 2

    def test_architecture_ann(self):
        """Test that ANN architecture is accepted."""
        result = InferenceResult(
            model_id="uuid-123",
            model_name="test_ann",
            model_version=1,
            architecture=ArchitectureType.ANN,
            cell_id=0,
            lookback_steps=10,
            forecast_steps=3,
            window_duration_seconds=60,
            input_data_start=1000,
            input_data_end=1600,
            input_fields=["rsrp_mean"],
            output_fields=["latency_mean"],
            predictions=[],
        )

        assert result.architecture == ArchitectureType.ANN
