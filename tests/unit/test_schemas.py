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
