"""Unit tests for InferenceService."""

import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch

from src.services.inference_service import InferenceService
from src.schemas.model import ModelConfig, ModelDetail
from src.schemas.inference import ForecastStepPrediction


@pytest.fixture
def sample_model_config():
    """Model config with small dimensions for fast tests."""
    return ModelConfig(
        architecture="lstm",
        input_fields=["rsrp_mean", "sinr_mean", "latency_mean"],
        output_fields=["latency_mean"],
        window_duration_seconds=60,
        lookback_steps=5,
        forecast_steps=3,
        hidden_size=32,
    )


@pytest.fixture
def sample_model_detail(sample_model_config):
    return ModelDetail(
        id="test-uuid",
        name="test_model",
        config=sample_model_config,
        event_type="PERF_DATA",
        created_at=None,
        latest_version=2,
        last_trained_at=None,
        mlflow_run_id=None,
        training_loss=None,
    )


@pytest.fixture
def sample_cell_data():
    """10 windows of cell data with known values."""
    return [
        {
            "window_start_time": 1000 + i * 60,
            "rsrp_mean": -85.0 + i,
            "sinr_mean": 10.0 + i * 0.5,
            "latency_mean": 20.0 + i * 2,
        }
        for i in range(10)
    ]


@pytest.fixture
def mock_mlflow_service():
    mock = MagicMock()
    # Set up a config so the best-model lookup in predict() resolves "test-uuid"
    mock_config = MagicMock()
    mock_config.model_id = "test-uuid"
    mock_config.output_fields = ["latency_mean"]
    mock.ml_config_service.list_all.return_value = [mock_config]
    # Tag the model as best for "latency_mean"
    mock_rm = MagicMock()
    mock_rm.tags = {"best_for:PERF_DATA:latency_mean": "true"}
    mock.client.get_registered_model.return_value = mock_rm
    return mock


@pytest.fixture
def mock_data_storage_client():
    mock = MagicMock()
    mock.fetch_data = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def inference_service(mock_mlflow_service, mock_data_storage_client):
    return InferenceService(
        mlflow_service=mock_mlflow_service,
        data_storage_client=mock_data_storage_client,
    )


class TestInferenceServicePredict:
    """Tests for InferenceService.predict method."""

    async def test_predict_success(
        self,
        inference_service,
        mock_mlflow_service,
        mock_data_storage_client,
        sample_model_detail,
        sample_cell_data,
    ):
        """Test full happy path returns correct structure."""
        config = sample_model_detail.config
        num_outputs = len(config.output_fields)
        forecast_steps = config.forecast_steps

        mock_mlflow_service.get_model.return_value = sample_model_detail
        mock_data_storage_client.fetch_data = AsyncMock(
            return_value=sample_cell_data
        )

        with patch(
            "src.services.inference_service.safe_predict",
            return_value=np.zeros((1, forecast_steps * num_outputs), dtype=np.float32),
        ):
            result = await inference_service.predict(
                output_field="latency_mean",
                tags={"snssai_sst": "1", "dnn": "internet", "event": "PERF_DATA"},
                model_id="test-uuid",
            )

        assert result["model_id"] == "test-uuid"
        assert result["model_name"] == "test_model"
        assert result["model_version"] == 2
        assert result["architecture"] == "lstm"
        assert result["tags"]["snssai_sst"] == "1"
        assert result["lookback_steps"] == config.lookback_steps
        assert result["forecast_steps"] == forecast_steps
        assert len(result["predictions"]) == forecast_steps

    async def test_predict_model_not_found(
        self, inference_service, mock_mlflow_service
    ):
        """Test that passing an unknown model_id raises ValueError."""
        with pytest.raises(ValueError, match="does not predict field"):
            await inference_service.predict(
                output_field="latency_mean", tags={"snssai_sst": "1", "dnn": "internet", "event": "PERF_DATA"}, model_id="bad-uuid"
            )

    async def test_predict_model_not_trained(
        self, inference_service, mock_mlflow_service, sample_model_detail
    ):
        """Test that untrained model (latest_version=None) raises ValueError."""
        sample_model_detail.latest_version = None
        mock_mlflow_service.get_model.return_value = sample_model_detail

        with pytest.raises(ValueError, match="no trained versions"):
            await inference_service.predict(
                output_field="latency_mean", tags={"snssai_sst": "1", "dnn": "internet", "event": "PERF_DATA"}, model_id="test-uuid"
            )

    async def test_predict_no_data(
        self,
        inference_service,
        mock_mlflow_service,
        mock_data_storage_client,
        sample_model_detail,
    ):
        """Test that empty cell data raises ValueError."""
        mock_mlflow_service.get_model.return_value = sample_model_detail
        mock_data_storage_client.fetch_data = AsyncMock(return_value=[])

        with pytest.raises(ValueError, match="No data for tags"):
            await inference_service.predict(
                output_field="latency_mean", tags={"snssai_sst": "1", "dnn": "internet", "event": "PERF_DATA"}, model_id="test-uuid"
            )

    async def test_predict_insufficient_data(
        self,
        inference_service,
        mock_mlflow_service,
        mock_data_storage_client,
        sample_model_detail,
    ):
        """Test that fewer windows than lookback raises ValueError."""
        mock_mlflow_service.get_model.return_value = sample_model_detail
        # Only 3 windows, but lookback_steps=5
        sparse_data = [
            {"window_start_time": 1000 + i * 60, "rsrp_mean": 1.0}
            for i in range(3)
        ]
        mock_data_storage_client.fetch_data = AsyncMock(
            return_value=sparse_data
        )

        with pytest.raises(ValueError, match="Insufficient data"):
            await inference_service.predict(
                output_field="latency_mean", tags={"snssai_sst": "1", "dnn": "internet", "event": "PERF_DATA"}, model_id="test-uuid"
            )

    async def test_predict_model_failure(
        self,
        inference_service,
        mock_mlflow_service,
        mock_data_storage_client,
        sample_model_detail,
        sample_cell_data,
    ):
        """Test that model.predict() failure raises RuntimeError."""
        mock_mlflow_service.get_model.return_value = sample_model_detail
        mock_data_storage_client.fetch_data = AsyncMock(
            return_value=sample_cell_data
        )

        with patch(
            "src.services.inference_service.safe_predict",
            side_effect=RuntimeError("Prediction failed: CUDA error"),
        ):
            with pytest.raises(RuntimeError, match="Prediction failed"):
                await inference_service.predict(
                    output_field="latency_mean", tags={"snssai_sst": "1", "dnn": "internet", "event": "PERF_DATA"}, model_id="test-uuid"
                )

    async def test_predict_correct_timestamp_calculation(
        self,
        inference_service,
        mock_mlflow_service,
        mock_data_storage_client,
        sample_model_detail,
        sample_cell_data,
    ):
        """Test that timestamps include the buffer window."""
        config = sample_model_detail.config
        mock_mlflow_service.get_model.return_value = sample_model_detail
        mock_data_storage_client.fetch_data = AsyncMock(
            return_value=sample_cell_data
        )

        with patch(
            "src.services.inference_service.safe_predict",
            return_value=np.zeros((1, config.forecast_steps * len(config.output_fields)), dtype=np.float32),
        ), patch(
            "src.services.inference_service.calculate_timestamps",
            return_value=(1000, 2000),
        ) as mock_ts:
            await inference_service.predict(
                output_field="latency_mean", tags={"snssai_sst": "1", "dnn": "internet", "event": "PERF_DATA"}, model_id="test-uuid"
            )

            expected_seconds = (
                config.lookback_steps * config.window_duration_seconds
                + config.window_duration_seconds
            )
            mock_ts.assert_called_once_with(expected_seconds)


class TestLoadTrainedModel:
    """Tests for InferenceService._load_trained_model."""

    def test_delegates_to_load_trained_model(self, inference_service, sample_model_config):
        """Test that _load_trained_model delegates to load_trained_model."""
        mock_model = MagicMock()
        with patch(
            "src.services.inference_service.load_trained_model",
            return_value=mock_model,
        ) as mock_loader:
            result = inference_service._load_trained_model(
                model_id="test-uuid", version=2, config=sample_model_config
            )
            mock_loader.assert_called_once_with(
                sample_model_config.architecture,
                "models:/test-uuid/2",
                sample_model_config,
            )
            assert result == mock_model

    def test_mlflow_load_failure_raises_valueerror(
        self, inference_service, sample_model_config
    ):
        """Test that load_trained_model failure propagates."""
        with patch(
            "src.services.inference_service.load_trained_model",
            side_effect=ValueError("Failed to load model artifact from models:/test-uuid/2"),
        ):
            with pytest.raises(ValueError, match="Failed to load model artifact"):
                inference_service._load_trained_model(
                    model_id="test-uuid", version=2, config=sample_model_config
                )


class TestStructurePredictions:
    """Tests for InferenceService._structure_predictions."""

    @pytest.fixture
    def service(self):
        return InferenceService(
            mlflow_service=MagicMock(),
            data_storage_client=MagicMock(),
        )

    def test_basic_reshaping(self, service):
        """Test reshaping flat output into per-step predictions."""
        raw = np.array([[1.0, 2.0, 3.0]])

        result = service._structure_predictions(
            raw_predictions=raw,
            output_fields=["latency_mean"],
            forecast_steps=3,
            last_window_end=1000,
            window_duration_seconds=60,
            window_overlap=0,
        )

        assert len(result) == 3
        assert result[0].values["latency_mean"] == 1.0
        assert result[1].values["latency_mean"] == 2.0
        assert result[2].values["latency_mean"] == 3.0

    def test_multiple_output_fields(self, service):
        """Test reshaping with multiple output fields per step."""
        raw = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

        result = service._structure_predictions(
            raw_predictions=raw,
            output_fields=["latency_mean", "throughput_mean"],
            forecast_steps=3,
            last_window_end=1000,
            window_duration_seconds=60,
            window_overlap=0,
        )

        assert len(result) == 3
        assert result[0].values == {"latency_mean": 1.0, "throughput_mean": 2.0}
        assert result[1].values == {"latency_mean": 3.0, "throughput_mean": 4.0}
        assert result[2].values == {"latency_mean": 5.0, "throughput_mean": 6.0}

    def test_values_rounded(self, service):
        """Test that values are rounded to 6 decimal places."""
        raw = np.array([[1.1234567890]])

        result = service._structure_predictions(
            raw_predictions=raw,
            output_fields=["latency_mean"],
            forecast_steps=1,
            last_window_end=1000,
            window_duration_seconds=60,
            window_overlap=0,
        )

        assert result[0].values["latency_mean"] == 1.123457

    def test_step_numbering_starts_at_one(self, service):
        """Test that steps are 1-indexed."""
        raw = np.array([[1.0, 2.0, 3.0]])

        result = service._structure_predictions(
            raw_predictions=raw,
            output_fields=["latency_mean"],
            forecast_steps=3,
            last_window_end=1000,
            window_duration_seconds=60,
            window_overlap=0,
        )

        assert result[0].step == 1
        assert result[1].step == 2
        assert result[2].step == 3

    def test_returns_forecast_step_prediction_type(self, service):
        """Test that each element is a ForecastStepPrediction."""
        raw = np.array([[1.0]])

        result = service._structure_predictions(
            raw_predictions=raw,
            output_fields=["latency_mean"],
            forecast_steps=1,
            last_window_end=1000,
            window_duration_seconds=60,
            window_overlap=0,
        )

        assert isinstance(result[0], ForecastStepPrediction)

    def test_shape_mismatch_raises(self, service):
        """Test that mismatched output shape raises RuntimeError."""
        # 4 values but expect 3 (3 steps * 1 field)
        raw = np.array([[1.0, 2.0, 3.0, 4.0]])

        with pytest.raises(RuntimeError, match="Model output shape mismatch"):
            service._structure_predictions(
                raw_predictions=raw,
                output_fields=["latency_mean"],
                forecast_steps=3,
                last_window_end=1000,
                window_duration_seconds=60,
                window_overlap=0,
            )

    def test_window_timestamps_tumbling(self, service):
        """Test window timestamps for tumbling windows (overlap=0)."""
        raw = np.array([[1.0, 2.0, 3.0]])

        result = service._structure_predictions(
            raw_predictions=raw,
            output_fields=["latency_mean"],
            forecast_steps=3,
            last_window_end=1000,
            window_duration_seconds=60,
            window_overlap=0,
        )

        # For tumbling windows, each window starts where the previous ended
        assert result[0].window_start_time == 1000
        assert result[0].window_end_time == 1060
        assert result[1].window_start_time == 1060
        assert result[1].window_end_time == 1120
        assert result[2].window_start_time == 1120
        assert result[2].window_end_time == 1180

    def test_window_timestamps_sliding(self, service):
        """Test window timestamps for sliding windows (overlap>0)."""
        raw = np.array([[1.0, 2.0]])

        result = service._structure_predictions(
            raw_predictions=raw,
            output_fields=["latency_mean"],
            forecast_steps=2,
            last_window_end=1000,
            window_duration_seconds=60,
            window_overlap=30,
        )

        # For sliding windows with overlap=30, step_size=30
        assert result[0].window_start_time == 1000
        assert result[0].window_end_time == 1060
        assert result[1].window_start_time == 1030
        assert result[1].window_end_time == 1090
