"""Unit tests for TrainingService."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.schemas.model import ModelConfig
from src.services.training_service import TrainingService


@pytest.fixture
def mock_data_storage_client():
    mock = MagicMock()
    mock.fetch_data = AsyncMock(return_value=[])
    return mock


class TestTrainingService:
    """Test suite for TrainingService."""

    @pytest.fixture
    def mock_mlflow_service(self):
        """Mock MLflowService."""
        mock = MagicMock()
        mock.get_model = MagicMock()
        return mock

    @pytest.fixture
    def mock_config_service(self):
        """Mock MLConfigService."""
        return MagicMock()

    @pytest.fixture
    def training_service(
        self,
        mock_mlflow_service,
        mock_data_storage_client,
        mock_config_service,
        mock_db_session,
    ):
        """Create TrainingService instance with mocks."""
        return TrainingService(
            mlflow_service=mock_mlflow_service,
            data_storage_client=mock_data_storage_client,
            ml_config_service=mock_config_service,
            db=mock_db_session,
        )

    @pytest.fixture
    def sample_model_config(self):
        """Sample model configuration."""
        return ModelConfig(
            architecture="lstm",
            input_fields=["rsrp_mean", "sinr_mean"],
            output_fields=["latency_mean"],
            window_duration_seconds=60,
            lookback_steps=5,
            forecast_steps=2,
            hidden_size=32,
        )

    @pytest.fixture
    def sample_cell_data(self):
        """Sample cell data from data-storage API."""
        return [
            {
                "window_start_time": 1000 + i * 60,
                "window_end_time": 1060 + i * 60,
                "window_duration_seconds": 60.0,
                "cell_index": 0,
                "network": "5G",
                "rsrp_mean": -85.0 + i,
                "sinr_mean": 10.0 + i * 0.5,
                "latency_mean": 20.0 + i * 2,
            }
            for i in range(10)
        ]

    def test_extract_fields(self, training_service, sample_cell_data):
        """Test field extraction from window data."""
        input_fields = ["rsrp_mean", "sinr_mean"]
        output_fields = ["latency_mean"]

        input_data, output_data = training_service._extract_fields(
            sample_cell_data, input_fields, output_fields
        )

        # Check shapes
        assert len(input_data) == 10
        assert len(output_data) == 10
        assert len(input_data[0]) == 2  # 2 input fields
        assert len(output_data[0]) == 1  # 1 output field

        # Check first values
        assert input_data[0][0] == -85.0  # rsrp_mean
        assert input_data[0][1] == 10.0  # sinr_mean
        assert output_data[0][0] == 20.0  # latency_mean

    def test_prepare_sequences_basic(self, training_service, sample_cell_data):
        """Test sequence preparation with valid data."""
        input_fields = ["rsrp_mean", "sinr_mean"]
        output_fields = ["latency_mean"]
        lookback_steps = 5
        forecast_steps = 2

        X, y = training_service._prepare_sequences(
            sample_cell_data,
            input_fields,
            output_fields,
            lookback_steps,
            forecast_steps,
        )

        # With 10 windows, lookback=5, forecast=2:
        # Valid sequences: windows [0-4] -> [5-6], [1-5] -> [6-7], [2-6] -> [7-8]
        # Total: 10 - 5 - 2 + 1 = 4 sequences
        expected_sequences = 10 - lookback_steps - forecast_steps + 1
        assert X.shape == (expected_sequences, lookback_steps, 2)  # 2 input fields
        assert y.shape == (expected_sequences, forecast_steps, 1)  # 1 output field

        # Check first sequence
        # X[0] should be windows 0-4 input fields
        assert X[0, 0, 0] == -85.0  # Window 0, rsrp_mean
        assert X[0, 4, 0] == -81.0  # Window 4, rsrp_mean

        # y[0] should be windows 5-6 output fields
        assert y[0, 0, 0] == 30.0  # Window 5, latency_mean
        assert y[0, 1, 0] == 32.0  # Window 6, latency_mean

    def test_prepare_sequences_insufficient_data(
        self, training_service, sample_cell_data
    ):
        """Test sequence preparation with insufficient data."""
        input_fields = ["rsrp_mean"]
        output_fields = ["latency_mean"]
        lookback_steps = 20  # More than available
        forecast_steps = 2

        X, y = training_service._prepare_sequences(
            sample_cell_data,
            input_fields,
            output_fields,
            lookback_steps,
            forecast_steps,
        )

        # Should return empty arrays
        assert X.shape[0] == 0
        assert y.shape[0] == 0

    def test_prepare_sequences_empty_data(self, training_service):
        """Test sequence preparation with empty data."""
        X, y = training_service._prepare_sequences(
            [], ["rsrp_mean"], ["latency_mean"], 5, 2
        )

        assert X.shape[0] == 0
        assert y.shape[0] == 0

    def test_prepare_sequences_multiple_cells(self, training_service, sample_cell_data):
        """Test combining sequences from multiple cells."""
        # Create data for 2 cells
        cell_0_data = sample_cell_data
        cell_1_data = [{**window, "cell_index": 1} for window in sample_cell_data]

        all_sequences_X = []
        all_sequences_y = []

        for cell_data in [cell_0_data, cell_1_data]:
            X, y = training_service._prepare_sequences(
                cell_data, ["rsrp_mean"], ["latency_mean"], 5, 2
            )
            if len(X) > 0:
                all_sequences_X.append(X)
                all_sequences_y.append(y)

        # Combine sequences
        combined_X = np.concatenate(all_sequences_X, axis=0)
        combined_y = np.concatenate(all_sequences_y, axis=0)

        # Each cell produces 4 sequences, total 8
        assert combined_X.shape[0] == 8
        assert combined_y.shape[0] == 8

    def test_calculate_timestamps(self, training_service):
        """Test timestamp calculation from lookback_seconds."""
        lookback_seconds = 3600  # 1 hour

        with patch("src.services.data_preparation.datetime") as mock_datetime:
            mock_now = datetime(2026, 2, 2, 12, 0, 0)
            mock_datetime.now.return_value = mock_now

            start_ts, end_ts = training_service._calculate_timestamps(lookback_seconds)

            assert end_ts == int(mock_now.timestamp())
            assert start_ts == int(mock_now.timestamp()) - 3600

    def test_missing_fields_in_data(self, training_service):
        """Test handling of missing fields in data."""
        data = [
            {
                "window_start_time": 1000,
                "rsrp_mean": -85.0,
                # sinr_mean is missing
            }
        ]

        input_data, _ = training_service._extract_fields(
            data, ["rsrp_mean", "sinr_mean"], ["latency_mean"]
        )

        # Missing fields should be filled with 0.0
        assert input_data[0][1] == 0.0  # sinr_mean defaults to 0.0

    def test_data_type_conversion(self, training_service, sample_cell_data):
        """Test that data is properly converted to numpy arrays."""
        X, y = training_service._prepare_sequences(
            sample_cell_data, ["rsrp_mean"], ["latency_mean"], 5, 2
        )

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.dtype == np.float32
        assert y.dtype == np.float32
