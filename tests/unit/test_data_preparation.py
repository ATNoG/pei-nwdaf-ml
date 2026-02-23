"""Unit tests for data preparation utilities."""

import pytest
import numpy as np
from unittest.mock import patch
from datetime import datetime

from src.services.data_preparation import (
    calculate_timestamps,
    extract_fields,
    prepare_sequences,
    prepare_last_sequence,
)


class TestCalculateTimestamps:
    """Tests for calculate_timestamps function."""

    @patch("src.services.data_preparation.datetime")
    def test_returns_correct_range(self, mock_datetime):
        """Test that start and end timestamps have the correct difference."""
        fixed_time = datetime(2026, 2, 2, 12, 0, 0)
        mock_datetime.now.return_value = fixed_time

        lookback = 3600
        start_ts, end_ts = calculate_timestamps(lookback)

        assert end_ts == int(fixed_time.timestamp())
        assert start_ts == end_ts - lookback

    @patch("src.services.data_preparation.datetime")
    def test_returns_ints(self, mock_datetime):
        """Test that both timestamps are integers."""
        mock_datetime.now.return_value = datetime(2026, 2, 2, 12, 0, 0)

        start_ts, end_ts = calculate_timestamps(600)

        assert isinstance(start_ts, int)
        assert isinstance(end_ts, int)

    @patch("src.services.data_preparation.datetime")
    def test_zero_lookback(self, mock_datetime):
        """Test that zero lookback returns equal start and end."""
        mock_datetime.now.return_value = datetime(2026, 2, 2, 12, 0, 0)

        start_ts, end_ts = calculate_timestamps(0)

        assert start_ts == end_ts


class TestExtractFields:
    """Tests for extract_fields function."""

    @pytest.fixture
    def sample_data(self):
        return [
            {"rsrp_mean": -85.0, "sinr_mean": 10.0, "latency_mean": 20.0},
            {"rsrp_mean": -80.0, "sinr_mean": 12.0, "latency_mean": 25.0},
        ]

    def test_extracts_correct_values(self, sample_data):
        """Test that input and output values are extracted correctly."""
        inputs, outputs = extract_fields(
            sample_data,
            input_fields=["rsrp_mean", "sinr_mean"],
            output_fields=["latency_mean"],
        )

        assert inputs == [[-85.0, 10.0], [-80.0, 12.0]]
        assert outputs == [[20.0], [25.0]]

    def test_missing_field_defaults_to_zero(self):
        """Test that missing fields default to 0.0."""
        data = [{"rsrp_mean": -85.0}]

        inputs, outputs = extract_fields(
            data,
            input_fields=["rsrp_mean", "missing_field"],
            output_fields=["also_missing"],
        )

        assert inputs == [[-85.0, 0.0]]
        assert outputs == [[0.0]]

    def test_empty_data(self):
        """Test that empty data returns empty lists."""
        inputs, outputs = extract_fields(
            [],
            input_fields=["rsrp_mean"],
            output_fields=["latency_mean"],
        )

        assert inputs == []
        assert outputs == []

    def test_multiple_output_fields(self, sample_data):
        """Test extraction with multiple output fields."""
        inputs, outputs = extract_fields(
            sample_data,
            input_fields=["rsrp_mean"],
            output_fields=["latency_mean", "sinr_mean"],
        )

        assert outputs == [[20.0, 10.0], [25.0, 12.0]]


class TestPrepareSequences:
    """Tests for prepare_sequences function."""

    @pytest.fixture
    def cell_data(self):
        """10 windows with 3 fields."""
        return [
            {
                "window_start_time": 1000 + i * 60,
                "rsrp_mean": float(i),
                "sinr_mean": float(i * 10),
                "latency_mean": float(i * 100),
            }
            for i in range(10)
        ]

    def test_correct_shapes(self, cell_data):
        """Test X and y have correct shapes for 10 windows, lookback=5, forecast=2."""
        X, y = prepare_sequences(
            cell_data,
            input_fields=["rsrp_mean", "sinr_mean", "latency_mean"],
            output_fields=["latency_mean"],
            lookback_steps=5,
            forecast_steps=2,
        )

        # num_sequences = 10 - 5 - 2 + 1 = 4
        assert X.shape == (4, 5, 3)
        assert y.shape == (4, 2, 1)

    def test_insufficient_data_returns_empty(self):
        """Test that insufficient data returns empty arrays with correct trailing dims."""
        data = [{"rsrp_mean": 1.0, "latency_mean": 2.0} for _ in range(3)]

        X, y = prepare_sequences(
            data,
            input_fields=["rsrp_mean"],
            output_fields=["latency_mean"],
            lookback_steps=5,
            forecast_steps=2,
        )

        assert X.shape[0] == 0
        assert y.shape[0] == 0
        assert X.shape == (0, 5, 1)
        assert y.shape == (0, 2, 1)

    def test_exact_minimum_data(self):
        """Test that exactly lookback + forecast windows produces 1 sequence."""
        data = [{"rsrp_mean": float(i), "latency_mean": float(i)} for i in range(7)]

        X, y = prepare_sequences(
            data,
            input_fields=["rsrp_mean"],
            output_fields=["latency_mean"],
            lookback_steps=5,
            forecast_steps=2,
        )

        assert X.shape[0] == 1

    def test_dtype_float32(self, cell_data):
        """Test that output arrays are float32."""
        X, y = prepare_sequences(
            cell_data,
            input_fields=["rsrp_mean"],
            output_fields=["latency_mean"],
            lookback_steps=3,
            forecast_steps=2,
        )

        assert X.dtype == np.float32
        assert y.dtype == np.float32

    def test_values_correct(self, cell_data):
        """Test that the first sequence contains correct values."""
        X, y = prepare_sequences(
            cell_data,
            input_fields=["rsrp_mean"],
            output_fields=["latency_mean"],
            lookback_steps=3,
            forecast_steps=2,
        )

        # First sequence: windows 0,1,2 for input, windows 3,4 for output
        assert X[0, 0, 0] == 0.0  # rsrp_mean of window 0
        assert X[0, 2, 0] == 2.0  # rsrp_mean of window 2
        assert y[0, 0, 0] == 300.0  # latency_mean of window 3
        assert y[0, 1, 0] == 400.0  # latency_mean of window 4


class TestPrepareLastSequence:
    """Tests for prepare_last_sequence function."""

    @pytest.fixture
    def cell_data(self):
        """10 windows with known values."""
        return [
            {
                "window_start_time": 1000 + i * 60,
                "rsrp_mean": float(i),
                "sinr_mean": float(i * 10),
                "latency_mean": float(i * 100),
            }
            for i in range(10)
        ]

    def test_correct_shape(self, cell_data):
        """Test output shape is (1, lookback, num_inputs)."""
        X = prepare_last_sequence(
            cell_data,
            input_fields=["rsrp_mean", "sinr_mean", "latency_mean"],
            lookback_steps=5,
        )

        assert X.shape == (1, 5, 3)

    def test_takes_last_windows(self, cell_data):
        """Test that the last N windows are used, not the first N."""
        X = prepare_last_sequence(
            cell_data,
            input_fields=["rsrp_mean"],
            lookback_steps=3,
        )

        # Last 3 windows are indices 7, 8, 9 with rsrp_mean = 7.0, 8.0, 9.0
        assert X[0, 0, 0] == 7.0
        assert X[0, 1, 0] == 8.0
        assert X[0, 2, 0] == 9.0

    def test_insufficient_data_raises(self):
        """Test that fewer windows than lookback raises ValueError."""
        data = [{"rsrp_mean": 1.0} for _ in range(3)]

        with pytest.raises(ValueError, match="Insufficient data"):
            prepare_last_sequence(
                data,
                input_fields=["rsrp_mean"],
                lookback_steps=5,
            )

    def test_empty_data_raises(self):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="Insufficient data"):
            prepare_last_sequence(
                [],
                input_fields=["rsrp_mean"],
                lookback_steps=1,
            )

    def test_exact_data(self):
        """Test that exactly lookback_steps windows works."""
        data = [{"rsrp_mean": float(i)} for i in range(5)]

        X = prepare_last_sequence(
            data,
            input_fields=["rsrp_mean"],
            lookback_steps=5,
        )

        assert X.shape == (1, 5, 1)

    def test_missing_fields_default_zero(self):
        """Test that missing fields default to 0.0."""
        data = [{"rsrp_mean": 1.0} for _ in range(3)]

        X = prepare_last_sequence(
            data,
            input_fields=["rsrp_mean", "missing_field"],
            lookback_steps=3,
        )

        # rsrp_mean should be 1.0, missing_field should be 0.0
        assert X[0, 0, 0] == 1.0
        assert X[0, 0, 1] == 0.0

    def test_dtype_float32(self, cell_data):
        """Test that output is float32."""
        X = prepare_last_sequence(
            cell_data,
            input_fields=["rsrp_mean"],
            lookback_steps=3,
        )

        assert X.dtype == np.float32
