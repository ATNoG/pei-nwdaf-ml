"""Shared data preparation utilities for training and inference."""

import logging
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


def calculate_timestamps(lookback_seconds: int) -> tuple[int, int]:
    """
    Calculate start and end timestamps from lookback duration.

    Args:
        lookback_seconds: Duration to look back from now.

    Returns:
        Tuple of (start_timestamp, end_timestamp) as epoch ints.
    """
    end_time = datetime.now()
    start_time = end_time.timestamp() - lookback_seconds
    return (int(start_time), int(end_time.timestamp()))


def extract_fields(
    data: list[dict],
    input_fields: list[str],
    output_fields: list[str],
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Extract input and output field values from data windows.

    Args:
        data: List of data windows.
        input_fields: Fields to use as inputs.
        output_fields: Fields to use as outputs.

    Returns:
        Tuple of (input_data, output_data) as lists of lists.
    """
    input_data = []
    output_data = []

    for window in data:
        input_row = [float(window.get(field, 0.0)) for field in input_fields]
        input_data.append(input_row)

        output_row = [float(window.get(field, 0.0)) for field in output_fields]
        output_data.append(output_row)

    return input_data, output_data


def prepare_sequences(
    cell_data: list[dict],
    input_fields: list[str],
    output_fields: list[str],
    lookback_steps: int,
    forecast_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window sequences from cell data.

    Args:
        cell_data: Sorted list of data windows for one cell.
        input_fields: Input feature field names.
        output_fields: Output target field names.
        lookback_steps: Number of past windows for input.
        forecast_steps: Number of future windows for output.

    Returns:
        Tuple of (X, y) numpy arrays.
        X shape: (num_sequences, lookback_steps, num_input_fields)
        y shape: (num_sequences, forecast_steps, num_output_fields)
    """
    input_data, output_data = extract_fields(cell_data, input_fields, output_fields)

    input_arr = np.array(input_data, dtype=np.float32)
    output_arr = np.array(output_data, dtype=np.float32)

    min_length = lookback_steps + forecast_steps
    if len(input_arr) < min_length:
        return (
            np.empty((0, lookback_steps, len(input_fields)), dtype=np.float32),
            np.empty((0, forecast_steps, len(output_fields)), dtype=np.float32),
        )

    X_sequences = []
    y_sequences = []

    for i in range(len(input_arr) - lookback_steps - forecast_steps + 1):
        X_seq = input_arr[i : i + lookback_steps]
        y_seq = output_arr[i + lookback_steps : i + lookback_steps + forecast_steps]
        X_sequences.append(X_seq)
        y_sequences.append(y_seq)

    return np.array(X_sequences, dtype=np.float32), np.array(
        y_sequences, dtype=np.float32
    )


def prepare_last_sequence(
    cell_data: list[dict],
    input_fields: list[str],
    lookback_steps: int,
) -> np.ndarray:
    """
    Prepare the most recent input sequence for inference.

    Takes the last lookback_steps windows and extracts input fields.
    Returns a single sequence ready for model.predict().

    Args:
        cell_data: Sorted list of data windows (ascending by time).
        input_fields: Input feature field names.
        lookback_steps: Number of past windows needed.

    Returns:
        numpy array of shape (1, lookback_steps, num_input_fields)

    Raises:
        ValueError: If insufficient data windows are available.
    """
    if len(cell_data) < lookback_steps:
        raise ValueError(
            f"Insufficient data: got {len(cell_data)} windows, "
            f"need at least {lookback_steps}"
        )

    recent_windows = cell_data[-lookback_steps:]

    input_data = []
    for window in recent_windows:
        input_row = [float(window.get(field, 0.0)) for field in input_fields]
        input_data.append(input_row)

    input_arr = np.array(input_data, dtype=np.float32)
    return input_arr[np.newaxis, :]
