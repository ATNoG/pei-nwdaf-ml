from abc import ABC, abstractmethod
from typing import Any

class ModelInterface(ABC):
    """Base interface for all ML models"""

    def __init__(
        self,
        input_fields: list[str],
        output_fields: list[str],
        window_duration_seconds: int,
        lookback_steps: int,
        forecast_steps: int,
        hidden_size: int
    ):
        """
        Initialize model with configuration parameters.

        Args:
            input_fields: List of input field names
            output_fields: List of output field names
            window_duration_seconds: Data window granularity in seconds
            lookback_steps: Number of past time windows to use as input
            forecast_steps: Number of future time windows to predict
            hidden_size: Neural network hidden layer size
        """
        self.input_fields = input_fields
        self.output_fields = output_fields
        self.window_duration_seconds = window_duration_seconds
        self.lookback_steps = lookback_steps
        self.forecast_steps = forecast_steps
        self.hidden_size = hidden_size

    @abstractmethod
    def train(self, X: Any, y: Any, max_epochs: int = 100, status_callback=None) -> float:
        """
        Train the model

        Args:
            X: Training features
            y: Training targets
            max_epochs: Maximum number of epochs
            status_callback: Optional callback function(current_epoch, total_epochs, loss) for status updates

        Returns:
            Final training loss
        """
        pass

    @abstractmethod
    def predict(self, X: Any):
        """
        Predict output values.

        Args:
            X: Input data (numpy array)

        Returns:
            Predictions (numpy array)
        """
        pass
