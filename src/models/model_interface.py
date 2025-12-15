from abc import ABC, abstractmethod
from typing import Any

class ModelInterface(ABC):
    """Base interface for all ML models"""

    FRAMEWORK: str = "generic"
    SEQUENCE_LENGTH: int = 1

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
    def predict(self, X: Any) -> float:
        """Predict a single scalar value"""
        pass
