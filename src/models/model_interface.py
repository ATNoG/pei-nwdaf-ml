from abc import ABC, abstractmethod
from typing import Any
import threading

class ModelInterface(ABC):
    """Base interface for all ML models"""

    FRAMEWORK: str = "generic"
    SEQUENCE_LENGTH: int = 1
    _is_training: bool = False
    _lock: threading.Lock = None

    @classmethod
    def _get_lock(cls) -> threading.Lock:
        """Get or create the class-level lock for thread-safe flag access."""
        if cls._lock is None:
            cls._lock = threading.Lock()
        return cls._lock

    @classmethod
    def reserve_training(cls) -> bool:
        """Try to reserve this model class for training.
        
        Returns:
            True if reservation succeeded, False if already training.
        """
        lock = cls._get_lock()
        with lock:
            if cls._is_training:
                return False
            cls._is_training = True
            return True

    @classmethod
    def release_training(cls) -> None:
        """Release the training reservation for this model class."""
        lock = cls._get_lock()
        with lock:
            cls._is_training = False

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
