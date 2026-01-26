from abc import ABC, abstractmethod
from typing import Any
import threading

class ModelInterface(ABC):
    """Base interface for all ML models"""

    FRAMEWORK: str = "generic"
    SEQUENCE_LENGTH: int = 1
    _training_lock: threading.RLock = None

    @classmethod
    def _get_training_lock(cls) -> threading.RLock:
        """Get or create the class-level training lock."""
        if cls._training_lock is None:
            cls._training_lock = threading.RLock()
        return cls._training_lock

    @classmethod
    def reserve_training(cls) -> bool:
        """Try to reserve this model class for training.
        
        Returns:
            True if reservation succeeded, False if already training.
        """
        lock = cls._get_training_lock()
        return lock.acquire(blocking=False)

    @classmethod
    def release_training(cls) -> None:
        """Release the training reservation for this model class."""
        lock = cls._get_training_lock()
        try:
            lock.release()
        except RuntimeError:
            # Lock was not acquired, ignore
            pass

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
