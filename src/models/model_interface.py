from abc import ABC, abstractmethod
from typing import Any

class ModelInterface(ABC):
    """Base interface for all ML models"""

    FRAMEWORK: str = "generic"
    SEQUENCE_LENGTH: int = 1

    @abstractmethod
    def train(self, X: Any, y: Any, max_epochs: int = 100) -> float:
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X: Any) -> float:
        """Predict a single scalar value"""
        pass

    @abstractmethod
    def serialize(self) -> bytes:
        """Serialize model to bytes"""
        pass

    @classmethod
    @abstractmethod
    def deserialize(cls, b: bytes) -> "ModelInterface":
        """Deserialize model from bytes"""
        pass
