"""
Author: T. Vicente
"""

from abc import ABC, abstractmethod
from typing import Type
import numpy as np

class ModelI(ABC):
    SEQUENCE_LENGTH = 1
    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Model receives data and
        produces an inference over it

        Args:
            data: Input features for prediction

        Returns:
            Predictions as numpy array
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, max_epochs: int, X: np.ndarray, y: np.ndarray) -> float:
        """
        Model trains over provided data and returns the final loss

        Args:
            max_epochs: Number of training epochs
            X: Training features
            y: Training targets

        Returns:
            Final loss value
        """
        raise NotImplementedError

    @abstractmethod
    def serialize(self)-> bytes:
        """
        Serializes the model to bytes using pickle
        Returns:
            bytes: The pickle model object with a header
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def deserialize(cls:Type['ModelI'],b:bytes) -> 'ModelI':
        """
        Load a model from serialized bytes
        Returns:
            ModelI: an instance loaded from the provided bytes
        Raises:
            TypeError: if the bytes header does not match the model
        """
        raise NotImplementedError
