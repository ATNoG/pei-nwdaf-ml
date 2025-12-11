"""
Author: T. Vicente
"""

from abc import ABC, abstractmethod
from typing import Type

class ModelI(ABC):
    @abstractmethod
    def predict(self,**arg) -> ...:
        """
        Model receives data and
        produces an inference over it
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, max_epochs:int, **args) -> ...:
        """
        Model trains over provided data and returns the final loss
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
