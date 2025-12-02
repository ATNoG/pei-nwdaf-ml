from abc import ABC, abstractmethod
from typing import Type

class ModelI(ABC):
    @abstractmethod
    def infer(self,**arg) -> ...:
        """
        Model receives data and
        produces an inference over it
        """
        raise NotImplementedError

    @abstractmethod
    def train(self,min_loss:float, max_epochs:int,**args) -> ...:
        """
        Model infers over provided data,
        manipulates its weights if loss is still over a threshold
        and returns the current loss when max epochs is reached or current loss is less than the provided
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
