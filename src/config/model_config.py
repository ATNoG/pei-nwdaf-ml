"""
Model configuration dataclasses for parameterizable model instantiation.
"""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class OptimizerType(str, Enum):
    """Supported optimizer types"""
    ADAM = "adam"
    SGD = "sgd"

class LossType(str, Enum):
    """Supported loss function types"""
    MSE = "mse"
    MAE = "mae"

class ActivationType(str, Enum):
    """Supported activation function types"""
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"

@dataclass
class TrainingConfig:
    """Training hyperparameters configuration"""
    learning_rate: float = 0.001
    optimizer: OptimizerType = OptimizerType.ADAM
    loss_function: LossType = LossType.MSE
    batch_size: int|None = None
    max_epochs: int = 50
    early_stopping_patience: int|None = None
    weight_decay: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer.value,
            "loss_function": self.loss_function.value,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "weight_decay": self.weight_decay,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        return cls(
            learning_rate=data.get("learning_rate", 0.001),
            optimizer=OptimizerType(data.get("optimizer", "adam")),
            loss_function=LossType(data.get("loss_function", "mse")),
            batch_size=data.get("batch_size"),
            max_epochs=data.get("max_epochs", 50),
            early_stopping_patience=data.get("early_stopping_patience"),
            weight_decay=data.get("weight_decay", 0.0),
        )


@dataclass
class ArchitectureConfig:
    """Model architecture configuration"""
    hidden_size: int = 32
    num_layers: int = 2
    dropout: float = 0.2
    activation: ActivationType = ActivationType.RELU
    hidden_layers: list[int]|None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "activation": self.activation.value,
            "hidden_layers": self.hidden_layers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArchitectureConfig":
        return cls(
            hidden_size=data.get("hidden_size", 32),
            num_layers=data.get("num_layers", 2),
            dropout=data.get("dropout", 0.2),
            activation=ActivationType(data.get("activation", "relu")),
            hidden_layers=data.get("hidden_layers"),
        )


@dataclass
class SequenceConfig:
    """Sequence configuration"""
    sequence_length: int = 5
    prediction_horizon: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "prediction_horizon": self.prediction_horizon,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SequenceConfig":
        return cls(
            sequence_length=data.get("sequence_length", 5),
            prediction_horizon=data.get("prediction_horizon", 1),
        )


@dataclass
class ModelConfig:
    """Complete model configuration combining all sub-configs"""
    training: TrainingConfig = field(default_factory=TrainingConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "training": self.training.to_dict(),
            "architecture": self.architecture.to_dict(),
            "sequence": self.sequence.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(
            training=TrainingConfig.from_dict(data.get("training", {})),
            architecture=ArchitectureConfig.from_dict(data.get("architecture", {})),
            sequence=SequenceConfig.from_dict(data.get("sequence", {})),
        )

    @classmethod
    def default(cls) -> "ModelConfig":
        """Return default configuration for backward compatibility"""
        return cls()
