"""
Pydantic schemas for model configuration in API requests.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class OptimizerType(str, Enum):
    """Supported optimizer types"""
    adam = "adam"
    sgd = "sgd"
    adamw = "adamw"
    rmsprop = "rmsprop"


class LossType(str, Enum):
    """Supported loss function types"""
    mse = "mse"
    mae = "mae"
    huber = "huber"


class ActivationType(str, Enum):
    """Supported activation function types"""
    relu = "relu"
    tanh = "tanh"
    sigmoid = "sigmoid"
    leaky_relu = "leaky_relu"


class TrainingConfigSchema(BaseModel):
    """Training hyperparameters for API requests"""
    learning_rate: float = Field(default=0.001, ge=0.0, le=1.0, description="Learning rate for optimizer")
    optimizer: OptimizerType = Field(default=OptimizerType.adam, description="Optimizer type")
    loss_function: LossType = Field(default=LossType.mse, description="Loss function type")
    batch_size: Optional[int] = Field(default=None, ge=1, description="Batch size (None for auto)")
    max_epochs: int = Field(default=50, ge=1, le=1000, description="Maximum training epochs")
    early_stopping_patience: Optional[int] = Field(default=None, ge=1, description="Early stopping patience")
    weight_decay: float = Field(default=0.0, ge=0.0, description="Weight decay for regularization")

    class Config:
        json_schema_extra = {
            "example": {
                "learning_rate": 0.001,
                "optimizer": "adam",
                "loss_function": "mse",
                "max_epochs": 100
            }
        }


class ArchitectureConfigSchema(BaseModel):
    """Model architecture configuration for API requests"""
    hidden_size: int = Field(default=32, ge=1, description="Hidden layer size")
    num_layers: int = Field(default=2, ge=1, le=10, description="Number of layers (for LSTM)")
    dropout: float = Field(default=0.2, ge=0.0, le=0.9, description="Dropout rate")
    activation: ActivationType = Field(default=ActivationType.relu, description="Activation function")
    hidden_layers: Optional[List[int]] = Field(default=None, description="Custom layer sizes for ANN")

    class Config:
        json_schema_extra = {
            "example": {
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.3,
                "activation": "relu"
            }
        }


class SequenceConfigSchema(BaseModel):
    """Sequence/temporal configuration for API requests"""
    sequence_length: int = Field(default=5, ge=1, le=100, description="Input sequence length")
    prediction_horizon: int = Field(default=1, ge=1, description="Steps ahead to predict")
    stride: int = Field(default=1, ge=1, description="Window stride for sequences")

    class Config:
        json_schema_extra = {
            "example": {
                "sequence_length": 10,
                "prediction_horizon": 1
            }
        }


class ModelConfigSchema(BaseModel):
    """
    Complete model configuration for API requests.

    All fields are optional - omitted fields use sensible defaults.
    """
    training: Optional[TrainingConfigSchema] = None
    architecture: Optional[ArchitectureConfigSchema] = None
    sequence: Optional[SequenceConfigSchema] = None

    class Config:
        json_schema_extra = {
            "example": {
                "training": {
                    "learning_rate": 0.001,
                    "optimizer": "adam",
                    "max_epochs": 100
                },
                "architecture": {
                    "hidden_size": 64,
                    "num_layers": 2,
                    "dropout": 0.2
                },
                "sequence": {
                    "sequence_length": 5
                }
            }
        }

    def to_model_config(self):
        """Convert Pydantic schema to dataclass ModelConfig"""
        from src.config.model_config import (
            ModelConfig, TrainingConfig, ArchitectureConfig, SequenceConfig,
            OptimizerType as OptType, LossType as LType, ActivationType as ActType
        )

        training = TrainingConfig()
        if self.training:
            training = TrainingConfig(
                learning_rate=self.training.learning_rate,
                optimizer=OptType(self.training.optimizer.value),
                loss_function=LType(self.training.loss_function.value),
                batch_size=self.training.batch_size,
                max_epochs=self.training.max_epochs,
                early_stopping_patience=self.training.early_stopping_patience,
                weight_decay=self.training.weight_decay,
            )

        architecture = ArchitectureConfig()
        if self.architecture:
            architecture = ArchitectureConfig(
                hidden_size=self.architecture.hidden_size,
                num_layers=self.architecture.num_layers,
                dropout=self.architecture.dropout,
                activation=ActType(self.architecture.activation.value),
                hidden_layers=self.architecture.hidden_layers,
            )

        sequence = SequenceConfig()
        if self.sequence:
            sequence = SequenceConfig(
                sequence_length=self.sequence.sequence_length,
                prediction_horizon=self.sequence.prediction_horizon,
            )

        return ModelConfig(
            training=training,
            architecture=architecture,
            sequence=sequence,
        )
