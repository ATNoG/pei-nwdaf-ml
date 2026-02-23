"""Pydantic schemas for model configuration and management."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ArchitectureType(str, Enum):
    """Supported model architectures."""

    ANN = "ann"
    LSTM = "lstm"


class ModelConfig(BaseModel):
    """
    Immutable model configuration.

    Defines the model's structure, data shape, and window settings.
    Once created, this config cannot be changed.
    """

    architecture: ArchitectureType = Field(
        ..., description="Model architecture type"
    )
    input_fields: list[str] = Field(
        ...,
        min_length=1,
        description="List of input field names (e.g., ['latency_mean', 'throughput_mean'])",
    )
    output_fields: list[str] = Field(
        ...,
        min_length=1,
        description="List of output field names to predict (e.g., ['latency_mean'])",
    )
    window_duration_seconds: int = Field(
        ...,
        ge=1,
        description="Data window granularity in seconds (60 or 300)",
    )
    lookback_steps: int = Field(
        ...,
        ge=1,
        description="Number of past time windows to use as input (e.g., 30 windows)",
    )
    forecast_steps: int = Field(
        ...,
        ge=1,
        description="Number of future time windows to predict (e.g., 5 windows)",
    )
    hidden_size: int = Field(
        default=32,
        ge=4,
        description="Neural network hidden layer size",
    )


class ModelCreate(BaseModel):
    """Request schema for creating a new model."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Unique model name (alphanumeric, underscore, hyphen only)",
    )
    config: ModelConfig = Field(..., description="Model configuration")


class ModelSummary(BaseModel):
    """Summary response for listing models."""

    id: str = Field(..., description="Model ID ")
    name: str = Field(..., description="Model name")
    architecture: ArchitectureType = Field(..., description="Model architecture")
    created_at: datetime|None = Field(
        None, description="Model creation timestamp"
    )
    latest_version: int|None = Field(
        None, description="Latest registered model version number"
    )


class ModelDetail(BaseModel):
    """Detailed response for a specific model."""

    id: str = Field(..., description="Model ID")
    name: str = Field(..., description="Model name")
    config: ModelConfig = Field(..., description="Full model configuration")
    created_at: datetime|None = Field(None, description="Creation timestamp")
    latest_version: int|None = Field(None, description="Latest version number")
    last_trained_at: datetime|None = Field(
        None, description="Last training completion timestamp"
    )
    mlflow_run_id: str|None = Field(
        None, description="MLflow run ID of the latest training"
    )
    training_loss: float|None = Field(
        None, description="Final training loss from latest run"
    )
