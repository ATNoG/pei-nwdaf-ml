"""Pydantic schemas for model performance monitoring."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from src.schemas.model import ArchitectureType


class EvalMetricType(str, Enum):
    """Supported evaluation metrics for model performance scoring."""

    RMSE = "rmse"
    MAE = "mae"
    MAPE = "mape"
    R2 = "r2"


class ModelPerformance(BaseModel):
    """Performance info for a single model on a specific output field."""

    model_id: str = Field(..., description="Model UUID")
    model_name: str = Field(..., description="Human-readable model name")
    architecture: ArchitectureType = Field(..., description="Model architecture")
    latest_version: int | None = Field(None, description="Latest trained version number")
    training_loss: float | None = Field(None, description="Final MSE from last training run")
    score: float | None = Field(
        None,
        description="Evaluation score for the queried field; None if never evaluated",
    )
    metric: str | None = Field(None, description="Which metric produced the score (e.g. 'rmse')")
    is_best: bool = Field(False, description="True for the model with the best score for this field")
    last_trained_at: datetime | None = Field(None, description="Last training completion timestamp")
    evaluated_at: datetime | None = Field(None, description="When the score was last computed")


class FieldEvaluationResponse(BaseModel):
    """Evaluation results for all models predicting a given output field."""

    field_name: str = Field(..., description="The output field being evaluated")
    models: list[ModelPerformance] = Field(
        ...,
        description="Models sorted: best score first, unscored last",
    )
    best_model_id: str | None = Field(
        None, description="Model ID with the best score; None if no evaluation has run"
    )
    last_evaluated_at: datetime | None = Field(
        None, description="Timestamp of the most recent evaluation"
    )


class ScoreHistoryEntry(BaseModel):
    """A single score measurement for a model on a specific field."""

    model_id: str
    field_name: str
    score: float
    metric: str
    measured_at: datetime
    trigger: str = Field(..., description="What triggered this measurement: evaluate | monitor | auto_monitor")


class ScoreHistoryResponse(BaseModel):
    """Full score history for a field, ordered oldest to newest."""

    field_name: str
    entries: list[ScoreHistoryEntry]
