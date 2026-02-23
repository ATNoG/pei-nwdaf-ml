"""Pydantic schemas for model performance monitoring."""

from datetime import datetime

from pydantic import BaseModel, Field

from src.schemas.model import ArchitectureType


class ModelPerformance(BaseModel):
    """Performance info for a single model on a specific output field."""

    model_id: str = Field(..., description="Model UUID")
    model_name: str = Field(..., description="Human-readable model name")
    architecture: ArchitectureType = Field(..., description="Model architecture")
    latest_version: int | None = Field(None, description="Latest trained version number")
    training_loss: float | None = Field(None, description="Final MSE from last training run")
    rmse: float | None = Field(
        None,
        description="RMSE for the queried field computed against live cell data; None if never evaluated",
    )
    is_best: bool = Field(False, description="True for the model with the lowest RMSE for this field")
    last_trained_at: datetime | None = Field(None, description="Last training completion timestamp")
    evaluated_at: datetime | None = Field(None, description="When RMSE was last computed")


class FieldEvaluationResponse(BaseModel):
    """Evaluation results for all models predicting a given output field."""

    field_name: str = Field(..., description="The output field being evaluated")
    models: list[ModelPerformance] = Field(
        ...,
        description="Models sorted: scored ascending by RMSE first, unscored last",
    )
    best_model_id: str | None = Field(
        None, description="Model ID with the lowest RMSE; None if no evaluation has run"
    )
    last_evaluated_at: datetime | None = Field(
        None, description="Timestamp of the most recent evaluation"
    )
