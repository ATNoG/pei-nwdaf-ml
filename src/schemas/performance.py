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
    baseline_score: float | None = Field(None, description="Score at election time — used as degradation reference")
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


class MonitoringStatusResponse(BaseModel):
    """Current state machine status for a monitored field."""

    field_name: str
    state: str = Field(..., description="Current state: monitoring | retraining | evaluating")
    active_job_ids: list[str] = Field(default_factory=list, description="Training job IDs tracked during retraining")
    last_checked_at: datetime | None = Field(None, description="When the last successful monitoring cycle completed")
    monitoring_enabled: bool
    monitoring_interval_seconds: int
    monitoring_degradation_factor: float


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


class FeatureImportanceResponse(BaseModel):
    """Permutation importance scores for each input feature of the best model."""

    field_name: str
    model_id: str
    importances: dict[str, float] = Field(
        ...,
        description="Per-feature permutation importance (unbounded, in metric units) "
                    "Positive = shuffling this feature worsened the score (feature is useful) "
                    "Negative = shuffling this feature improved the score (model is using it counterproductively) "
                    "Near zero = feature has little influence on predictions.",
    )
    metric: str = Field(..., description="Metric used")
    computed_at: datetime | None = Field(None, description="When importance was last computed")
