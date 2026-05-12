"""Pydantic schemas for model performance monitoring."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from src.schemas.tags import Tags


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
    architecture: str = Field(..., description="Model architecture")
    latest_version: int | None = Field(None, description="Latest trained version number")
    training_loss: float | None = Field(None, description="Final MSE from last training run")
    score: float | None = Field(
        None,
        description="Evaluation score for the queried field; None if never evaluated",
    )
    metric: str | None = Field(None, description="Which metric produced the score (e.g. 'rmse')")
    is_best: bool = Field(False, description="True for the model with the best score for this field")
    baseline_score: float | None = Field(None, description="Score at election time - used as degradation reference")
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


class FeatureImportanceValue(BaseModel):
    """Mean and standard deviation of permutation importance across n_repeats shuffles."""

    mean: float = Field(..., description="Mean score degradation across n_repeats shuffles")
    std: float = Field(..., description="Std deviation across n_repeats shuffles")


class FeatureImportanceResponse(BaseModel):
    """Permutation importance scores for each input feature of the best model."""

    field_name: str
    model_id: str
    importances: dict[str, FeatureImportanceValue] = Field(
        ...,
        description="Per-feature permutation importance."
                    "mean: score degradation when feature is shuffled."
                    "std: stability of the estimate across repeats.",
    )
    metric: str = Field(..., description="Metric used")
    computed_at: datetime | None = Field(None, description="When importance was last computed")


class LocalExplanationResponse(BaseModel):
    """Local explanation for a single prediction."""

    field_name: str
    model_id: str
    method: str = Field("kernelshap", description="Explainer used")
    tags: Tags
    prediction: list[float] = Field(..., description="Forecast values for the explained field across forecast steps",)
    attributions: dict[str, float] = Field(
        ...,
        description="Per-feature SHAP value."
                    "Positive = feature pushed prediction above baseline"
                    "Negative = feature pushed prediction below baseline"
                    "Sum of all attributions = prediction_mean - baseline",
    )
    baseline: float = Field(..., description="Expected model output over training background dataset",)
    computed_at: datetime
