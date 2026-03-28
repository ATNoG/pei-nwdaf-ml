"""Pydantic schemas for anomaly detection."""

from datetime import datetime

from pydantic import BaseModel, Field

from src.schemas.kube import JobResources

# ── Model configuration ──────────────────────────────────────────────


class AnomalyModelConfig(BaseModel):
    """Immutable anomaly model configuration."""

    input_fields: list[str] = Field(
        ...,
        min_length=1,
        description="List of input field names (e.g., ['latency_mean', 'throughput_mean'])",
    )
    window_duration_seconds: int = Field(
        ..., ge=1, description="Data window granularity in seconds"
    )
    hidden_size: int = Field(
        default=32, ge=4, description="Autoencoder hidden layer size"
    )
    threshold_percentile: float = Field(
        default=95.0,
        ge=0.0,
        le=100.0,
        description="Percentile of training reconstruction errors used as anomaly threshold",
    )


class AnomalyModelCreate(BaseModel):
    """Request schema for creating an anomaly model."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Unique model name (alphanumeric, underscore, hyphen only)",
    )
    config: AnomalyModelConfig = Field(..., description="Anomaly model configuration")


class AnomalyModelSummary(BaseModel):
    """Summary response for listing anomaly models."""

    id: str = Field(..., description="Model ID")
    name: str = Field(..., description="Model name")
    created_at: datetime | None = Field(None, description="Creation timestamp")
    latest_version: int | None = Field(
        None, description="Latest registered model version number"
    )


class AnomalyModelDetail(BaseModel):
    """Detailed response for an anomaly model."""

    id: str = Field(..., description="Model ID")
    name: str = Field(..., description="Model name")
    config: AnomalyModelConfig = Field(..., description="Full model configuration")
    threshold_value: float | None = Field(
        None, description="Computed anomaly threshold (set after training)"
    )
    created_at: datetime | None = Field(None, description="Creation timestamp")
    latest_version: int | None = Field(None, description="Latest version number")
    last_trained_at: datetime | None = Field(
        None, description="Last training completion timestamp"
    )
    mlflow_run_id: str | None = Field(
        None, description="MLflow run ID of the latest training"
    )
    training_loss: float | None = Field(
        None, description="Final training loss from latest run"
    )


# ── Training ─────────────────────────────────────────────────────────


class AnomalyTrainingRequest(BaseModel):
    """Request schema for anomaly model training."""

    model_id: str = Field(..., description="UUID of the anomaly model to train")
    lookback_seconds: int = Field(
        ..., gt=0, description="How far back to fetch data (e.g., 604800 for 7 days)"
    )
    resources: JobResources | None = Field(
        None, description="Resource requirements for the training job"
    )


class AnomalyTrainingResponse(BaseModel):
    """Response schema for anomaly training request."""

    job_id: str = Field(..., description="Unique training job ID")
    model_id: str = Field(..., description="Model ID being trained")
    status: str = Field(..., description="Training status")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(..., description="Job creation timestamp")


class AnomalyTrainingJobDetail(BaseModel):
    """Detailed information about an anomaly training job."""

    job_id: str = Field(..., description="Unique training job ID")
    model_id: str = Field(..., description="Model ID being trained")
    status: str = Field(..., description="Current status")
    mlflow_run_id: str | None = Field(
        None, description="MLflow run ID (available when training starts)"
    )
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: datetime | None = Field(None, description="Training start timestamp")
    completed_at: datetime | None = Field(
        None, description="Training completion timestamp"
    )
    error_message: str | None = Field(None, description="Error message if failed")


class AnomalyTrainingJobSummary(BaseModel):
    """Summary information for listing anomaly training jobs."""

    job_id: str
    model_id: str
    status: str
    created_at: datetime
    started_at: datetime | None = None


# ── Detection ────────────────────────────────────────────────────────


class AnomalyDetectionRequest(BaseModel):
    """Request schema for running anomaly detection."""

    cell_id: int = Field(..., ge=0, description="Cell index to analyse")
    model_id: str | None = Field(
        None,
        description="UUID of the trained anomaly model to use. "
        "If omitted, the system selects the compatible trained model with lowest training loss.",
    )
    lookback_seconds: int = Field(
        default=1800,
        gt=0,
        description="How far back to fetch data (seconds, default 30min)",
    )


class WindowScore(BaseModel):
    """Anomaly score for a single time window."""

    window_start_time: int = Field(..., description="Window start timestamp (epoch)")
    reconstruction_error: float = Field(
        ..., description="Reconstruction error for this window"
    )
    is_anomaly: bool = Field(..., description="Whether error exceeds the threshold")


class IPAnomalyResult(BaseModel):
    """Anomaly detection results for a single IP address."""

    ip_src: str = Field(..., description="Source IP address")
    num_windows: int = Field(..., description="Number of windows scored")
    num_anomalies: int = Field(..., description="Number of anomalous windows")
    scores: list[WindowScore] = Field(..., description="Per-window scores")


class AnomalyDetectionResult(BaseModel):
    """Full anomaly detection response for a cell."""

    model_id: str = Field(..., description="Model ID used for detection")
    model_name: str = Field(..., description="Human-readable model name")
    cell_id: int = Field(..., description="Cell index analysed")
    threshold_value: float = Field(..., description="Anomaly threshold used")
    window_duration_seconds: int = Field(..., description="Window duration in seconds")
    input_fields: list[str] = Field(..., description="Input fields used")
    results: list[IPAnomalyResult] = Field(
        ..., description="Per-IP anomaly detection results"
    )


class AnomalyModelMeta(BaseModel):
    """Metadata for a model used in a summary."""

    name: str
    fields: list[str]
    threshold: float
    window_duration_seconds: int


class AnomalyDetectionSummary(BaseModel):
    """Compact multi-model anomaly detection summary for a cell."""

    cell_id: int
    models: dict[str, AnomalyModelMeta] = Field(
        ..., description="Model metadata keyed by model name"
    )
    ip_anomalies: dict[str, dict[str, str]] = Field(
        ...,
        description="Per-IP anomaly counts keyed by IP then model name (e.g. '3/10')",
    )
