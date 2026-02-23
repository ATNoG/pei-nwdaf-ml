"""Training request/response schemas."""

from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class TrainingJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TrainingRequest(BaseModel):
    """Request schema for model training."""

    model_id: str = Field(..., description="UUID of the model to train")
    lookback_seconds: int = Field(..., gt=0, description="How far back to fetch data (e.g., 604800 for 7 days)")


class TrainingResponse(BaseModel):
    """Response schema for training request."""

    job_id: str = Field(..., description="Unique training job ID")
    model_id: str = Field(..., description="Model ID being trained")
    status: str = Field(..., description="Training status (queued, running, completed, failed, cancelled)")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(..., description="Training job creation timestamp")


class TrainingJobDetail(BaseModel):
    """Detailed information about a training job."""

    job_id: str = Field(..., description="Unique training job ID")
    model_id: str = Field(..., description="Model ID being trained")
    status: str = Field(..., description="Current status")
    mlflow_run_id: str | None = Field(None, description="MLflow run ID (available when training starts)")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: datetime | None = Field(None, description="Training start timestamp")
    completed_at: datetime | None = Field(None, description="Training completion timestamp")
    error_message: str | None = Field(None, description="Error message if failed")


class TrainingJobSummary(BaseModel):
    """Summary information for listing jobs."""

    job_id: str
    model_id: str
    status: str
    created_at: datetime
    started_at: datetime | None = None
