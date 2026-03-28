"""Router for model training endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.core.dependencies import get_db, get_mlflow_client
from src.schemas.training import (
    TrainingJobDetail,
    TrainingJobSummary,
    TrainingRequest,
    TrainingResponse,
)
from src.services.data_storage_client import DataStorageClient
from src.services.mlflow_service import MLflowService
from src.services.training_service import TrainingService

logger = logging.getLogger(__name__)
router = APIRouter()


def get_training_service(
    mlflow_service: MLflowService = Depends(get_mlflow_client),
    db: Session = Depends(get_db),
) -> TrainingService:
    """Dependency for TrainingService."""
    data_storage_client = DataStorageClient()
    return TrainingService(
        mlflow_service=mlflow_service,
        data_storage_client=data_storage_client,
        ml_config_service=mlflow_service.ml_config_service,
        db=db,
    )


@router.post("/train", response_model=TrainingResponse, status_code=202)
async def train_model(
    request: TrainingRequest,
    training_service: TrainingService = Depends(get_training_service),
) -> TrainingResponse:
    try:
        job_info = training_service.create_training_job(
            request.model_id, request.lookback_seconds
        )
        training_service.dispatch(job_info["job_id"], request.resources)
        logger.info(f"Training job {job_info['job_id']} dispatched")

        return TrainingResponse(
            job_id=job_info["job_id"],
            model_id=job_info["model_id"],
            status=job_info["status"],
            message="Training job queued successfully. Check status at GET /training/jobs/{job_id}",
            created_at=job_info["created_at"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create training job: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create training job: {str(e)}"
        )


@router.get("/jobs", response_model=list[TrainingJobSummary])
async def list_training_jobs(
    model_id: str | None = None,
    status: str | None = None,
    training_service: TrainingService = Depends(get_training_service),
) -> list[TrainingJobSummary]:
    """
    List training jobs with optional filters.

    Args:
        model_id: Filter by model ID
        status: Filter by status (queued, running, completed, failed, cancelled)
        training_service: Training service dependency

    Returns:
        List of training job summaries
    """
    try:
        jobs = training_service.list_jobs(model_id=model_id, status=status)
        return [
            TrainingJobSummary(
                job_id=job.job_id,
                model_id=job.model_id,
                status=job.status,
                created_at=job.created_at,
                started_at=job.started_at,
            )
            for job in jobs
        ]
    except Exception as e:
        logger.error(f"Failed to list jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@router.get("/jobs/{job_id}", response_model=TrainingJobDetail)
async def get_training_job(
    job_id: str,
    training_service: TrainingService = Depends(get_training_service),
) -> TrainingJobDetail:
    """
    Get detailed information about a specific training job.

    Args:
        job_id: Training job ID
        training_service: Training service dependency

    Returns:
        Detailed job information
    """
    try:
        job = training_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        return TrainingJobDetail(
            job_id=job.job_id,
            model_id=job.model_id,
            status=job.status,
            mlflow_run_id=job.mlflow_run_id,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job: {str(e)}")


@router.delete("/jobs/{job_id}", status_code=204)
async def cancel_training_job(
    job_id: str,
    training_service: TrainingService = Depends(get_training_service),
):
    """
    Cancel a training job.

    Args:
        job_id: Training job ID to cancel
        training_service: Training service dependency

    Returns:
        No content (204)
    """
    try:
        training_service.cancel_job(job_id)
        logger.info(f"Training job {job_id} cancelled")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to cancel job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")
