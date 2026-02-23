"""Router for model performance monitoring endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from src.core.dependencies import get_mlflow_client
from src.schemas.performance import FieldEvaluationResponse, ModelPerformance
from src.services.data_storage_client import DataStorageClient
from src.services.mlflow_service import MLflowService
from src.services.performance_service import PerformanceService

logger = logging.getLogger(__name__)
router = APIRouter()


def get_performance_service(
    mlflow_service: MLflowService = Depends(get_mlflow_client),
) -> PerformanceService:
    """Dependency for PerformanceService."""
    return PerformanceService(
        mlflow_service=mlflow_service,
        ml_config_service=mlflow_service.ml_config_service,
        data_storage_client=DataStorageClient(),
    )

@router.post("/{field_name}/evaluate", response_model=FieldEvaluationResponse)
async def evaluate_field(
    field_name: str,
    performance_service: PerformanceService = Depends(get_performance_service),
) -> FieldEvaluationResponse:
    """
    Score all trained models that predict field_name.

    Computes RMSE against live cell data, persists
    scores as MLflow tags, and designates the best model via the
    best_for:{field_name} tag. Returns a ranked list of ModelPerformance objects.
    """
    try:
        return await performance_service.evaluate_field(field_name)
    except Exception as e:
        logger.error(f"Evaluation failed for field '{field_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{field_name}/best", response_model=ModelPerformance)
async def get_best_model(
    field_name: str,
    performance_service: PerformanceService = Depends(get_performance_service),
) -> ModelPerformance:
    """
    Return the model currently designated as best for field_name.

    Reads the persisted best_for:{field_name} MLflow tag. Returns 404 if no
    evaluation has been run yet for this field.
    """
    try:
        best = performance_service.get_best_model(field_name)
        if best is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No best model designated for field '{field_name}'. "
                    f"Run POST /v1/performance/{field_name}/evaluate first."
                ),
            )
        return best
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get best model for field '{field_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{field_name}/monitor", response_model=ModelPerformance)
async def monitor_best_model(
    field_name: str,
    performance_service: PerformanceService = Depends(get_performance_service),
) -> ModelPerformance:
    """
    Re-evaluate only the current best model for field_name.

    Designed for repeated/scheduled calling to track score drift over time.
    Updates rmse_for and eval_at MLflow tags on the best model but does NOT
    change the best_for designation (only a full /evaluate call can change that).

    Returns 422 if no best model has been designated yet.
    """
    try:
        return await performance_service.monitor_best_model(field_name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Monitor failed for field '{field_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{field_name}", response_model=FieldEvaluationResponse)
async def get_evaluation(
    field_name: str,
    performance_service: PerformanceService = Depends(get_performance_service),
) -> FieldEvaluationResponse:
    """
    Return the last cached evaluation result for field_name.

    Reads persisted MLflow tags — no model loading or data fetching occurs.
    Models that have never been evaluated will appear with rmse=None.
    """
    try:
        return performance_service.get_cached_evaluation(field_name)
    except Exception as e:
        logger.error(f"Failed to get cached evaluation for field '{field_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))
