"""Router for model performance monitoring endpoints."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.core.dependencies import get_mlflow_client
from src.db.database import get_db
from src.core.config import settings
from src.core.monitoring_state import get_field_jobs, get_field_state, get_last_checked, set_field_state, set_last_checked
from src.schemas.performance import (
    EvalMetricType,
    FeatureImportanceResponse,
    FieldEvaluationResponse,
    ModelPerformance,
    MonitoringStatusResponse,
    ScoreHistoryResponse,
)
from src.services.data_storage_client import DataStorageClient
from src.services.mlflow_service import MLflowService
from src.services.performance_service import PerformanceService

logger = logging.getLogger(__name__)
router = APIRouter()


def get_performance_service(
    mlflow_service: MLflowService = Depends(get_mlflow_client),
    db: Session = Depends(get_db),
) -> PerformanceService:
    """Dependency for PerformanceService. Injects db for history row writes."""
    return PerformanceService(
        mlflow_service=mlflow_service,
        ml_config_service=mlflow_service.ml_config_service,
        data_storage_client=DataStorageClient(),
        db=db,
    )


@router.post("/{field_name}/evaluate", response_model=FieldEvaluationResponse)
async def evaluate_field(
    field_name: str,
    metric: EvalMetricType = EvalMetricType.RMSE,
    performance_service: PerformanceService = Depends(get_performance_service),
) -> FieldEvaluationResponse:
    """
    Score all trained models that predict field_name using the chosen metric.

    All models competing on the same field are scored with the same metric,
    keeping comparison fair. The metric is persisted so subsequent /monitor
    calls automatically use the same one. Defaults to RMSE.
    """
    set_field_state(field_name, "evaluating")
    try:
        result = await performance_service.evaluate_field(field_name, metric.value)
        set_last_checked(field_name, datetime.now(tz=timezone.utc))
        return result
    except Exception as e:
        logger.error(f"Evaluation failed for field '{field_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        set_field_state(field_name, "monitoring")


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

@router.post("/{field_name}/set-best/{model_id}", response_model=ModelPerformance)
async def set_best_model(
    field_name: str,
    model_id: str,
    performance_service: PerformanceService = Depends(get_performance_service),
) -> ModelPerformance:
    """Override the best model for a field without running a new evaluation."""
    try:
        return performance_service.set_best_model(field_name, model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to set best model for field '{field_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{field_name}/monitor", response_model=ModelPerformance)
async def monitor_best_model(
    field_name: str,
    performance_service: PerformanceService = Depends(get_performance_service),
) -> ModelPerformance:
    """
    Re-evaluate only the current best model for field_name.

    Uses the same metric as the original election (stored as eval_metric:{field}
    MLflow tag). Updates score_for and eval_at tags but does NOT change the
    best_for designation — only a full /evaluate call can change that.

    Returns 422 if no best model has been designated yet.
    """
    try:
        return await performance_service.monitor_best_model(field_name, trigger="monitor")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Monitor failed for field '{field_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{field_name}/status", response_model=MonitoringStatusResponse)
async def get_monitoring_status(field_name: str) -> MonitoringStatusResponse:
    """Return the current state machine state for a monitored field."""
    return MonitoringStatusResponse(
        field_name=field_name,
        state=get_field_state(field_name),
        active_job_ids=get_field_jobs(field_name),
        last_checked_at=get_last_checked(field_name),
        monitoring_enabled=settings.MONITORING_ENABLED,
        monitoring_interval_seconds=settings.MONITORING_INTERVAL_SECONDS,
        monitoring_degradation_factor=settings.MONITORING_DEGRADATION_FACTOR,
    )


@router.get("/{field_name}/history", response_model=ScoreHistoryResponse)
async def get_score_history(
    field_name: str,
    model_id: str | None = None,
    performance_service: PerformanceService = Depends(get_performance_service),
) -> ScoreHistoryResponse:
    """
    Return the full score measurement history for field_name, oldest first.

    Includes every evaluate, monitor, and auto_monitor measurement written
    to the score_history table. Optionally filter by model_id.
    """
    try:
        return performance_service.get_score_history(field_name, model_id)
    except Exception as e:
        logger.error(f"Failed to get score history for field '{field_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{field_name}/importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(
    field_name: str,
    model_id: str | None = None,
    performance_service: PerformanceService = Depends(get_performance_service),
) -> FeatureImportanceResponse:
    """
    Return cached permutation importance for field_name.

    If model_id is omitted, returns importance for the elected best model.
    Returns 404 if no importance has been computed yet.
    """
    result = performance_service.get_feature_importance(field_name, model_id)
    if result is None:
        detail = (
            f"No importance data for model '{model_id}' on field '{field_name}'."
            if model_id
            else f"No importance data for field '{field_name}'. Run POST /evaluate or POST /{field_name}/models/{{model_id}}/importance first."
        )
        raise HTTPException(status_code=404, detail=detail)
    return result


@router.post("/{field_name}/models/{model_id}/importance", response_model=FeatureImportanceResponse)
async def trigger_model_importance(
    field_name: str,
    model_id: str,
    performance_service: PerformanceService = Depends(get_performance_service),
) -> FeatureImportanceResponse:
    """
    Compute and store permutation importance for a specific model on demand.

    Loads the model, fetches data, runs alibi PermutationImportance, stores results
    as MLflow tags and returns the result. Use GET /importance to read cached results.
    """
    cells = await performance_service.data_storage_client.get_known_cells()
    try:
        return await performance_service.trigger_feature_importance(field_name, model_id, cells)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Importance computation failed for model '{model_id}' on field '{field_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{field_name}", response_model=FieldEvaluationResponse)
async def get_evaluation(
    field_name: str,
    performance_service: PerformanceService = Depends(get_performance_service),
) -> FieldEvaluationResponse:
    """
    Return the last cached evaluation result for field_name.

    Reads persisted MLflow tags — no model loading or data fetching occurs.
    Models that have never been evaluated will appear with score=None.
    """
    try:
        return performance_service.get_cached_evaluation(field_name)
    except Exception as e:
        logger.error(f"Failed to get cached evaluation for field '{field_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))
