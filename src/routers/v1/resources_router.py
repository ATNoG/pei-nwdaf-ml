"""Router for resource usage and model resource defaults."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from src.core.dependencies import get_db
from src.schemas.resources import (
    ModelResourceDefaults,
    ResourcesUsageResponse,
)
from src.services.resources_service import (
    generate_prometheus_metrics,
    get_model_defaults,
    list_all_model_usage,
    set_model_defaults,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/usage", response_model=ResourcesUsageResponse)
async def get_resources_usage(db: Session = Depends(get_db)) -> ResourcesUsageResponse:
    """Return resource usage for all models; idle models have null resource fields."""
    return ResourcesUsageResponse(jobs=list_all_model_usage(db))


@router.get("/metrics", response_class=Response)
async def get_resources_metrics(db: Session = Depends(get_db)) -> Response:
    """Return Prometheus text-format metrics for all models (0 when idle)."""
    return Response(content=generate_prometheus_metrics(db), media_type="text/plain; version=0.0.4; charset=utf-8")


@router.get("/metrics/{model_id}", response_class=Response)
async def get_model_resources_metrics(model_id: str, db: Session = Depends(get_db)) -> Response:
    """Return Prometheus text-format metrics for a specific model (0 when idle)."""
    return Response(content=generate_prometheus_metrics(db, model_id=model_id), media_type="text/plain; version=0.0.4; charset=utf-8")


@router.put("/defaults/{model_id}", response_model=ModelResourceDefaults)
async def set_model_resource_defaults(
    model_id: str,
    body: ModelResourceDefaults,
    db: Session = Depends(get_db),
) -> ModelResourceDefaults:
    """Set default resource requests for a model's training jobs (forecast or anomaly)."""
    try:
        return set_model_defaults(model_id, body.cpu, body.memory, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/defaults/{model_id}", response_model=ModelResourceDefaults | None)
async def get_model_resource_defaults(
    model_id: str,
    db: Session = Depends(get_db),
) -> ModelResourceDefaults | None:
    """Get default resource requests for a model (forecast or anomaly). Returns null if not set."""
    try:
        return get_model_defaults(model_id, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
