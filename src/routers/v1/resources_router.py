"""Router for resource usage and model resource defaults."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.core.dependencies import get_db
from src.schemas.resources import (
    ModelResourceDefaults,
    ResourcesUsageResponse,
    ResourceUsageEntry,
)
from src.services.resources_service import (
    get_model_defaults,
    list_active_job_usage,
    set_model_defaults,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/usage", response_model=ResourcesUsageResponse)
async def get_resources_usage(db: Session = Depends(get_db)) -> ResourcesUsageResponse:
    """Return resource allocations and live usage for all running or queued training jobs."""
    entries: list[ResourceUsageEntry] = list_active_job_usage(db)
    return ResourcesUsageResponse(jobs=entries)


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
