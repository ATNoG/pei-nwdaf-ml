from fastapi import APIRouter
from src.routers.v1 import cell_inference, training, config

v1_router = APIRouter()

v1_router.include_router(
    cell_inference.router,
    prefix="/analytics",
    tags=["v1", "analytics"]
)

v1_router.include_router(
    training.router,
    prefix="/training",
    tags=["v1", "training"]
)
v1_router.include_router(
    config.router,
    prefix="/config",
    tags=["v1", "config"]
)
