from fastapi import APIRouter
from src.routers.v1 import inference_router, training_router, config_router, model_router, websocket

v1_router = APIRouter()

v1_router.include_router(
    inference_router.router,
    prefix="/analytics",
    tags=["v1", "analytics"]
)

v1_router.include_router(
    training_router.router,
    prefix="/training",
    tags=["v1", "training"]
)
v1_router.include_router(
    config_router.router,
    prefix="/config",
    tags=["v1", "config"]
)
v1_router.include_router(
    model_router.router,
    prefix="/model",
    tags=["v1", "model"]
)

v1_router.include_router(
    websocket.router,
    prefix="/websocket",
    tags=["v1", "websocket"]
)
