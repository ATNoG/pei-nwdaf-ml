from fastapi import APIRouter
from src.routers.v1 import cell_inference

v1_router = APIRouter()

v1_router.include_router(
    cell_inference.router,
    prefix="/analytics",
    tags=["v1", "analytics"]
)
