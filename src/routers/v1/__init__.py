from .model_router import router as model_router
from .fields_router import router as fields_router
from fastapi import APIRouter

router = APIRouter()
router.include_router(model_router, prefix="/models")
router.include_router(fields_router, prefix="/fields")
