from .model_router import router as model_router
from .fields_router import router as fields_router
from .training_router import router as training_router
from fastapi import APIRouter

router = APIRouter()
router.include_router(model_router, prefix="/models")
router.include_router(fields_router, prefix="/fields")
router.include_router(training_router, prefix="/training")
