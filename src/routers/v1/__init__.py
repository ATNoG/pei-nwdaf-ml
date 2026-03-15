from .model_router import router as model_router
from .fields_router import router as fields_router
from .training_router import router as training_router
from .inference_router import router as inference_router
from .performance_router import router as performance_router
<<<<<<< HEAD
from .expiry_router import router as expiry_router
from .anomaly_router import router as anomaly_router
from fastapi import APIRouter

router = APIRouter()
router.include_router(model_router, prefix="/models")
router.include_router(fields_router, prefix="/fields")
router.include_router(training_router, prefix="/training")
router.include_router(inference_router, prefix="/inference", tags=["Inference"])
router.include_router(performance_router, prefix="/performance", tags=["Performance"])
router.include_router(expiry_router, prefix="/expiry", tags=["Expiry"])
router.include_router(anomaly_router, prefix="/anomaly", tags=["Anomaly Detection"])
