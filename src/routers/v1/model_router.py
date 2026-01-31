from fastapi import APIRouter, Depends
from src.services.mlflow_service import MLflowService
from src.core.dependencies import get_mlflow_client
from src.schemas.model import (ModelConfig,ModelCreate,ModelDetail,ModelSummary)
router = APIRouter()

@router.get("")
async def get_models(mlflow_service: MLflowService = Depends(get_mlflow_client)) -> list[ModelSummary]:
    """Get all instanced models"""
    models = mlflow_service.list_models()
    return models


@router.post("")
async def create_model(model_create: ModelCreate, mlflow_service: MLflowService = Depends(get_mlflow_client)) -> ModelDetail:
    """Get all instanced models"""
    model = mlflow_service.create_model(model_create.name, model_create.config)
    return model
