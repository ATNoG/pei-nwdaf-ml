from fastapi import APIRouter, Depends, HTTPException
from src.services.mlflow_service import MLflowService
from src.services.data_storage_client import DataStorageClient
from src.core.dependencies import get_mlflow_client
from src.schemas.model import (ModelConfig,ModelCreate,ModelDetail,ModelSummary)
router = APIRouter()

@router.get("", response_model=list[ModelSummary])
async def get_models(mlflow_service: MLflowService = Depends(get_mlflow_client)) -> list[ModelSummary]:
    """Get all registered models"""
    models = mlflow_service.list_models()
    return models


@router.post("", response_model=ModelDetail, status_code=201)
async def create_model(
    model_create: ModelCreate,
    mlflow_service: MLflowService = Depends(get_mlflow_client),
    data_storage_client: DataStorageClient = Depends(DataStorageClient)
) -> ModelDetail:
    """Create a new model with field validation"""
    try:
        # Validate input fields
        all_fields = model_create.config.input_fields + model_create.config.output_fields
        is_valid, invalid_fields = await data_storage_client.validate_fields(all_fields)

        if not is_valid:
            available_fields = await data_storage_client.get_available_fields()
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid field names provided",
                    "invalid_fields": sorted(invalid_fields),
                    "available_fields": sorted(available_fields)
                }
            )

        # Create model
        model = mlflow_service.create_model(model_create.name, model_create.config)
        return model
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{model_id}", response_model=ModelDetail)
async def get_model(model_id: str, mlflow_service: MLflowService = Depends(get_mlflow_client)) -> ModelDetail:
    """Get model by ID"""
    try:
        model = mlflow_service.get_model(model_id)
        return model
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{model_id}", status_code=204)
async def delete_model(model_id: str, mlflow_service: MLflowService = Depends(get_mlflow_client)) -> None:
    """Delete model by ID"""
    try:
        mlflow_service.delete_model(model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
