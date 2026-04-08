from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from src.services.mlflow_service import MLflowService
from src.services.data_storage_client import DataStorageClient
from src.core.dependencies import get_mlflow_client
from src.schemas.model import ModelConfig, ModelCreate, ModelDetail, ModelSummary, ModelSummaryDetail
router = APIRouter()


def _get_policy_client(request: Request) -> Optional[Any]:
    """Get the policy client from app state if available."""
    return getattr(request.app.state, "policy_client", None)

@router.get("", response_model=list[ModelSummary] | list[ModelSummaryDetail])
async def get_models(
    output_field: str | None = None,
    include_details: bool = False,
    mlflow_service: MLflowService = Depends(get_mlflow_client),
) -> list[ModelSummary] | list[ModelSummaryDetail]:
    """Get all registered models, optionally filtered by output field name.

    Pass include_details=true to receive enriched metadata (last_trained_at,
    best_for_fields, score_per_field, is_training, etc.) at no extra cost —
    uses a single MLflow search call regardless of model count.
    """
    if include_details:
        models = mlflow_service.list_models_detailed()
    else:
        models = mlflow_service.list_models()

    if output_field:
        valid_ids = {
            c.model_id
            for c in mlflow_service.ml_config_service.list_all()
            if output_field in c.output_fields
        }
        models = [m for m in models if m.id in valid_ids]

    return models


@router.post("", response_model=ModelDetail, status_code=201)
async def create_model(
    model_create: ModelCreate,
    request: Request,
    mlflow_service: MLflowService = Depends(get_mlflow_client),
    data_storage_client: DataStorageClient = Depends(DataStorageClient),
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

        # Register with policy service if enabled
        policy_client = _get_policy_client(request)
        if policy_client:
            try:
                data_type = mlflow_service.infer_data_type(model_create.config.output_fields)
                await policy_client.register_ml_model(
                    model_id=model.id,
                    model_name=model.name,
                    input_fields=model_create.config.input_fields,
                    output_fields=model_create.config.output_fields,
                    data_type=data_type,
                    architecture=model_create.config.architecture,
                    window_duration_seconds=model_create.config.window_duration_seconds,
                )
            except Exception as e:
                # Non-blocking: log but don't fail model creation
                import logging
                logging.getLogger(__name__).warning(f"Failed to register model with policy: {e}")

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
async def delete_model(
    model_id: str,
    request: Request,
    mlflow_service: MLflowService = Depends(get_mlflow_client),
) -> None:
    """Delete model by ID"""
    try:
        # Get model name before deletion for policy unregistration
        policy_client = _get_policy_client(request)
        model_name = None
        if policy_client:
            try:
                model = mlflow_service.get_model(model_id)
                model_name = model.name
            except Exception:
                pass  # Model might not exist, continue with deletion

        mlflow_service.delete_model(model_id)

        # Unregister from policy service if enabled
        if policy_client and model_name:
            try:
                await policy_client.unregister_ml_model(model_name)
            except Exception as e:
                # Non-blocking: log but don't fail model deletion
                import logging
                logging.getLogger(__name__).warning(f"Failed to unregister model from policy: {e}")

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
