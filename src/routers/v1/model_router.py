"""
Model Instance Creation API Endpoint
"""
from fastapi import APIRouter, HTTPException, Request
import logging

from src.services.model_service import ModelService
from src.schemas.model import (
    ModelCreationRequest,
    ModelCreationResponse,
    ModelDeletionResponse,
    ModelInfo,
    ModelDetailedInfo
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/instance", response_model=ModelCreationResponse)
async def create_model_instance(
    model_request: ModelCreationRequest,
    request: Request
):
    """
    Create a new model instance for an analytics type.

    Args:
        model_request: Analytics type, horizon, model type, and optional config

    Returns:
        Confirmation that model instance was created

    Raises:
        400: Invalid parameters or model already exists
        404: Analytics type/horizon not configured or model type not found
        500: ML Interface not initialized or error creating model

    Example:
        POST /api/v1/model/instance
        {
            "analytics_type": "latency",
            "horizon": 60,
            "model_type": "ann",
            "config": {
                "architecture": {"hidden_size": 64},
                "sequence": {"sequence_length": 5}
            }
        }
    """
    ml_interface = request.app.state.ml_interface
    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    service = ModelService(ml_interface)

    try:
        # Convert Pydantic config to dataclass if provided
        model_config = None
        if model_request.config:
            model_config = model_request.config.to_model_config()

        model_name = model_request.name

        logger.info(
            f"Creating model instance for analytics_type={model_request.analytics_type}, "
            f"horizon={model_request.horizon}s, model_type={model_request.model_type}, "
            f"name={model_request.name or '(auto)'}"
        )

        # Create model and get the actual model name used
        service.create_model_instance(
            horizon=model_request.horizon,
            analytics_type=model_request.analytics_type,
            model_type=model_request.model_type,
            model_config=model_config,
            name=model_name,
        )

        logger.info(f"Successfully created model instance: {model_name}")

        return ModelCreationResponse(
            status="created",
            model_name=model_name,
            message=f"Model instance {model_name} created successfully"
        )

    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg.lower() or "not accepted" in error_msg.lower():
            raise HTTPException(status_code=404, detail=error_msg)
        elif "already exists" in error_msg.lower():
            raise HTTPException(status_code=400, detail=error_msg)
        else:
            raise HTTPException(status_code=400, detail=error_msg)

    except Exception as e:
        logger.error(f"Error creating model instance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/models", response_model=list[ModelInfo])
async def list_models(request: Request):
    """List all available ML models from MLFlow registry"""
    ml_interface = request.app.state.ml_interface

    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    if not ml_interface.is_mlflow_connected():
        raise HTTPException(status_code=503, detail="MLFlow not connected")

    try:
        models = ml_interface.list_registered_models()
        return [ModelInfo(**model) for model in models]
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/instance/{model_name}", response_model=ModelDetailedInfo)
async def get_model_details(
    model_name: str,
    request: Request
):
    """
    Get detailed information about a model.

    Args:
        model_name: Name of the model

    Returns:
        Detailed model information including architecture and config

    Raises:
        404: Model not found
        500: ML Interface not initialized or error

    Example:
        GET /api/v1/model/instance/latency_ann_60
    """
    ml_interface = request.app.state.ml_interface
    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    service = ModelService(ml_interface)

    try:
        details = service.get_model_details(model_name)
        return ModelDetailedInfo(**details)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting model details: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/instance/{model_name}", response_model=ModelDeletionResponse)
async def delete_model_instance(
    model_name: str,
    request: Request
):
    """
    Delete a model instance from MLflow registry.

    Args:
        model_name: Name of the model to delete

    Returns:
        Confirmation that model instance was deleted

    Raises:
        404: Model instance not found
        500: ML Interface not initialized or error deleting model

    Example:
        DELETE /api/v1/model/instance/latency_ann_60
    """
    ml_interface = request.app.state.ml_interface
    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    service = ModelService(ml_interface)

    try:
        logger.info(f"Deleting model instance: {model_name}")

        service.delete_model_instance(model_name=model_name)

        logger.info(f"Successfully deleted model instance: {model_name}")

        return ModelDeletionResponse(
            status="deleted",
            model_name=model_name,
            message=f"Model instance {model_name} deleted successfully"
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"Error deleting model instance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
