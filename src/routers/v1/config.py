"""Configuration endpoint for API metadata"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.services.config import ConfigService
from src.schemas.config import ConfigResponse

router = APIRouter()


class UpdateDefaultModelRequest(BaseModel):
    analytics_type: str
    horizon: int
    model_type: str


@router.get("", response_model=ConfigResponse)
async def get_config():
    """
    Get all available configurations.

    Returns supported inference types, their horizons, and available model types.

    Returns:
        ConfigResponse: Available inference types and supported models
    """
    return ConfigService.get_system_config()


@router.patch("")
async def update_default_model(request: UpdateDefaultModelRequest):
    """
    Update the default model for a specific inference type configuration.

    Args:
        request: Analytics type, horizon, and new model type

    Returns:
        Success message with updated configuration

    Raises:
        404: Config not found
        400: Invalid model type

    Example:
        PATCH /api/v1/config
        {
            "analytics_type": "latency",
            "horizon": 60,
            "model_type": "xgboost"
        }
    """
    try:
        result = ConfigService.update_default_model(
            analytics_type=request.analytics_type,
            horizon=request.horizon,
            model_type=request.model_type
        )
        return result
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg.lower():
            raise HTTPException(status_code=404, detail=error_msg)
        else:
            raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
