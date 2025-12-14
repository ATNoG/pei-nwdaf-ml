"""
Analytics predictions endpoint for NWDAF
"""
from fastapi import APIRouter, HTTPException, Request
import logging

from src.services.inference import InferenceService
from src.schemas.inference import (
    AnalyticsRequest,
    PredictionHorizon
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("", response_model=PredictionHorizon)
async def get_cell_analytics(
    analytics_request: AnalyticsRequest,
    request: Request
):
    """
    Get analytics prediction for a cell.

    Returns prediction for the specified analytics type and time horizon.

    Args:
        analytics_request: Analytics request with cell_id, horizon, and optional model_type

    Returns:
        PredictionHorizon: Prediction result with interval, value, and confidence

    Raises:
        400: Invalid parameters or insufficient data
        404: Analytics type not registered or model not found
        500: ML Interface not initialized
    """
    ml_interface = request.app.state.ml_interface
    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    service = InferenceService(ml_interface)

    try:
        return await service.predict_cell_analytics(
            analytics_type=analytics_request.analytics_type,
            cell_index=analytics_request.cell_index,
            horizon=analytics_request.horizon,
            model_type=analytics_request.model_type
        )

    except ValueError as e:
        # Configuration errors, invalid parameters, data issues
        error_msg = str(e)
        if "No config found" in error_msg or "not found" in error_msg:
            raise HTTPException(status_code=404, detail=error_msg)
        else:
            raise HTTPException(status_code=400, detail=error_msg)

    except RuntimeError as e:
        # Model loading errors
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
