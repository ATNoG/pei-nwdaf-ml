"""
Analytics predictions endpoint for NWDAF
"""
from fastapi import APIRouter, HTTPException, Request
from datetime import datetime
import logging

from src.inference.inference import InferenceMaker
from src.schemas.inference import (
    CellAnalyticsResponse,
    AnalyticsTypePrediction,
    PredictionHorizon as PredictionHorizonModel
)
from src.config.inference_type import get_all_inference_types, get_inference_config

logger = logging.getLogger(__name__)

router = APIRouter()

async def fetch_latest_cell_data(
    ml_interface,
    analytics_type: str,
    cell_id: int,
    seconds:int
):
    """
    Fetch latest data window for a cell.

    Args:
        ml_interface: MLInterface instance
        analytics_type: Analytics type (e.g., 'latency')
        cell_id: Cell identifier

    Returns:
        dict: Latest data window or None
    """
    config = get_inference_config(analytics_type)
    if not config:
        logger.error(f"Analytics type not found: {analytics_type}")
        return None

    try:
        # Query window
        # Note: Using wide range as storage may have test data with incorrect timestamps
        params = {
            "cell_index": cell_id,
            "start_time": 0,
            "end_time": 9999999999,
            "offset": 0,
            "limit": 1
        }

        logger.info(f"Fetching latest data for cell {cell_id}")

        data = await ml_interface.request_data_from_storage_async(
            endpoint=config.storage_endpoint,
            params=params,
            method="GET"
        )

        if not data or len(data) == 0:
            logger.warning(f"No data found for cell {cell_id}")
            return None

        window = data[0] if isinstance(data, list) else data
        logger.info(f"Found data for cell {cell_id}")
        return window

    except Exception as e:
        logger.error(f"Error fetching data for cell {cell_id}: {e}", exc_info=True)
        return None


@router.get("/{cell_id}", response_model=CellAnalyticsResponse)
async def get_cell_analytics(
    cell_id: int,
    request: Request
):
    """
    Get all analytics predictions for a cell.

    Returns predictions for all registered analytics types (latency, etc.)
    across all time horizons (PT1M, PT1H, P1D, P1W).

    Args:
        cell_id: Cell identifier

    Returns:
        All analytics predictions for the cell

    Raises:
        404: No data found for cell
        500: ML Interface not initialized

    Example:
        GET /api/v1/analytics/26379009

        Response:
        {
            "cell_id": 26379009,
            "timestamp": 1733828700.0,
            "analytics": [
                {
                    "analytics_type": "latency",
                    "predictions": [
                        {"interval": "PT1M", "predicted_value": 45.2, "confidence": 0.85},
                        {"interval": "PT1H", "predicted_value": 48.1, "confidence": 0.75}
                    ]
                }
            ]
        }
    """
    ml_interface = request.app.state.ml_interface
    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    logger.info(f"Analytics request for cell {cell_id}")

    # Time horizons to predict (in seconds)
    horizons = {
        "PT1M": 60,
        #"PT1H": 3600,
        #"P1D": 86400,
        #"P1W": 604800
    }

    analytics_results = []
    inference_types = get_all_inference_types()

    for analytics_type, config in inference_types.items():

        inference_maker = InferenceMaker(ml_interface)
        model = inference_maker._load_inference_type_model(analytics_type, "xgboost")

        if model is None or not hasattr(model, "predict"):
            logger.warning(f"No model for {analytics_type}")
            continue

        # Generate predictions for all horizons
        predictions = []
        for interval_str, horizon_seconds in horizons.items():
            try:

                window_data = await fetch_latest_cell_data(
                    ml_interface,
                    analytics_type,
                    cell_id,
                    horizon_seconds
                )

                if window_data is None:
                    logger.warning(f"No data for {analytics_type}, cell {cell_id}")
                    continue

                # Extract features
                inference_data = {
                    k: v for k, v in window_data.items()
                    if k not in {
                        "window_start_time", "window_end_time",
                        "window_duration_seconds", "cell_index",
                        "network", "sample_count"
                    }
                    and v is not None
                }

                if not inference_data:
                    continue

                # Prepare data
                prepared_data = inference_maker._prepare_data_for_prediction(inference_data)

                result = model.predict(prepared_data)
                result = inference_maker._convert_result_for_serialization(result)

                if isinstance(result, list) and result:
                    result = result[0]

                # Confidence degrades with longer horizon
                base_confidence = 0.85
                horizon_factor = min(1.0, 60 / horizon_seconds)
                confidence = base_confidence * horizon_factor

                predictions.append(PredictionHorizonModel(
                    interval=interval_str,
                    predicted_value=float(result),
                    confidence=round(confidence, 2),
                    data=inference_data
                ))
            except Exception as e:
                logger.error(f"Error predicting {interval_str} for {analytics_type}: {e}")

        if predictions:
            analytics_results.append(AnalyticsTypePrediction(
                analytics_type=analytics_type,
                predictions=predictions
            ))

    if not analytics_results:
        raise HTTPException(status_code=404, detail="no data")

    return CellAnalyticsResponse(
        cell_id=cell_id,
        timestamp=datetime.now().timestamp(),
        analytics=analytics_results
    )
