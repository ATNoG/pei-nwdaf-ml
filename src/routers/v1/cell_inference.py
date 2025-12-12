"""
Analytics predictions endpoint for NWDAF
"""
from fastapi import APIRouter, HTTPException, Request
import logging
import numpy as np

from src.inference.inference import InferenceMaker
from src.schemas.inference import (
    AnalyticsRequest,
    PredictionHorizon as PredictionHorizonModel
)
from src.config.inference_type import get_inference_config
from src.models import models_dict

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("", response_model=PredictionHorizonModel)
async def get_cell_analytics(
    analytics_request: AnalyticsRequest,
    request: Request
):
    """
    Get all analytics predictions for a cell.

    Returns predictions for all registered analytics types (latency, etc.)
    for the specified time horizon using the specified model.

    Args:
        analytics_request: Analytics request with cell_id, horizon, and model_type

    Returns:
        All analytics predictions for the cell

    Raises:
        404: No data found for cell
        404: Analytics type not registered
        500: ML Interface not initialized

    """
    ml_interface = request.app.state.ml_interface
    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    cell_id = analytics_request.cell_id
    analytics_type = analytics_request.analytics_type
    horizon = analytics_request.horizon
    model_type = analytics_request.model_type

    logger.info(f"Analytics request for cell {cell_id}, horizon={horizon}s, model_type={model_type}")

    # Convert horizon to ISO 8601 format
    if horizon < 60:
        interval_str = f"PT{horizon}S"
    elif horizon < 3600:
        minutes = horizon // 60
        interval_str = f"PT{minutes}M"
    elif horizon < 86400:
        hours = horizon // 3600
        interval_str = f"PT{hours}H"
    elif horizon < 604800:
        days = horizon // 86400
        interval_str = f"P{days}D"
    else:
        weeks = horizon // 604800
        interval_str = f"P{weeks}W"

    key = (analytics_type, horizon)
    config = get_inference_config(key)
    if config is None:
        raise HTTPException(status_code=404, detail="Analytics type not registered or analytics not allowed for given horizon")

    model_class = models_dict.get(model_type.lower())
    if model_class is None:
        raise HTTPException(status_code=404, detail=f"Model type not found: {model_type}")

    # Generate prediction for the requested horizon
    try:
        # Fetch data based on model's sequence length requirement
        windows_data = await ml_interface.fetch_latest_cell_data(
            endpoint=config.storage_endpoint,
            cell_id=cell_id,
            window_duration_seconds=config.window_duration_seconds,
            num_windows=model_class.SEQUENCE_LENGTH
        )

        if windows_data is None:
            raise HTTPException(status_code=404, detail=f"No data for {analytics_type}, cell {cell_id}")

        # Ensure we have a list of windows
        if isinstance(windows_data, dict):
            windows_data = [windows_data]

        if len(windows_data) < model_class.SEQUENCE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough data to use model [{model_type}]. Need {model_class.SEQUENCE_LENGTH} windows, got {len(windows_data)}"
            )

        # Load the trained model from MLflow
        inference_maker = InferenceMaker(ml_interface)
        model = inference_maker._load_inference_type_model(analytics_type, horizon, model_type)

        if model is None or not hasattr(model, "predict"):
            raise HTTPException(status_code=404, detail=f"No model for {analytics_type} (horizon={horizon}s) with type {model_type}")

        # Extract features from windows
        def extract_features(window_dict):
            return {
                k: v for k, v in window_dict.items()
                if k not in {
                    "window_start_time", "window_end_time",
                    "window_duration_seconds", "cell_index",
                    "network", "sample_count"
                }
                and not k.startswith(analytics_type + '_')  # Exclude target columns (latency_*, throughput_*, etc.)
                and v is not None
            }

        # Handle both single window and sequence cases
        if model_class.SEQUENCE_LENGTH == 1:
            # Single window model: extract features from first (and only) window
            inference_data = extract_features(windows_data[0])
            prepared_data = inference_maker._prepare_data_for_prediction(inference_data)
            last_window = windows_data[0]
        else:
            # Sequence model: extract features from all windows and stack
            features_list = [extract_features(w) for w in windows_data]

            if not features_list or not features_list[0]:
                raise HTTPException(status_code=404, detail=f"No valid features for {analytics_type}, cell {cell_id}")

            # Prepare each window and stack them
            prepared_list = [inference_maker._prepare_data_for_prediction(f) for f in features_list]
            prepared_data = np.array(prepared_list)

            inference_data = features_list[0]
            last_window = windows_data[-1]

        if not inference_data:
            raise HTTPException(status_code=404, detail=f"No valid features for {analytics_type}, cell {cell_id}")

        # Ensure data is float32 for PyTorch models
        if isinstance(prepared_data, np.ndarray):
            prepared_data = prepared_data.astype(np.float32)
        elif hasattr(prepared_data, 'astype'):  # pandas DataFrame
            prepared_data = prepared_data.astype(np.float32)

        # Get prediction
        result = model.predict(prepared_data)
        result = inference_maker._convert_result_for_serialization(result)

        # Extract scalar value from nested lists/arrays
        while isinstance(result, (list, np.ndarray)) and len(result) > 0:
            result = result[0]

        # TODO: compute confidence
        # maybe based on number of samples and stability
        base_confidence = 0.85
        horizon_factor = min(1.0, 60 / horizon)
        confidence = base_confidence * horizon_factor

        # Calculate prediction window from last data window
        last_window_end = last_window.get("window_end_time", 0)
        target_start = last_window_end
        target_end = last_window_end + horizon

        return PredictionHorizonModel(
            used_model=model_type,
            cell_index=cell_id,
            interval=interval_str,
            predicted_value=float(result),
            confidence=round(confidence, 2),
            data=inference_data,
            target_start_time=target_start,
            target_end_time=target_end,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting for {analytics_type}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Failed to predict")
