"""
Cell-specific inference endpoint that auto-fetches latest data from storage
"""
from fastapi import APIRouter, HTTPException, Request
from datetime import datetime, timedelta
import logging

from src.inference.inference import InferenceMaker
from src.schemas.inference import CellInferenceRequest, CellInferenceResponse

logger = logging.getLogger(__name__)

router = APIRouter()


async def fetch_latest_cell_data(ml_interface, cell_index: int):
    """
    Fetch the latest data window for a cell from data storage.

    Args:
        ml_interface: MLInterface instance
        cell_index: Cell index to query

    Returns:
        dict: Latest data window or None if not found
    """
    try:
        # Query last 24 hours of data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)

        params = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "cell_index": cell_index,
            "offset": 0,
            "limit": 100  # Get last 100 windows
        }

        logger.info(f"Fetching data for cell {cell_index} from storage")

        data = await ml_interface.request_data_from_storage_async(
            endpoint="/api/v1/processed/latency/",
            params=params,
            method="GET"
        )

        if not data or len(data) == 0:
            logger.warning(f"No data found for cell {cell_index}")
            return None

        # Data is a list of windows, get the most recent one
        # Sort by window_end_time to get the latest
        if isinstance(data, list):
            sorted_data = sorted(
                data,
                key=lambda x: x.get('window_end_time', ''),
                reverse=True
            )
            latest_window = sorted_data[0]
            logger.info(f"Found latest data window for cell {cell_index}: {latest_window.get('window_start_time')} to {latest_window.get('window_end_time')}")
            return latest_window
        else:
            return data

    except Exception as e:
        logger.error(f"Error fetching data for cell {cell_index}: {e}")
        return None


@router.post("/cell", response_model=CellInferenceResponse)
async def cell_inference_with_storage(
    req: CellInferenceRequest,
    request: Request
):
    """
    Perform inference for a specific cell by automatically fetching the latest data from storage.

    This endpoint:
    1. Fetches the latest data window for the specified cell from data storage
    2. Loads the cell-specific model
    3. Performs inference on the latest data
    4. Returns predictions or "no data or cell found" error

    Args:
        req: Request containing cell_index and optional model_type

    Returns:
        CellInferenceResponse with predictions and metadata

    Raises:
        HTTPException: If no data found, model not available, or inference fails

    Example:
        POST /api/v1/inference/cell
        {
            "cell_index": 12898855,
            "model_type": "xgboost"
        }
    """
    ml_interface = request.app.state.ml_interface

    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    try:
        logger.info(f"Cell inference request for cell {req.cell_index} with model type {req.model_type}")

        # Fetch latest data from storage
        latest_data = await fetch_latest_cell_data(ml_interface, req.cell_index)

        if latest_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"No data or cell found for cell_index {req.cell_index}"
            )

        # Extract feature data for inference (exclude metadata fields)
        inference_data = {
            k: v for k, v in latest_data.items()
            if k not in ['window_start_time', 'window_end_time', 'window_duration_seconds',
                        'cell_index', 'network', 'sample_count']
            and v is not None
        }

        if not inference_data:
            raise HTTPException(
                status_code=400,
                detail=f"No valid feature data found in latest window for cell {req.cell_index}"
            )

        logger.info(f"Using {len(inference_data)} features for inference: {list(inference_data.keys())}")

        # Get inference maker and perform inference
        inference_maker = InferenceMaker(ml_interface)

        result = inference_maker.infer(
            cell_index=str(req.cell_index),
            data=inference_data,
            model_type=req.model_type
        )

        if result is None:
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed for cell {req.cell_index}. Model may not exist or data is invalid."
            )

        model_used = inference_maker._get_cell_model_name(req.cell_index, req.model_type)

        return CellInferenceResponse(
            cell_index=str(req.cell_index),
            predictions=result,
            model_used=model_used,
            timestamp=datetime.now().timestamp(),
            data_window_start=latest_data.get('window_start_time'),
            data_window_end=latest_data.get('window_end_time')
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during cell inference for {req.cell_index}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference error for cell {req.cell_index}: {str(e)}"
        )
