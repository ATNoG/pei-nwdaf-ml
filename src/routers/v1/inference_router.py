"""Router for inference endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from src.services.inference_service import InferenceService
from src.services.mlflow_service import MLflowService
from src.services.data_storage_client import DataStorageClient
from src.core.dependencies import get_mlflow_client
from src.schemas.inference import InferenceRequest, InferenceResult

logger = logging.getLogger(__name__)
router = APIRouter()


def get_inference_service(
    mlflow_service: MLflowService = Depends(get_mlflow_client),
) -> InferenceService:
    """Dependency for InferenceService."""
    data_storage_client = DataStorageClient()
    return InferenceService(
        mlflow_service=mlflow_service,
        data_storage_client=data_storage_client,
    )


@router.post("", response_model=InferenceResult)
async def run_inference(
    request: InferenceRequest,
    inference_service: InferenceService = Depends(get_inference_service),
) -> InferenceResult:
    """
    Run inference on a trained model for a specific cell.

    Fetches the most recent data for the given cell, runs the trained
    model, and returns structured predictions per forecast step.
    """
    try:
        result = await inference_service.predict(
            model_id=request.model_id,
            cell_id=request.cell_id,
        )
        return InferenceResult(**result)
    except ValueError as e:
        logger.warning(f"Inference validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Inference prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
