"""Router for inference endpoints."""

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from src.services.inference_service import InferenceService
from src.services.mlflow_service import MLflowService
from src.services.data_storage_client import DataStorageClient
from src.core.dependencies import get_mlflow_client
from src.schemas.inference import InferenceRequest, InferenceResult

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_policy_client(request: Request) -> Optional[Any]:
    """Get the policy client from app state if available."""
    return getattr(request.app.state, "policy_client", None)


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
    http_request: Request,
    inference_service: InferenceService = Depends(get_inference_service),
) -> InferenceResult:
    """
    Run inference on a trained model for a specific cell.

    Fetches the most recent data for the given cell, runs the trained
    model, and returns structured predictions per forecast step.
    """
    try:
        result = await inference_service.predict(
            output_field=request.output_field,
            cell_id=request.cell_id,
            model_id=request.model_id,
            explain=request.explain,
        )

        # Apply policy filtering if enabled
        policy_client = _get_policy_client(http_request)
        if policy_client:
            try:
                # Get caller's component ID from request header (default to "anonymous")
                caller_id = http_request.headers.get("X-Component-ID", "anonymous")
                model_name = result.get("model_name", "unknown")

                # Filter each forecast step's values based on policy
                for step in result.get("forecast", []):
                    if "values" in step:
                        process_result = await policy_client.process_data(
                            source_id=f"ml-{model_name}",
                            sink_id=caller_id,
                            data=step["values"],
                            action="read",
                        )

                        if process_result.allowed:
                            # Use filtered data
                            step["values"] = process_result.data or step["values"]
                            # Add metadata about transformations
                            if process_result.transformations:
                                step["policy_transformations"] = process_result.transformations
                        else:
                            # Authorization denied - redact all values
                            step["values"] = {k: "***REDACTED***" for k in step["values"]}
                            step["policy_denied"] = True
                            step["policy_reason"] = process_result.reason
                            # Also redact metric if present
                            if "metric" in step:
                                metric_name = step.get("metric", {}).get("name", "unknown")
                                step["metric"] = {"value": "***REDACTED***", "name": metric_name}

            except Exception as e:
                # Non-blocking: log error but return unfiltered result on policy failure
                logger.warning(f"Policy filtering failed, returning unfiltered result: {e}")

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
