from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import json

from src.inference.inference import InferenceMaker

logger = logging.getLogger(__name__)

router = APIRouter()

_inference_maker: Optional[InferenceMaker] = None

class InferenceRequest(BaseModel):
    """Request model for ML inference"""
    data: Dict[str, Any]
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_stage: Optional[str] = "Production"
    publish_result: bool = False


class TrainingRequest(BaseModel):
    """Request model for ML training"""
    model_name: str
    config: Dict[str, Any]


class InferenceResponse(BaseModel):
    """Response model for inference results"""
    status: str
    model_used: Optional[str]
    predictions: Any
    published_to_kafka: bool = False


class ModelInfo(BaseModel):
    """Model information from MLFlow registry"""
    name: str
    creation_timestamp: Optional[int] = None
    last_updated_timestamp: Optional[int] = None
    description: Optional[str] = None
    latest_versions: Optional[List[Dict[str, Any]]] = None


class ModelSelectionRequest(BaseModel):
    """Request to manually select a model for inference"""
    model_name: str
    version: Optional[str] = None
    stage: Optional[str] = "Production"


class AutoModeRequest(BaseModel):
    """Request to toggle auto-select mode"""
    auto_mode: bool


def get_inference_maker(ml_interface) -> InferenceMaker:
    """Get or create the inference maker instance"""
    global _inference_maker
    if _inference_maker is None:
        _inference_maker = InferenceMaker(ml_interface)
        logger.info("Initialized InferenceMaker")
    return _inference_maker


@router.post("/inference", response_model=InferenceResponse)
async def ml_inference(req: InferenceRequest, request: Request, background_tasks: BackgroundTasks):
    """
    Trigger ML inference using the configured model.

    The model can be:
    - Auto-selected (best performing model)
    - Manually specified by name/version/stage, in the request
    - Configured with the /set-model endpoint
    """
    ml_interface = request.app.state.ml_interface

    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    try:
        inference_maker = get_inference_maker(ml_interface)

        # If specific model requested, temporarily set it
        model_used = None
        if req.model_name:
            logger.info(f"Using requested model: {req.model_name}")
            success = inference_maker.set_model_by_name(
                req.model_name,
                version=req.model_version,
                stage=req.model_stage
            )
            if not success:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model not found: {req.model_name}"
                )
            model_used = f"{req.model_name}:{req.model_version or req.model_stage}"

        result = inference_maker.infer(data=req.data)

        if result is None:
            raise HTTPException(
                status_code=500,
                detail="Inference failed - check model availability"
            )

        model_info = inference_maker.get_current_model_info()
        model_used = model_used or model_info.get('model_id', 'unknown')

        published = False
        if req.publish_result and ml_interface:
            try:
                result_message = json.dumps({
                    "model_used": model_used,
                    "result": result if isinstance(result, (dict, list, str, int, float)) else str(result),
                    "timestamp": None  # TODO: add timestamp
                })
                published = ml_interface.produce_to_kafka(req.result_topic, result_message)
                if published:
                    logger.info(f"Published inference result to {req.result_topic}")
            except Exception as e:
                logger.error(f"Failed to publish inference result: {e}")

        return InferenceResponse(
            status="success",
            model_used=model_used,
            predictions=result,
            published_to_kafka=published
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.post("/set-model")
async def set_model(req: ModelSelectionRequest, request: Request):
    """
    Manually select a specific model for inference.
    Disables auto-selection mode.
    """
    ml_interface = request.app.state.ml_interface

    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    try:
        inference_maker = get_inference_maker(ml_interface)

        success = inference_maker.set_model_by_name(
            req.model_name,
            version=req.version,
            stage=req.stage
        )

        if success:
            model_info = inference_maker.get_current_model_info()
            return {
                "status": "success",
                "message": f"Model set to {req.model_name}",
                "model_info": model_info
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found or failed to load: {req.model_name}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auto-mode")
async def toggle_auto_mode(req: AutoModeRequest, request: Request):
    """
    Enable or disable automatic model selection.
    When enabled, the best performing model will be selected automatically.
    """
    ml_interface = request.app.state.ml_interface

    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    try:
        inference_maker = get_inference_maker(ml_interface)
        inference_maker.toggle_auto_select(req.auto_mode)

        return {
            "status": "success",
            "auto_mode": req.auto_mode,
            "message": f"Auto-select mode {'enabled' if req.auto_mode else 'disabled'}"
        }

    except Exception as e:
        logger.error(f"Error toggling auto mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current-model")
async def get_current_model(request: Request):
    """Get information about the currently configured model"""
    ml_interface = request.app.state.ml_interface

    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    try:
        inference_maker = get_inference_maker(ml_interface)
        model_info = inference_maker.get_current_model_info()

        return {
            "status": "success",
            "model_info": model_info
        }

    except Exception as e:
        logger.error(f"Error getting current model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-cache")
async def clear_model_cache(request: Request):
    """Clear the cached model, forcing reload on next inference"""
    ml_interface = request.app.state.ml_interface

    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    try:
        inference_maker = get_inference_maker(ml_interface)
        inference_maker.clear_cache()

        return {
            "status": "success",
            "message": "Model cache cleared"
        }

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=List[ModelInfo])
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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}/status")
async def get_model_status(model_name: str, request: Request):
    """Get detailed status of a specific model"""
    ml_interface = request.app.state.ml_interface

    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    if not ml_interface.is_mlflow_connected():
        raise HTTPException(status_code=503, detail="MLFlow not connected")

    try:
        all_models = ml_interface.list_registered_models()
        model_info = next((m for m in all_models if m['name'] == model_name), None)

        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

        metrics = {}
        if model_info.get('latest_versions'):
            for version in model_info['latest_versions']:
                if version.get('stage') == 'Production':
                    run_id = version.get('run_id')
                    if run_id:
                        metrics = ml_interface.get_model_metrics(run_id)
                    break

        return {
            "status": "success",
            "model_name": model_name,
            "model_info": model_info,
            "metrics": metrics
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/best-model")
async def get_best_model(request: Request, metric: str = "accuracy"):
    """Get information about the best performing model"""
    ml_interface = request.app.state.ml_interface

    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    if not ml_interface.is_mlflow_connected():
        raise HTTPException(status_code=503, detail="MLFlow not connected")

    try:
        best_model = ml_interface.get_best_model(metric=metric)

        if not best_model:
            raise HTTPException(
                status_code=404,
                detail=f"No best model found with metric: {metric}"
            )

        return {
            "status": "success",
            "best_model": best_model
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting best model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
async def ml_train(
    req: TrainingRequest,
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Trigger ML training (placeholder for future implementation)
    """
    ml_interface = request.app.state.ml_interface

    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    logger.info(f"ML training requested for model: {req.model_name}")

    # TODO: Integrate with training pipeline
    # For now, just return a placeholder response

    return {
        "status": "not_implemented",
        "message": "Training endpoint not yet implemented",
        "model_name": req.model_name
    }
