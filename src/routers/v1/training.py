"""
Model Training API Endpoint

Author: T.Vicente
"""
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
import logging

from src.trainer.trainer import ModelTrainer
from src.schemas.training import (
    ModelTrainingRequest,
    ModelTrainingStartResponse,
    ModelTrainingInfo
)
from src.config.inference_type import get_inference_config

logger = logging.getLogger(__name__)

router = APIRouter()


def _train_model_background(
    ml_interface,
    analytics_type: str,
    horizon: int,
    model_type: str
):
    """Background task to train model"""
    try:
        trainer = ModelTrainer(ml_interface)
        result = trainer.train_model(
            analytics_type=analytics_type,
            horizon=horizon,
            model_type=model_type,
            max_epochs=100,
            data_limit_per_cell=100
        )
        logger.info(f"Background training completed: {result}")
    except Exception as e:
        logger.error(f"Background training failed: {e}", exc_info=True)


@router.post("", response_model=ModelTrainingStartResponse)
async def start_training(
    training_request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    request: Request
):
    """
    Start training a model for an analytics type.

    Training runs in the background. Use GET endpoint to check status.

    Args:
        training_request: Analytics type and model type to train

    Returns:
        Confirmation that training has started

    Example:
        POST /api/v1/training
        {
            "analytics_type": "latency",
            "model_type": "xgboost"
        }
    """
    ml_interface = request.app.state.ml_interface
    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    # Validate analytics type and horizon match
    key = (training_request.analytics_type, training_request.horizon)
    config = get_inference_config(key)

    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"No model configuration found for analytics_type={training_request.analytics_type} with horizon={training_request.horizon}s"
        )

    model_name = config.get_model_name(training_request.model_type)

    logger.info(
        f"Starting background training for {model_name} "
        f"(analytics_type={training_request.analytics_type}, "
        f"horizon={training_request.horizon}s, "
        f"model_type={training_request.model_type})"
    )

    # Start training in background
    background_tasks.add_task(
        _train_model_background,
        ml_interface,
        training_request.analytics_type,
        training_request.horizon,
        training_request.model_type
    )

    return ModelTrainingStartResponse(
        status="training_started",
        model_name=model_name,
        message=f"Training started for {model_name}"
    )


@router.get("/{analytics_type}/{horizon}/{model_type}", response_model=ModelTrainingInfo)
async def get_training_info(
    analytics_type: str,
    horizon: int,
    model_type: str,
    request: Request
):
    """
    Get training information for a model.

    Args:
        analytics_type: Analytics type (e.g., 'latency')
        horizon: Prediction horizon in seconds (e.g., 60)
        model_type: Model type (e.g., 'xgboost')

    Returns:
        Training information including last training time and metrics

    Example:
        GET /api/v1/training/latency/60/xgboost
    """
    ml_interface = request.app.state.ml_interface
    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    # Get config using key lookup
    key = (analytics_type, horizon)
    config = get_inference_config(key)

    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"No model configuration found for analytics_type={analytics_type} with horizon={horizon}s"
        )

    trainer = ModelTrainer(ml_interface)
    info = trainer.get_model_info(analytics_type, horizon, model_type)

    if info["status"] == "error":
        raise HTTPException(status_code=500, detail=info["message"])
    elif info["status"] == "not_found":
        raise HTTPException(status_code=404, detail=info["message"])

    return ModelTrainingInfo(
        model_name=info["model_name"],
        model_version=info.get("model_version"),
        last_training_time=info.get("last_training_time"),
        training_loss=info.get("training_loss"),
        samples_used=int(info.get("samples_used", 0)) if info.get("samples_used") else None,
        features_used=int(info.get("features_used", 0)) if info.get("features_used") else None,
        run_id=info.get("run_id")
    )
