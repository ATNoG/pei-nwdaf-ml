"""
Model Training API Endpoint

"""
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
import logging
import asyncio

from src.services.training_service import TrainingService
from src.schemas.training import (
    ModelTrainingRequest,
    ModelTrainingStartResponse,
    ModelTrainingInfo
)
from src.routers.websocket import get_training_status_manager

logger = logging.getLogger(__name__)

router = APIRouter()


def _train_model_background(
    ml_interface,
    analytics_type: str,
    horizon: int,
    model_type: str
):
    """Background task to train model with WebSocket status updates"""
    from src.config.inference_type import get_inference_config

    config = get_inference_config((analytics_type, horizon))
    if not config:
        logger.error(f"No config found for analytics_type={analytics_type}, horizon={horizon}")
        return

    model_name = config.get_model_name(model_type)
    status_manager = get_training_status_manager()

    def status_callback(current_epoch: int, total_epochs: int, loss: float = None):
        """Callback to update WebSocket clients with training progress"""
        status = {
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "status": "training" if current_epoch < total_epochs else "completed",
        }
        if loss is not None:
            status["loss"] = float(loss)

        # Run async broadcast in event loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(status_manager.update_training_status(model_name, status))
            loop.close()
        except Exception as e:
            logger.warning(f"Failed to update WebSocket status: {e}")

    try:
        # Initial status
        status_callback(0, 100, None)

        service = TrainingService(ml_interface)
        result = service.start_training(
            analytics_type=analytics_type,
            horizon=horizon,
            model_type=model_type,
            max_epochs=100,
            data_limit_per_cell=100,
            status_callback=status_callback
        )

        # Final status with result
        final_status = {
            "current_epoch": 100,
            "total_epochs": 100,
            "status": "completed",
            "message": "Training completed successfully",
            "training_loss": result.get("training_loss"),
            "samples_used": result.get("samples_used"),
        }
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(status_manager.update_training_status(model_name, final_status))
        loop.close()

    except Exception as e:
        logger.error(f"Background training failed: {e}", exc_info=True)

        # Error status
        error_status = {
            "current_epoch": 0,
            "total_epochs": 100,
            "status": "error",
            "message": str(e)
        }
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(status_manager.update_training_status(model_name, error_status))
            loop.close()
        except Exception:
            pass


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
        training_request: Analytics type, horizon, and model type to train

    Returns:
        Confirmation that training has started

    Raises:
        404: Analytics type/horizon not configured
        500: ML Interface not initialized

    Example:
        POST /api/v1/training
        {
            "analytics_type": "latency",
            "horizon": 60,
            "model_type": "xgboost"
        }
    """
    ml_interface = request.app.state.ml_interface
    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    service = TrainingService(ml_interface)

    try:
        # Validate request
        model_name = service.validate_training_request(
            analytics_type=training_request.analytics_type,
            horizon=training_request.horizon,
            model_type=training_request.model_type
        )

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

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


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

    Raises:
        404: Config or model not found
        500: ML Interface not initialized or error retrieving info

    Example:
        GET /api/v1/training/latency/60/xgboost
    """
    ml_interface = request.app.state.ml_interface
    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    service = TrainingService(ml_interface)

    try:
        info = service.get_model_info(analytics_type, horizon, model_type)

        return ModelTrainingInfo(
            model_name=info["model_name"],
            model_version=info.get("model_version"),
            last_training_time=info.get("last_training_time"),
            training_loss=info.get("training_loss"),
            samples_used=int(info.get("samples_used", 0)) if info.get("samples_used") else None,
            features_used=int(info.get("features_used", 0)) if info.get("features_used") else None,
            run_id=info.get("run_id")
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting training info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
