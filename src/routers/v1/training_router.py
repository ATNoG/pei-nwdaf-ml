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
from src.routers.v1.websocket import get_training_status_manager

logger = logging.getLogger(__name__)

router = APIRouter()


class ModelAlreadyTrainingError(HTTPException):
    """Raised when attempting to train a model that is already being trained"""
    def __init__(self, model_name: str):
        super().__init__(
            status_code=409,  # 409 Conflict
            detail=f"Model '{model_name}' is already being trained. Please wait for the current training to complete."
        )


def _train_model_background(
    ml_interface,
    model_name: str,
):
    """Background task to train model with WebSocket status updates"""
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
        service = TrainingService(ml_interface)
        result = service.train_model_by_name(
            model_name=model_name,
            data_limit_per_cell=100,
            status_callback=status_callback,
        )

        # Final status with result
        final_status = {
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
    finally:
        # Release the training lock
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(status_manager.release_training_lock(model_name))
            loop.close()
        except Exception as e:
            logger.error(f"Error releasing training lock: {e}")


@router.post("", response_model=ModelTrainingStartResponse)
async def start_training(
    training_request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    request: Request
):
    """
    Start training a model by name.

    Training runs in the background. Use GET endpoint to check status.
    Uses the model's stored configuration for training parameters.

    Args:
        training_request: Model name

    Returns:
        Confirmation that training has started

    Raises:
        404: Model not found or missing metadata
        409: Model is already being trained
        500: ML Interface not initialized

    Example:
        POST /api/v1/training
        {
            "model_name": "latency_lstm_60"
        }
    """
    ml_interface = request.app.state.ml_interface
    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    service = TrainingService(ml_interface)
    status_manager = get_training_status_manager()

    try:
        # Validate model exists and has metadata
        service.get_model_metadata(training_request.model_name)

        # Check if model is already being trained
        if not await status_manager.acquire_training_lock(training_request.model_name):
            raise ModelAlreadyTrainingError(training_request.model_name)

        logger.info(f"Starting background training for {training_request.model_name}")

        # Start training in background
        background_tasks.add_task(
            _train_model_background,
            ml_interface,
            training_request.model_name,
        )

        return ModelTrainingStartResponse(
            status="training_started",
            model_name=training_request.model_name,
            message=f"Training started for {training_request.model_name}"
        )

    except ModelAlreadyTrainingError:
        raise
    except ValueError as e:
        # Release lock if we acquired it but validation failed
        await status_manager.release_training_lock(training_request.model_name)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Release lock if we acquired it but something else failed
        await status_manager.release_training_lock(training_request.model_name)
        logger.error(f"Error starting training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{model_name}", response_model=ModelTrainingInfo)
async def get_training_info(
    model_name: str,
    request: Request
):
    """
    Get training information for a model by name.

    Args:
        model_name: Name of the model (e.g., 'latency_ann_60')

    Returns:
        Training information including last training time and metrics

    Raises:
        404: Model not found
        500: ML Interface not initialized or error retrieving info

    Example:
        GET /api/v1/training/latency_ann_60
    """
    ml_interface = request.app.state.ml_interface
    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    service = TrainingService(ml_interface)

    try:
        info = service.get_model_info_by_name(model_name)

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
