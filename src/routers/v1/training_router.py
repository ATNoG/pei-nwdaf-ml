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


def _train_model_background(
    ml_interface,
    model_name: str,
    model_type: str,
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

        # Check if training was blocked due to concurrent job
        if result.get("status") == "error" and "already training" in result.get("message", ""):
            # Update status to show blocked/conflict
            conflict_status = {
                "current_epoch": 0,
                "total_epochs": 100,
                "status": "conflict",
                "message": result.get("message")
            }
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(status_manager.update_training_status(model_name, conflict_status))
            loop.close()
            return

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
        # Always release the model lock when training completes or fails
        from src.models import get_trainer_class
        try:
            TrainerClass = get_trainer_class(model_type)
            if TrainerClass:
                TrainerClass.release_training()
        except Exception as e:
            logger.warning(f"Error releasing training lock: {e}")


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

    try:
        # Validate model exists and has metadata
        metadata = service.get_model_metadata(training_request.model_name)

        # Check if model is already training BEFORE scheduling background task
        from src.models import get_trainer_class
        TrainerClass = get_trainer_class(metadata["model_type"])
        if TrainerClass:
            # Try to acquire the lock
            if not TrainerClass.reserve_training():
                raise HTTPException(
                    status_code=409,
                    detail=f"Model {training_request.model_name} is already training. Please wait for the current training job to complete."
                )
            # Lock acquired - background task will release it when done
        
        logger.info(f"Starting background training for {training_request.model_name}")

        # Start training in background (lock already held, will be released in finally)
        background_tasks.add_task(
            _train_model_background,
            ml_interface,
            training_request.model_name,
            metadata["model_type"],
        )

        return ModelTrainingStartResponse(
            status="training_started",
            model_name=training_request.model_name,
            message=f"Training started for {training_request.model_name}"
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        # Re-raise HTTP exceptions (like our 409)
        raise
    except Exception as e:
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
