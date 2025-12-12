"""
Training Service

Business logic for model training operations.
Handles training orchestration and model information retrieval.
"""
import logging
from typing import Dict, Any

from src.trainer.trainer import ModelTrainer
from src.config.inference_type import get_inference_config

logger = logging.getLogger(__name__)


class TrainingService:
    """Service for handling model training operations"""

    def __init__(self, ml_interface):
        """
        Initialize the service.

        Args:
            ml_interface: MLInterface instance
        """
        self.ml_interface = ml_interface
        self.trainer = ModelTrainer(ml_interface)

    def validate_training_request(
        self,
        analytics_type: str,
        horizon: int,
        model_type: str
    ) -> str:
        """
        Validate training request parameters.

        Args:
            analytics_type: Analytics type (e.g., 'latency')
            horizon: Prediction horizon in seconds
            model_type: Model type (e.g., 'xgboost')

        Returns:
            str: Model name

        Raises:
            ValueError: If config not found
        """
        key = (analytics_type, horizon)
        config = get_inference_config(key)

        if not config:
            raise ValueError(
                f"No model configuration found for analytics_type={analytics_type} "
                f"with horizon={horizon}s"
            )

        return config.get_model_name(model_type)

    def start_training(
        self,
        analytics_type: str,
        horizon: int,
        model_type: str,
        max_epochs: int = 100,
        data_limit_per_cell: int = 100
    ) -> Dict[str, Any]:
        """
        Execute model training.

        Args:
            analytics_type: Analytics type
            horizon: Prediction horizon in seconds
            model_type: Model type
            max_epochs: Number of training epochs
            data_limit_per_cell: Max data samples per cell

        Returns:
            dict: Training result

        Raises:
            RuntimeError: If training fails
        """
        try:
            result = self.trainer.train_model(
                analytics_type=analytics_type,
                horizon=horizon,
                model_type=model_type,
                max_epochs=max_epochs,
                data_limit_per_cell=data_limit_per_cell
            )

            if result.get("status") == "error":
                raise RuntimeError(result.get("message", "Training failed"))

            logger.info(f"Training completed successfully: {result}")
            return result

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise RuntimeError(f"Training failed: {str(e)}")

    def get_model_info(
        self,
        analytics_type: str,
        horizon: int,
        model_type: str
    ) -> Dict[str, Any]:
        """
        Get training information for a model.

        Args:
            analytics_type: Analytics type
            horizon: Prediction horizon in seconds
            model_type: Model type

        Returns:
            dict: Model training information

        Raises:
            ValueError: If config not found
            RuntimeError: If model not found or error retrieving info
        """
        # Validate config exists
        key = (analytics_type, horizon)
        config = get_inference_config(key)

        if not config:
            raise ValueError(
                f"No model configuration found for analytics_type={analytics_type} "
                f"with horizon={horizon}s"
            )

        # Get model info
        info = self.trainer.get_model_info(analytics_type, horizon, model_type)

        if info["status"] == "error":
            raise RuntimeError(info.get("message", "Error retrieving model info"))
        elif info["status"] == "not_found":
            raise ValueError(info.get("message", "Model not found"))

        return info
