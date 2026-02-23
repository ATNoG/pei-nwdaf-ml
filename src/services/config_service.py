"""
Configuration Service

Business logic for retrieving system configuration.
Handles aggregation of inference types and model types.

"""
import logging
from typing import Any

from src.config.inference_type import get_all_inference_types, get_inference_config
from src.models import get_available_model_types
from src.schemas.config import InferenceTypeConfig, ConfigResponse

logger = logging.getLogger(__name__)


class ConfigService:
    """Service for handling configuration queries"""

    @staticmethod
    def get_system_config() -> ConfigResponse:
        """
        Get all available system configurations.

        Returns:
            ConfigResponse: Available inference types and supported models
        """
        inference_configs = get_all_inference_types()

        # Build inference types list
        inference_types_list = []
        for _, config in inference_configs.items():
            inference_types_list.append(
                InferenceTypeConfig(
                    name=config.name,
                    horizon=config.window_duration_seconds,
                    default_model=config.default_model or "",
                    description=config.description,
                )
            )

        # Get supported model types
        supported_model_types = get_available_model_types()

        return ConfigResponse(
            inference_types=inference_types_list,
            supported_model_types=supported_model_types
        )

    @staticmethod
    def update_default_model(
        analytics_type: str,
        horizon: int,
        model_type: str
    ) -> dict[str, Any]:
        """
        Update the default model for an inference type configuration.

        Args:
            analytics_type: Analytics type (e.g., 'latency')
            horizon: Prediction horizon in seconds
            model_type: New model type to set as default

        Returns:
            dict: Success message with updated configuration

        Raises:
            ValueError: If config not found or model type invalid
        """
        # Validate config exists
        key = (analytics_type, horizon)
        config = get_inference_config(key)

        if not config:
            raise ValueError(
                f"No configuration found for analytics_type={analytics_type} "
                f"with horizon={horizon}s"
            )

        # Validate model type exists
        available_types = get_available_model_types()
        if model_type.lower() not in available_types:
            raise ValueError(
                f"Invalid model type: {model_type}. "
                f"Supported types: {available_types}"
            )

        # Update default model
        old_default = config.default_model
        config.set_default_model(model_type)

        logger.info(
            f"Updated default model for {analytics_type} (horizon={horizon}s): "
            f"{old_default} -> {model_type}"
        )

        return {
            "status": "success",
            "message": f"Default model updated for {analytics_type} (horizon={horizon}s)",
            "analytics_type": analytics_type,
            "horizon": horizon,
            "old_default_model": old_default,
            "new_default_model": model_type
        }
