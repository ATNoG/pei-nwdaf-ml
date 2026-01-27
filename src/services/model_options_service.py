"""
Service for providing model configuration options to the frontend.
"""

from src.config.model_config import OptimizerType, LossType, ActivationType
from src.models import get_available_model_types
from src.config.inference_type import get_all_inference_types


class ModelOptionsService:
    """Service for retrieving available model configuration options"""

    @staticmethod
    def get_config_options():
        """
        Get all available configuration options for model creation.

        Returns:
            Dictionary containing analytics types, horizons, model types,
            optimizers, loss functions, activations, and default values.
        """
        inference_configs = get_all_inference_types()
        analytics_types = list(set(cfg.name for cfg in inference_configs.values()))
        horizons = list(set(cfg.window_duration_seconds for cfg in inference_configs.values()))
        model_types = get_available_model_types()

        return {
            "analytics_types": sorted(analytics_types),
            "horizons": sorted(horizons),
            "model_types": sorted(model_types),
            "optimizers": [e.value for e in OptimizerType],
            "loss_functions": [e.value for e in LossType],
            "activations": [e.value for e in ActivationType],
            "defaults": {
                "learning_rate": 0.001,
                "max_epochs": 50,
                "hidden_size": 32,
                "num_layers": 2,
                "dropout": 0.2,
                "sequence_length": 5
            }
        }
