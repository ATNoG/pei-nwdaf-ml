"""Author: T.Vicente"""

from dataclasses import dataclass
from typing import Optional, Dict

@dataclass(frozen=True)
class InferenceConfig:
    """
    Configuration for each inference type.
    Defines how to fetch data, name models, and handle inference for a specific metric.
    """
    name: str                          # config name
    storage_endpoint: str              # datastorage endpoint for fetching processed data
    example_endpoint: str              # endpoint to fetch example data format. Used to "format" models
    model_prefix: str                  # Prefix for model names in MLFlow
    window_duration_seconds:int        # Duration of window for this model
    description: Optional[str] = None  # Human-readable description

    def get_model_name(self, model_type: str) -> str:
        """
        Generate model name for this inference type and model type.

        Args:
            model_type: Type of model

        Returns:
            Model name string
        """
        return f"{self.model_prefix}_{model_type.lower()}_{self.window_duration_seconds}"


# Registry of all available inference types
INFERENCE_TYPES: Dict[str, InferenceConfig] = {}


def register_inference_type(config: InferenceConfig) -> InferenceConfig:
    """Register an inference type configuration."""
    INFERENCE_TYPES[config.name] = config
    return config


def get_inference_config(inference_type: str) -> Optional[InferenceConfig]:
    """Get inference configuration by name."""
    return INFERENCE_TYPES.get(inference_type)


def get_all_inference_types() -> Dict[str, InferenceConfig]:
    """Get all registered inference types."""
    return INFERENCE_TYPES


# Define available inference types
LATENCY_60 = register_inference_type(InferenceConfig(
    name="latency",
    storage_endpoint="/api/v1/processed/latency/",
    example_endpoint="/api/v1/processed/latency/example",
    model_prefix="latency",
    window_duration_seconds=60,
    description="Network latency prediction and analysis"
))
