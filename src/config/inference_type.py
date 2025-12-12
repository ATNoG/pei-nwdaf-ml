"""Author: T.Vicente"""

from dataclasses import dataclass
from typing import Dict

@dataclass
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
    description: None|str = None       # Human-readable description
    _default_model:None|str = None      # Default model for the config

    @property
    def default_model(self)->str|None:
        return self._default_model

    def set_default_model(self,model:str)->None:
        """Change config default model. DOES NOT VALIDATE MODEL EXISTENCE"""
        self._default_model = model

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
INFERENCE_TYPES: Dict[tuple[str,int], InferenceConfig] = {}


def register_inference_type(config: InferenceConfig) -> InferenceConfig:
    """Register an inference type configuration."""
    key = (config.name,config.window_duration_seconds)
    INFERENCE_TYPES[key] = config
    return config


def get_inference_config(key:tuple[str,int]) -> InferenceConfig|None:
    """Get inference configuration by name."""
    return INFERENCE_TYPES.get(key)


def get_all_inference_types() -> Dict[tuple[str,int], InferenceConfig]:
    """Get all registered inference types."""
    return INFERENCE_TYPES


# Define available inference types
LATENCY_60 = register_inference_type(InferenceConfig(
    name="latency",
    storage_endpoint="/api/v1/processed/latency/",
    example_endpoint="/api/v1/processed/latency/example",
    model_prefix="latency",
    window_duration_seconds=60,
    description="Predict next 1 minute window",
))

LATENCY_300 = register_inference_type(InferenceConfig(
    name="latency",
    storage_endpoint="/api/v1/processed/latency/",
    example_endpoint="/api/v1/processed/latency/example",
    model_prefix="latency",
    window_duration_seconds=300,
    description="Predict next 5 minute window"
))
