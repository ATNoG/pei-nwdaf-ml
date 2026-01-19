"""
Factory module for creating models and trainers.
"""

from typing import Type

from src.config.model_config import ModelConfig
from src.models.trainers.base_torch import BaseTrainer
from src.models.trainers.ann import ANNTrainer
from src.models.trainers.lstm import LSTMTrainer


# >TOCHECK< should this be stored on a db?
# vcnt: for now idts because we won't accept upload of models, this is a fixed list
_TRAINER_REGISTRY: dict[str, Type[BaseTrainer]] = {
    "ann": ANNTrainer,
    "lstm": LSTMTrainer,
}


def get_available_model_types() -> list[str]:
    """Return list of available model type names"""
    return list(_TRAINER_REGISTRY.keys())


def get_trainer_class(model_type: str) -> Type[BaseTrainer]:
    """
    Get trainer class for a model type.

    Args:
        model_type: Model type name (e.g., "ann", "lstm")

    Returns:
        Trainer class

    Raises:
        ValueError: If model type is not registered
    """
    model_type_lower = model_type.lower()
    if model_type_lower not in _TRAINER_REGISTRY:
        available = ", ".join(_TRAINER_REGISTRY.keys())
        raise ValueError(
            f"Unknown model type: {model_type}. Available types: {available}"
        )
    return _TRAINER_REGISTRY[model_type_lower]


def create_trainer(
    model_type: str,
    config: ModelConfig|None = None,
) -> BaseTrainer:
    """
    Create a trainer instance for the specified model type.

    Args:
        model_type: Model type name
        config: Optional model configuration. Uses defaults if not provided.

    Returns:
        Configured trainer instance ready for training

    Raises:
        ValueError: If model type is not registered
    """
    trainer_class = get_trainer_class(model_type)
    return trainer_class(config=config)
