"""
Unit tests for model parameterization components
"""
import pytest

from src.config.model_config import (
    ModelConfig,
    TrainingConfig,
    ArchitectureConfig,
    SequenceConfig,
    OptimizerType,
    LossType,
    ActivationType
)
from src.models.factory import get_available_model_types, get_trainer_class, create_trainer


class TestModelConfig:
    """Test model configuration dataclasses"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ModelConfig.default()
        assert config.training.learning_rate == 0.001
        assert config.training.optimizer == OptimizerType.ADAM
        assert config.architecture.hidden_size == 32
        assert config.sequence.sequence_length == 5

    def test_config_serialization(self):
        """Test config to_dict and from_dict"""
        config = ModelConfig()
        config_dict = config.to_dict()

        restored = ModelConfig.from_dict(config_dict)
        assert restored.training.learning_rate == config.training.learning_rate
        assert restored.architecture.hidden_size == config.architecture.hidden_size

    def test_custom_config(self):
        """Test custom configuration"""
        config = ModelConfig(
            training=TrainingConfig(learning_rate=0.01, max_epochs=200),
            architecture=ArchitectureConfig(hidden_size=128, dropout=0.5)
        )
        assert config.training.learning_rate == 0.01
        assert config.architecture.hidden_size == 128


class TestFactory:
    """Test factory pattern for creating trainers"""

    def test_get_available_types(self):
        """Test getting available model types"""
        types = get_available_model_types()
        assert isinstance(types, list)
        assert "ann" in types
        assert "lstm" in types

    def test_get_trainer_class(self):
        """Test retrieving trainer classes"""
        ann_trainer = get_trainer_class("ann")
        lstm_trainer = get_trainer_class("lstm")
        assert ann_trainer.__name__ == "ANNTrainer"
        assert lstm_trainer.__name__ == "LSTMTrainer"

    def test_case_insensitive(self):
        """Test case insensitive model type lookup"""
        trainer1 = get_trainer_class("ANN")
        trainer2 = get_trainer_class("Ann")
        assert trainer1 == trainer2

    def test_invalid_type(self):
        """Test error on invalid model type"""
        with pytest.raises(ValueError, match="Unknown model type"):
            get_trainer_class("invalid")

    def test_create_trainer(self):
        """Test creating trainer instances"""
        config = ModelConfig.default()
        trainer = create_trainer("ann", config)
        assert trainer is not None
        assert trainer.config == config

    def test_create_both_types(self):
        """Test creating different trainer types"""
        config = ModelConfig.default()
        ann = create_trainer("ann", config)
        lstm = create_trainer("lstm", config)
        assert ann.__class__.__name__ == "ANNTrainer"
        assert lstm.__class__.__name__ == "LSTMTrainer"
