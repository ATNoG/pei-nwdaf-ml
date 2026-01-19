"""
Tests for the model factory module.
"""
import pytest

from src.models.factory import (
    get_available_model_types,
    get_trainer_class,
    create_trainer,
)
from src.models.trainers import ANNTrainer, LSTMTrainer, BaseTrainer
from src.config.model_config import ModelConfig, TrainingConfig, ArchitectureConfig


class TestGetAvailableModelTypes:
    """Tests for get_available_model_types()"""

    def test_returns_list(self):
        """Should return a list of model types"""
        types = get_available_model_types()
        assert isinstance(types, list)

    def test_contains_ann_and_lstm(self):
        """Should contain ann and lstm model types"""
        types = get_available_model_types()
        assert "ann" in types
        assert "lstm" in types


class TestGetTrainerClass:
    """Tests for get_trainer_class()"""

    def test_ann_returns_ann_trainer(self):
        """get_trainer_class('ann') should return ANNTrainer"""
        trainer_class = get_trainer_class("ann")
        assert trainer_class is ANNTrainer

    def test_lstm_returns_lstm_trainer(self):
        """get_trainer_class('lstm') should return LSTMTrainer"""
        trainer_class = get_trainer_class("lstm")
        assert trainer_class is LSTMTrainer

    def test_case_insensitive(self):
        """Model type lookup should be case insensitive"""
        assert get_trainer_class("ANN") is ANNTrainer
        assert get_trainer_class("LSTM") is LSTMTrainer
        assert get_trainer_class("Ann") is ANNTrainer

    def test_invalid_type_raises_value_error(self):
        """Invalid model type should raise ValueError"""
        with pytest.raises(ValueError) as exc_info:
            get_trainer_class("invalid_model")
        assert "Unknown model type" in str(exc_info.value)


class TestCreateTrainer:
    """Tests for create_trainer()"""

    def test_create_ann_trainer(self):
        """create_trainer('ann') should return ANNTrainer instance"""
        trainer = create_trainer("ann")
        assert isinstance(trainer, ANNTrainer)
        assert isinstance(trainer, BaseTrainer)

    def test_create_lstm_trainer(self):
        """create_trainer('lstm') should return LSTMTrainer instance"""
        trainer = create_trainer("lstm")
        assert isinstance(trainer, LSTMTrainer)
        assert isinstance(trainer, BaseTrainer)

    def test_create_with_default_config(self):
        """Trainer created without config should use defaults"""
        trainer = create_trainer("ann")
        assert trainer.config.training.learning_rate == 0.001
        assert trainer.config.architecture.hidden_size == 32

    def test_create_with_custom_config(self):
        """Trainer created with config should use custom values"""
        config = ModelConfig(
            training=TrainingConfig(learning_rate=0.01, max_epochs=200),
            architecture=ArchitectureConfig(hidden_size=128),
        )
        trainer = create_trainer("lstm", config)
        assert trainer.config.training.learning_rate == 0.01
        assert trainer.config.training.max_epochs == 200
        assert trainer.config.architecture.hidden_size == 128

    def test_create_invalid_type_raises_value_error(self):
        """create_trainer with invalid type should raise ValueError"""
        with pytest.raises(ValueError):
            create_trainer("nonexistent")
