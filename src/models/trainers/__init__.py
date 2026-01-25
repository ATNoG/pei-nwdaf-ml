"""
Model trainers
>TOCHECK< is this the best approach?
vcnt: tried to divide training from raw model in order to make the code readable and maintainable
"""

from src.models.trainers.base_torch import BaseTrainer
from src.models.trainers.ann import ANNTrainer
from src.models.trainers.lstm import LSTMTrainer

__all__ = ["BaseTrainer", "ANNTrainer", "LSTMTrainer"]
