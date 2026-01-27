"""
LSTM-specific trainer.
"""

import torch.nn as nn
from typing import Optional

from src.config.model_config import ModelConfig
from src.models.trainers.base_torch import BaseTrainer
from src.models.networks.lstm import LSTMNetwork


class LSTMTrainer(BaseTrainer):
    """
    Trainer for LSTM models.
    """
    
    _is_training = False
    _lock = None

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)

    def _create_model(self, input_size: int) -> nn.Module:
        """Create LSTMNetwork with config parameters"""
        arch = self.config.architecture

        return LSTMNetwork(
            input_size=input_size,
            hidden_size=arch.hidden_size,
            num_layers=arch.num_layers,
            dropout=arch.dropout,
            activation=arch.activation,
        )
