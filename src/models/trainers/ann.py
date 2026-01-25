"""
ANN-specific trainer.
"""

import torch.nn as nn

from src.config.model_config import ModelConfig
from src.models.trainers.base_torch import BaseTrainer
from src.models.networks.ann import ANNNetwork


class ANNTrainer(BaseTrainer):
    """
    Trainer for ANN models.
    """

    def __init__(self, config: ModelConfig|None = None):
        super().__init__(config)

    def _create_model(self, input_size: int) -> nn.Module:
        """Create ANNNetwork with config parameters"""
        arch = self.config.architecture

        total_input_size = input_size * self.sequence_length

        return ANNNetwork(
            input_size=total_input_size,
            hidden_size=arch.hidden_size,
            activation=arch.activation,
            hidden_layers=arch.hidden_layers,
            dropout=arch.dropout,
        )
