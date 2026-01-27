"""
ANN architecture definition.
"""

from typing import List

from src.config.model_config import ActivationType
from src.models.utils.utils_torch import get_activation, nn


class ANNNetwork(nn.Module):
    """
    Feedforward ANN for time series prediction.

    Supports two modes:
    1. Simple mode: input -> hidden -> output (when hidden_layers is None)
    2. Custom mode: input -> [custom layers] -> output (when hidden_layers is specified)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        activation: ActivationType = ActivationType.RELU,
        hidden_layers: List[int]|None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_size = input_size

        if hidden_layers:
            # Custom multi-layer configuration
            layers = []
            prev_size = input_size
            for layer_size in hidden_layers:
                layers.append(nn.Linear(prev_size, layer_size))
                layers.append(get_activation(activation))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_size = layer_size
            layers.append(nn.Linear(prev_size, 1))
            self.network = nn.Sequential(*layers)
        else:
            # Simple two-layer network
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                get_activation(activation),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_size, 1),
            )

    def forward(self, x):
        # Flatten sequence dimension: (batch, seq, features) -> (batch, seq*features)
        x = x.view(x.size(0), -1)
        return self.network(x)
