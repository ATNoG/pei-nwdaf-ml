"""
LSTM network architecture definition.
"""

from src.config.model_config import ActivationType
from src.models.utils.utils_torch import get_activation, nn


class LSTMNetwork(nn.Module):
    """
    LSTM network for time series prediction.

    Architecture: LSTM layers -> activation -> fully connected output
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        activation: ActivationType = ActivationType.RELU,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output layer
        self.activation = get_activation(activation)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape
        lstm_out, _ = self.lstm(x)

        # Take last timestep output
        last_out = lstm_out[:, -1, :]

        # Apply activation and output layer
        return self.fc(self.activation(last_out))
