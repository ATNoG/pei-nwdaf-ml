"""
Utils for pytorch based models
>TOCHECK< will we use other libraries?
"""

from src.config.model_config import ActivationType, OptimizerType, LossType
import torch.optim as optim
import torch.nn as nn


def get_activation(activation: ActivationType) -> nn.Module:
    """Get activation module from enum"""
    activations = {
        ActivationType.RELU: nn.ReLU(),
        ActivationType.TANH: nn.Tanh(),
        ActivationType.SIGMOID: nn.Sigmoid(),
    }
    return activations.get(activation, nn.ReLU())

OPTIMIZER_MAP = {
    OptimizerType.ADAM: optim.Adam,
    OptimizerType.SGD: optim.SGD,
}

LOSS_MAP = {
    LossType.MSE: nn.MSELoss(),
    LossType.MAE: nn.L1Loss(),
}
