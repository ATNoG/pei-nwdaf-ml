import torch
import torch.nn as nn
import numpy as np

class SimpleModel(ModelInterface):
    def train(self, X, y, max_epochs=100, status_callback=None) -> float:
        return 0.0

    def predict(self, X):
        return np.zeros((len(X), 1))
