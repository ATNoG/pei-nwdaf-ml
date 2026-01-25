"""
Base trainer with shared training logic for PyTorch models.
>TOCHECK< will we use other libraries?
"""

import logging
from typing import Callable, Any

import torch
import numpy as np
from src.models.utils.utils_torch import (OPTIMIZER_MAP, nn, optim, LOSS_MAP)
from src.config.model_config import ModelConfig

logger = logging.getLogger(__name__)


class BaseTrainer:
    """
    Base trainer class providing common training functionality.
    """

    FRAMEWORK = "pytorch"
    SEQUENCE_LENGTH = 5  # Default

    def __init__(self, config: ModelConfig|None = None):
        self.config = config or ModelConfig.default()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module|None = None

        # Override sequence length from config
        self.sequence_length = self.config.sequence.sequence_length

    def _get_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config"""
        if self.model is None:
            raise RuntimeError("Model must be set before creating optimizer")

        cfg = self.config.training


        opt_class = OPTIMIZER_MAP.get(cfg.optimizer, optim.Adam)
        return opt_class(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    def _get_criterion(self) -> nn.Module:
        """Create loss function based on config"""
        cfg = self.config.training
        return LOSS_MAP.get(cfg.loss_function, nn.MSELoss())

    def _get_batch_size(self, data_size: int) -> int:
        """Get batch size from config or auto-calculate"""
        if self.config.training.batch_size:
            return self.config.training.batch_size
        return min(32, max(1, data_size // 4))

    def _preprocess_data(self, X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess input data to numpy arrays"""
        X = np.nan_to_num(np.array(X, dtype=np.float32))
        y = np.nan_to_num(np.array(y, dtype=np.float32)).reshape(-1, 1)

        # Ensure 3D shape: (samples, sequence, features)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]

        return X, y

    def _create_model(self, input_size: int) -> nn.Module:
        """
        Create the neural network model. Must be implemented by subclasses.

        Args:
            input_size: Number of input features

        Returns:
            nn.Module instance
        """
        raise NotImplementedError("Subclasses must implement _create_model")

    def train(
        self,
        X: Any,
        y: Any,
        max_epochs: int|None = None,
        status_callback:Callable[[int, int, float], None]|None = None,
    ) -> float:
        """
        Train the model on provided data.

        Args:
            X: Training features (samples, sequence, features) or (samples, features)
            y: Training targets
            max_epochs: Override config max_epochs if provided
            status_callback: Optional callback(current_epoch, total_epochs, loss)

        Returns:
            Final training loss
        """
        X, y = self._preprocess_data(X, y)

        # Determine input size and create model
        input_size = X.shape[2]
        self.model = self._create_model(input_size).to(self.device)

        # Convert to tensors
        X_tensor = torch.from_numpy(X).to(self.device)
        y_tensor = torch.from_numpy(y).to(self.device)

        # Setup training
        optimizer = self._get_optimizer()
        criterion = self._get_criterion()

        epochs = max_epochs or self.config.training.max_epochs
        batch_size = self._get_batch_size(len(X))
        num_batches = (len(X) + batch_size - 1) // batch_size

        self.model.train()
        final_loss = 0.0

        for epoch in range(epochs):
            total_loss = 0.0
            perm = torch.randperm(len(X))

            for i in range(num_batches):
                idx = perm[i * batch_size : (i + 1) * batch_size]
                batch_X = X_tensor[idx]
                batch_y = y_tensor[idx]

                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)

                if torch.isnan(loss):
                    logger.warning("NaN detected in batch loss, skipping batch")
                    continue

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            epoch_loss = total_loss / num_batches
            final_loss = epoch_loss

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, loss={epoch_loss:.4f}")

                if status_callback:
                    try:
                        status_callback(epoch + 1, epochs, epoch_loss)
                    except Exception as e:
                        logger.warning(f"Status callback error: {e}")

        return float(final_loss)

    def get_model(self) -> nn.Module|None:
        """Return the underlying PyTorch model for MLflow logging"""
        return self.model

    def set_model(self, model: nn.Module) -> None:
        """Set the underlying model (e.g., after loading from MLflow)"""
        self.model = model.to(self.device)
