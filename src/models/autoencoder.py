"""MLP Autoencoder for unsupervised anomaly detection."""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.secure_wrapper import SecureWrapper

logger = logging.getLogger(__name__)


class AutoencoderNetwork(nn.Module):
    """Symmetric MLP autoencoder: encoder → bottleneck → decoder."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        bottleneck = max(4, hidden_size // 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class Autoencoder:
    """Autoencoder wrapper for training and scoring."""

    def __init__(self, input_size: int, hidden_size: int = 32):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SecureWrapper(AutoencoderNetwork(input_size, hidden_size)).to(self.device)

    @property
    def public_key(self) -> bytes:
        return self.model.public_key

    def decrypt(self, ciphertext: bytes) -> bytes:
        return self.model.decrypt(ciphertext)

    def train(
        self,
        X: np.ndarray,
        max_epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        status_callback=None,
    ) -> float:
        """
        Train the autoencoder on input data.

        Args:
            X: Input data of shape (N, num_features)
            max_epochs: Maximum training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            status_callback: Optional callback(epoch, max_epochs, loss)

        Returns:
            Final epoch loss
        """
        X_tensor = torch.from_numpy(
            np.nan_to_num(np.array(X, dtype=np.float32))
        ).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        batch_size = min(batch_size, max(1, len(X) // 4))
        num_batches = max(1, (len(X) + batch_size - 1) // batch_size)

        self.model.train()
        epoch_loss = 0.0

        for epoch in range(max_epochs):
            total_loss = 0.0
            perm = torch.randperm(len(X_tensor))

            for i in range(num_batches):
                idx = perm[i * batch_size : (i + 1) * batch_size]
                batch = X_tensor[idx]

                optimizer.zero_grad()
                reconstructed = self.model(batch)
                loss = criterion(reconstructed, batch)

                if torch.isnan(loss):
                    logger.warning("NaN detected in batch loss, skipping batch")
                    continue

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            epoch_loss = total_loss / num_batches

            if status_callback:
                try:
                    status_callback(epoch, max_epochs, epoch_loss)
                except InterruptedError:
                    raise
                except Exception as e:
                    logger.warning(f"Status callback error: {e}")

        return epoch_loss

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-sample reconstruction error (MSE).

        Args:
            X: Input data of shape (N, num_features)

        Returns:
            Array of shape (N,) with reconstruction error per sample
        """
        X_tensor = torch.from_numpy(
            np.nan_to_num(np.array(X, dtype=np.float32))
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = ((reconstructed - X_tensor) ** 2).mean(dim=1)

        return errors.cpu().numpy()
