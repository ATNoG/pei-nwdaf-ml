"""LSTM model for time series prediction using PyTorch"""
from src.models.modelI import ModelI
import pickle
from typing import Type
import numpy as np
import logging
import io

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class LSTMNetwork(nn.Module):
    """PyTorch LSTM network architecture"""

    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_size]
        lstm_out, _ = self.lstm(x)
        # Use last output
        last_out = lstm_out[:, -1, :]
        out = self.relu(self.fc1(last_out))
        out = self.fc2(out)
        return out


class LSTM(ModelI):
    """LSTM model for univariate time series forecasting"""

    HEADER = b"LSTM_MODEL"
    SEQUENCE_LENGTH = 5
    FRAMEWORK = "pytorch"

    def __init__(self):
        """Initialize LSTM model"""
        super().__init__()
        self.model = None
        self.feature_count = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def predict(self, data):
        """
        Make predictions.

        Args:
            data: Input data with shape [sequence_length, num_features] or
                  [batch_size, sequence_length, num_features]

        Returns:
            Scalar prediction value
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")

        if data is None:
            raise ValueError("No data provided for prediction")

        # If 2D, add batch dimension
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)

        # Convert to torch tensor
        data_tensor = torch.from_numpy(data.astype(np.float32)).to(self.device)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(data_tensor)

        # Return scalar value
        return float(predictions.cpu().numpy().flatten()[0])

    def train(self, max_epochs: int, X, y):
        """
        Train the LSTM model.

        Args:
            max_epochs: Number of training epochs
            X: Training features with shape [num_samples, sequence_length, num_features]
            y: Target values with shape [num_samples]

        Returns:
            Final loss value
        """

        # Convert to numpy if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # Handle shape: if X is 2D, reshape to 3D
        if len(X.shape) == 2:
            # X is [num_samples, num_features], reshape to [num_samples, 1, num_features]
            X = X.reshape(X.shape[0], 1, X.shape[1])

        if len(X.shape) != 3:
            raise ValueError(f"Expected X shape [num_samples, sequence_length, num_features], got {X.shape}")

        _, num_features = X.shape[1], X.shape[2]
        self.feature_count = num_features

        logger.info(f"Training LSTM with shape {X.shape}, target shape {y.shape}")

        # Build model if not exists
        if self.model is None:
            self.model = LSTMNetwork(input_size=num_features).to(self.device)
            logger.info(f"Built LSTM model with input size {num_features}")

        # Prepare data
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        y_tensor = torch.from_numpy(y.astype(np.float32)).reshape(-1, 1).to(self.device)

        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Training loop
        batch_size = min(32, max(1, len(X) // 4))
        num_batches = (len(X) + batch_size - 1) // batch_size

        for epoch in range(max_epochs):
            total_loss = 0.0
            self.model.train()

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X))

                batch_X = X_tensor[start_idx:end_idx]
                batch_y = y_tensor[start_idx:end_idx]

                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logger.info(f"step {epoch}/{max_epochs}")


        final_loss = float(total_loss / num_batches)
        logger.info(f"Training complete. Final loss: {final_loss}")
        return final_loss

    def serialize(self) -> bytes:
        """
        Serialize model to bytes.

        Returns:
            bytes: Model serialized as HEADER + pickled model
        """
        if self.model is None:
            raise RuntimeError("Model not trained, cannot serialize")

        # Save model state dict to bytes
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        model_bytes = buffer.getvalue()

        payload = {
            "model_state": model_bytes,
            "feature_count": self.feature_count
        }
        return self.HEADER + pickle.dumps(payload)

    @classmethod
    def deserialize(cls: Type['LSTM'], b: bytes) -> 'LSTM':
        """
        Deserialize model from bytes.

        Args:
            b: Serialized model bytes

        Returns:
            LSTM instance loaded from bytes

        Raises:
            TypeError: If header doesn't match
        """
        if not b.startswith(cls.HEADER):
            raise TypeError("Invalid model header for LSTM.")

        data = pickle.loads(b[len(cls.HEADER):])

        obj = cls()
        obj.feature_count = data.get("feature_count")

        # Reconstruct model
        if obj.feature_count is not None:
            obj.model = LSTMNetwork(input_size=obj.feature_count)
            model_bytes = data.get("model_state")
            buffer = io.BytesIO(model_bytes)
            state_dict = torch.load(buffer, map_location=obj.device)
            obj.model.load_state_dict(state_dict)
            obj.model = obj.model.to(obj.device)

        return obj
