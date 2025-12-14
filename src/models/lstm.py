import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import io
from typing import Type
from src.models.model_interface import ModelInterface
import logging

logger = logging.getLogger(__name__)

class LSTMNetwork(nn.Module):
    """Internal LSTM network"""

    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(self.relu(last_out))


class LSTM(ModelInterface):
    HEADER = b"LSTM_MODEL"
    SEQUENCE_LENGTH = 5
    FRAMEWORK = "pytorch"

    def __init__(self, input_size: int = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.input_size = input_size

    def _ensure_model(self):
        if self.model is None:
            if self.input_size is None:
                raise RuntimeError("input_size must be provided to initialize the model")
            self.model = LSTMNetwork(self.input_size).to(self.device)

    def train(self, X, y, max_epochs: int = 50) -> float:
        X = np.nan_to_num(np.array(X, dtype=np.float32))
        y = np.nan_to_num(np.array(y, dtype=np.float32)).reshape(-1, 1)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]

        self.input_size = X.shape[2]
        self._ensure_model()

        X_tensor = torch.from_numpy(X).to(self.device)
        y_tensor = torch.from_numpy(y).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        batch_size = min(32, max(1, len(X)//4))
        num_batches = (len(X)+batch_size-1)//batch_size

        # start pytorch training mode
        self.model.train()

        # train
        for epoch in range(max_epochs):
            total_loss = 0.0
            perm = torch.randperm(len(X))
            for i in range(num_batches):
                idx = perm[i*batch_size:(i+1)*batch_size]
                batch_X = X_tensor[idx]
                batch_y = y_tensor[idx]

                # reset gradient
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)
                if torch.isnan(loss):
                    logger.warning("NaN detected in batch loss, skipping batch")
                    continue
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{max_epochs}, loss={total_loss/num_batches:.4f}")

        # return loss
        return float(total_loss/num_batches)

    def predict(self, X):
        X = np.nan_to_num(np.array(X, dtype=np.float32))
        if X.ndim == 2:
            X = X[np.newaxis, :, :]
        self._ensure_model()

        # set evaluation mode
        self.model.eval()

        # disable gradient
        with torch.no_grad():
            pred = self.model(torch.from_numpy(X).to(self.device)).cpu().numpy()
        return float(np.nan_to_num(pred.flatten()[0]))

    def serialize(self) -> bytes:
        # check if model was trained
        if self.model is None:
            raise RuntimeError("Model not trained")
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        payload = {"input_size": self.input_size, "model_state": buffer.getvalue()}
        return self.HEADER + pickle.dumps(payload)

    @classmethod
    def deserialize(cls: Type["LSTM"], b: bytes) -> "LSTM":
        if not b.startswith(cls.HEADER):
            raise TypeError("Invalid model header")
        data = pickle.loads(b[len(cls.HEADER):])
        obj = cls(input_size=data["input_size"])
        obj._ensure_model()
        buffer = io.BytesIO(data["model_state"])
        obj.model.load_state_dict(torch.load(buffer, map_location=obj.device))
        return obj
