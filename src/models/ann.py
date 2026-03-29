import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.models.model_interface import ModelInterface
import logging

logger = logging.getLogger(__name__)

class SimpleANNNetwork(nn.Module):
    """Simple feedforward ANN for time series"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # flatten sequence dimension
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class ANN(ModelInterface):

    def __init__(
        self,
        input_fields: list[str],
        output_fields: list[str],
        window_duration_seconds: int,
        lookback_steps: int,
        forecast_steps: int,
        hidden_size: int
    ):
        super().__init__(
            input_fields=input_fields,
            output_fields=output_fields,
            window_duration_seconds=window_duration_seconds,
            lookback_steps=lookback_steps,
            forecast_steps=forecast_steps,
            hidden_size=hidden_size
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def _ensure_model(self):
        if self.model is None:
            # ANN flattens the sequence: lookback_steps * num_input_features
            flattened_input_size = self.lookback_steps * len(self.input_fields)
            # Output is flattened: forecast_steps * num_output_features
            output_size = self.forecast_steps * len(self.output_fields)

            self.model = SimpleANNNetwork(
                input_size=flattened_input_size,
                hidden_size=self.hidden_size,
                output_size=output_size
            ).to(self.device)

    def train(self, X, y, max_epochs: int = 50, status_callback=None) -> float:
        self._ensure_model()

        X_tensor = torch.from_numpy(X).to(self.device)
        y_tensor = torch.from_numpy(y).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        batch_size = min(32, max(1, len(X)//4))
        num_batches = (len(X)+batch_size-1)//batch_size

        self.model.train()

        for epoch in range(max_epochs):
            total_loss = 0.0
            perm = torch.randperm(len(X))
            for i in range(num_batches):
                idx = perm[i*batch_size:(i+1)*batch_size]
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
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{max_epochs}, loss={epoch_loss:.4f}")

                # Call status callback if provided
                if status_callback:
                    try:
                        status_callback(epoch, max_epochs, epoch_loss)
                    except Exception as e:
                        logger.warning(f"Status callback error: {e}")

        return float(total_loss/num_batches)

    def predict(self, X):
        self._ensure_model()

        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.from_numpy(X).to(self.device)).cpu().numpy()
        return pred
