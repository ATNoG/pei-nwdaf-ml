import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

logger = logging.getLogger(__name__)

class LSTMNetwork(nn.Module):
    """Internal LSTM network"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        dropout = 0.2 if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(self.relu(last_out))


class LSTM(ModelInterface):

    def __init__(
        self,
        input_fields: list[str],
        output_fields: list[str],
        window_duration_seconds: int,
        lookback_steps: int,
        forecast_steps: int,
        hidden_size: int,
        num_layers: int = 2
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
        self.num_layers = num_layers

    def _ensure_model(self):
        if self.model is None:
            # LSTM processes sequences per-timestep
            input_size = len(self.input_fields)
            # Output is flattened: forecast_steps * num_output_features
            output_size = self.forecast_steps * len(self.output_fields)

            self.model = LSTMNetwork(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
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

            epoch_loss = total_loss / num_batches
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{max_epochs}, loss={epoch_loss:.4f}")

                # Call status callback if provided
                if status_callback:
                    try:
                        status_callback(epoch, max_epochs, epoch_loss)
                    except Exception as e:
                        logger.warning(f"Status callback error: {e}")

        # return loss
        return float(total_loss/num_batches)

    def predict(self, X):
        self._ensure_model()

        # set evaluation mode
        self.model.eval()

        # disable gradient
        with torch.no_grad():
            pred = self.model(torch.from_numpy(X).to(self.device)).cpu().numpy()
        return pred
