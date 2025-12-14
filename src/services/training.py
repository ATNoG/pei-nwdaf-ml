import logging
import numpy as np
from typing import Dict, Any, Type
from src.models import models_dict, ModelInterface
from src.config.inference_type import get_inference_config

logger = logging.getLogger(__name__)

class TrainingService:
    """Handles model training with real data from storage"""

    def __init__(self, ml_interface):
        self.ml_interface = ml_interface

    def train_model_for_cell(
        self,
        analytics_type: str,
        horizon: int,
        model_type: str,
        max_epochs: int = 50,
        data_limit_per_cell: int = 100
    ) -> Dict[str, Any]:
        """Fetch data, prepare sequences, and train a model"""

        ModelClass: Type[ModelInterface] = models_dict.get(model_type.lower())
        if not ModelClass:
            raise ValueError(f"Model type not found: {model_type}")

        # Fetch known cells
        known_cells = self.ml_interface.fetch_known_cells()
        if not known_cells:
            return {"status": "error", "message": "No cells found in storage"}

        X_list, y_list = [], []

        # Collect data from each cell
        for cell in known_cells:
            windows = self.ml_interface.fetch_training_data_for_cell(
                analytics_type=analytics_type,
                horizon=horizon,
                cell_index=cell,
                limit=data_limit_per_cell
            )
            if not windows:
                continue

            features, target = self._prepare_training_data(windows, analytics_type, ModelClass.SEQUENCE_LENGTH)
            if features is not None and target is not None:
                X_list.append(features)
                y_list.append(target)

        if not X_list:
            return {"status": "error", "message": "No training data available"}

        # Stack all cells
        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        logger.info(f"Training {model_type} with {len(X)} samples")
        trainer = ModelClass(input_size=X.shape[2])
        loss = trainer.train(X, y, max_epochs=max_epochs)

        return {
            "status": "success",
            "model_type": model_type,
            "training_loss": float(loss),
            "samples_used": len(X),
            "features_used": X.shape[2]
        }

    def _prepare_training_data(self, windows_data: list, analytics_type: str, sequence_length: int):
        """Convert raw window dicts into sequences"""
        if len(windows_data) < sequence_length:
            return None, None

        X_list, y_list = [], []
        for i in range(len(windows_data) - sequence_length + 1):
            seq_windows = windows_data[i:i+sequence_length]
            features = []
            for w in seq_windows:
                f = [v if isinstance(v, (int, float)) and v is not None else 0.0 for v in w.values()]
                features.append(f)
            X_list.append(np.array(features, dtype=np.float32))
            # target is last window's value of target feature
            y_list.append(float(seq_windows[-1].get(f"{analytics_type}_mean", 0.0)))
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    def validate_training_request(
        self,
        analytics_type: str,
        horizon: int,
        model_type: str
    ) -> str:
        """
        Validate training request parameters.

        Args:
            analytics_type: Analytics type (e.g., 'latency')
            horizon: Prediction horizon in seconds
            model_type: Model type (e.g., 'xgboost')

        Returns:
            str: Model name

        Raises:
            ValueError: If config not found
        """
        key = (analytics_type, horizon)
        config = get_inference_config(key)

        if not config:
            raise ValueError(
                f"No model configuration found for analytics_type={analytics_type} "
                f"with horizon={horizon}s"
            )

        return config.get_model_name(model_type)
