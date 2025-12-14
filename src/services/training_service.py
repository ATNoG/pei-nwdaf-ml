import logging
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
from typing import Dict, Any, List
from collections import defaultdict

from src.models import models_dict
from src.config.inference_type import get_inference_config
from src.utils.features import extract_features
logger = logging.getLogger(__name__)


class TrainingService:
    """
    Handles model training with real data from storage.
    Aggregates data PER CELL before global training.
    """

    def __init__(self, ml_interface):
        self.ml_interface = ml_interface
        self.client = MlflowClient()

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

        model_class=models_dict.get(model_type,None)
        if  model_class is None:
            raise ValueError(
                f"Model [{model_type}] is not registered "
            )

        key = (analytics_type, horizon)
        config = get_inference_config(key)

        if not config:
            raise ValueError(
                f"No model configuration found for analytics_type={analytics_type} "
                f"with horizon={horizon}s"
            )

        return config.get_model_name(model_type)

    def start_training(
        self,
        analytics_type: str,
        horizon: int,
        model_type: str,
        max_epochs: int = 100,
        data_limit_per_cell: int = 100,
    ) -> Dict[str, Any]:

        config = get_inference_config((analytics_type, horizon))
        if not config:
            raise ValueError(
                f"No config found for analytics_type={analytics_type}, horizon={horizon}"
            )

        ModelClass = models_dict.get(model_type.lower())
        if not ModelClass:
            raise ValueError(f"Model type not found: {model_type}")

        model_name = config.get_model_name(model_type)
        logger.info(f"Starting training for {model_name}")

        # ------------------------------------------------------
        # Fetch + group data by cell
        # ------------------------------------------------------
        known_cells = self.ml_interface.fetch_known_cells()
        if not known_cells:
            return {"status": "error", "message": "No cells found in storage"}

        raw_data = self.ml_interface.fetch_training_data_for_cells(
            endpoint=config.storage_endpoint,
            cell_indexes=known_cells,
            window_duration_seconds=config.window_duration_seconds,
            data_limit_per_cell=data_limit_per_cell,
        )

        if not raw_data:
            return {"status": "error", "message": "No training data available"}

        cells_data = self._group_by_cell(raw_data)

        if ModelClass.SEQUENCE_LENGTH > 1:
            X, y = self._prepare_sequence_data(
                cells_data, analytics_type, ModelClass.SEQUENCE_LENGTH
            )
        else:
            X, y = self._prepare_tabular_data(
                cells_data, analytics_type
            )

        if X is None or len(X) == 0:
            return {"status": "error", "message": "No valid training samples"}

        n_samples = len(X)
        n_features = X.shape[-1]

        with mlflow.start_run(run_name=f"{model_name}_training") as run:
            model = ModelClass()
            loss = model.train(X=X, y=y, max_epochs=max_epochs)

            mlflow.log_params({
                "analytics_type": analytics_type,
                "model_type": model_type,
                "n_samples": n_samples,
                "n_features": n_features,
                "n_cells": len(cells_data),
                "max_epochs": max_epochs,
            })

            mlflow.log_metric("training_loss", float(loss))

            if ModelClass.FRAMEWORK == "pytorch":
                mlflow.pytorch.log_model(
                    model.model,
                    artifact_path="model",
                    registered_model_name=model_name,
                )
            else:
                mlflow.sklearn.log_model(
                    model.model,
                    artifact_path="model",
                    registered_model_name=model_name,
                )

            run_id = run.info.run_id

        latest_version = max(
            int(v.version)
            for v in self.client.search_model_versions(f"name='{model_name}'")
        )

        self.client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Production",
            archive_existing_versions=True,
        )

        self.client.set_registered_model_tag(model_name, "analytics_type", analytics_type)
        self.client.set_registered_model_tag(model_name, "model_type", model_type)
        self.client.set_registered_model_tag(
            model_name, "last_trained", str(datetime.utcnow().timestamp())
        )

        return {
            "status": "success",
            "model_name": model_name,
            "model_version": str(latest_version),
            "training_loss": float(loss),
            "samples_used": n_samples,
            "features_used": n_features,
            "run_id": run_id,
        }

    def _group_by_cell(self, raw_windows: List[Dict[str, Any]]):
        cells = defaultdict(list)
        for w in raw_windows:
            cell_id = w.get("cell_index")
            if cell_id is not None:
                cells[cell_id].append(w)
        return cells

    # ---------------- TABULAR MODELS ----------------
    def _prepare_tabular_data(self, cells_data, analytics_type: str):
        X_list, y_list = [], []
        target_key = f"{analytics_type}_mean"

        for _, windows in cells_data.items():
            for w in windows:
                if target_key not in w:
                    continue

                features = extract_features(w, analytics_type)
                if not features:
                    continue

                X_list.append(features)
                y_list.append(float(w[target_key]))

        if not X_list:
            return None, None

        keys = sorted(X_list[0].keys())

        X = np.array(
            [[f.get(k, 0.0) for k in keys] for f in X_list],
            dtype=np.float32,
        )
        y = np.array(y_list, dtype=np.float32)

        return np.nan_to_num(X), np.nan_to_num(y)

    # ---------------- SEQUENCE MODELS (LSTM) ----------------
    def _prepare_sequence_data(
        self,
        cells_data,
        analytics_type: str,
        seq_len: int,
    ):
        X_all, y_all = [], []
        target_key = f"{analytics_type}_mean"

        for _, windows in cells_data.items():
            if len(windows) < seq_len:
                continue

            for i in range(len(windows) - seq_len + 1):
                seq = windows[i : i + seq_len]
                features_seq = []

                for w in seq:
                    features = extract_features(w, analytics_type)
                    if not features:
                        break
                    features_seq.append(features)

                if len(features_seq) != seq_len:
                    continue

                keys = sorted(features_seq[0].keys())
                X_all.append(
                    [[f.get(k, 0.0) for k in keys] for f in features_seq]
                )
                y_all.append(float(seq[-1].get(target_key, 0.0)))

        if not X_all:
            return None, None

        X = np.array(X_all, dtype=np.float32)
        y = np.array(y_all, dtype=np.float32)

        return np.nan_to_num(X), np.nan_to_num(y)

    def get_model_info(self, analytics_type: str, horizon: int, model_type: str) -> Dict[str, Any]:
        """
        Get training information for a model.

        Args:
            analytics_type: Analytics type
            horizon: Prediction horizon in seconds
            model_type: Model type

        Returns:
            dict: Model information including training history
        """

        key = (analytics_type, horizon)
        config = get_inference_config(key)

        if not config:
            return {"status": "error", "message": f"No config found for analytics_type={analytics_type} with horizon={horizon}s"}

        model_name = config.get_model_name(model_type)

        try:
            # Check if model exists
            try:
                _ = self.client.get_registered_model(model_name)
            except Exception:
                return {
                    "status": "not_found",
                    "model_name": model_name,
                    "message": "Model not registered"
                }

            # Get latest version
            model_versions = self.client.search_model_versions(f"name='{model_name}'")
            if not model_versions:
                return {
                    "status": "not_found",
                    "model_name": model_name,
                    "message": "No versions found"
                }

            latest_version = max(model_versions, key=lambda v: int(v.version))

            # Get run info
            run = mlflow.get_run(latest_version.run_id)

            return {
                "status": "found",
                "model_name": model_name,
                "model_version": latest_version.version,
                "last_training_time": latest_version.creation_timestamp / 1000,
                "training_loss": run.data.metrics.get("training_loss"),
                "samples_used": run.data.params.get("n_samples"),
                "features_used": run.data.params.get("n_features"),
                "run_id": latest_version.run_id
            }

        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return {"status": "error", "message": str(e)}
