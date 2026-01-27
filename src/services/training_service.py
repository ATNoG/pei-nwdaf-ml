import logging
import json
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
from typing import Dict, Any, List, Callable, Optional
from collections import defaultdict
import tempfile

from src.models import create_trainer, get_trainer_class
from src.config.inference_type import get_inference_config
from src.config.model_config import ModelConfig
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

    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """
        Get model metadata from MLflow tags.

        Args:
            model_name: Name of the registered model

        Returns:
            dict: Model metadata including analytics_type, horizon, model_type

        Raises:
            ValueError: If model not found or missing metadata
        """
        try:
            model = self.client.get_registered_model(model_name)
        except Exception as e:
            raise ValueError(f"Model '{model_name}' not found in registry") from e

        tags = model.tags

        analytics_type = tags.get("analytics_type")
        model_type = tags.get("model_type")
        horizon_str = tags.get("horizon")

        if not analytics_type or not model_type or not horizon_str:
            raise ValueError(
                f"Model '{model_name}' is missing required metadata. "
                f"Expected tags: analytics_type, model_type, horizon"
            )

        try:
            horizon = int(horizon_str)
        except ValueError:
            raise ValueError(f"Invalid horizon value in model tags: {horizon_str}")

        return {
            "analytics_type": analytics_type,
            "model_type": model_type,
            "horizon": horizon
        }

    def _load_model_config(self, model_name: str) -> ModelConfig:
        """
        Load stored model config from MLflow artifacts.
        Config is static and set during model creation (version 1).

        Args:
            model_name: Name of the registered model

        Returns:
            ModelConfig: The stored model configuration

        Raises:
            ValueError: If config cannot be loaded
        """
        try:
            # Get version 1 (creation version) to load the static config
            version_1 = self.client.get_model_version(model_name, "1")

            artifact_path = self.client.download_artifacts(
                version_1.run_id, "config/model_config.json"
            )
            with open(artifact_path, 'r') as f:
                config_dict = json.load(f)

            return ModelConfig.from_dict(config_dict)

        except Exception as e:
            raise ValueError(f"Error loading config for '{model_name}': {e}") from e

    def train_model_by_name(
        self,
        model_name: str,
        data_limit_per_cell: int = 100,
        status_callback: Optional[Callable[[int, int, Optional[float]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Train a model by its name (retrieves metadata and config from MLflow).

        Args:
            model_name: Name of the model to train
            data_limit_per_cell: Max samples per cell
            status_callback: Training progress callback

        Returns:
            dict: Training result

        Raises:
            ValueError: If model not found or invalid metadata
        """
        # Get model metadata from MLflow tags
        metadata = self.get_model_metadata(model_name)

        # Load stored model config
        model_config = self._load_model_config(model_name)

        # Use max_epochs from the loaded config
        max_epochs = model_config.training.max_epochs

        # Call the original training method with the actual model name
        return self.start_training(
            model_name=model_name,
            analytics_type=metadata["analytics_type"],
            horizon=metadata["horizon"],
            model_type=metadata["model_type"],
            max_epochs=max_epochs,
            data_limit_per_cell=data_limit_per_cell,
            status_callback=status_callback,
            model_config=model_config,
        )

    def start_training(
        self,
        model_name: str,
        analytics_type: str,
        horizon: int,
        model_type: str,
        max_epochs: int = 100,
        data_limit_per_cell: int = 100,
        status_callback: Optional[Callable[[int, int, Optional[float]], None]] = None,
        model_config: Optional[ModelConfig] = None,
    ) -> Dict[str, Any]:

        inf_config = get_inference_config((analytics_type, horizon))
        if not inf_config:
            raise ValueError(
                f"No config found for analytics_type={analytics_type}, horizon={horizon}"
            )

        # Validate model type and get trainer class
        try:
            get_trainer_class(model_type)
        except ValueError:
            raise ValueError(f"Model type not found: {model_type}")

        # Use provided config or defaults
        config = model_config or ModelConfig.default()

        logger.info(f"Starting training for {model_name}")

        # ------------------------------------------------------
        # Fetch + group data by cell
        # ------------------------------------------------------
        known_cells = self.ml_interface.fetch_known_cells()
        if not known_cells:
            return {"status": "error", "message": "No cells found in storage"}

        raw_data = self.ml_interface.fetch_training_data_for_cells(
            endpoint=inf_config.storage_endpoint,
            cell_indexes=known_cells,
            window_duration_seconds=inf_config.window_duration_seconds,
            data_limit_per_cell=data_limit_per_cell,
        )

        if not raw_data:
            return {"status": "error", "message": "No training data available"}

        cells_data = self._group_by_cell(raw_data)

        # Get sequence length from config
        sequence_length = config.sequence.sequence_length

        if sequence_length > 1:
            X, y = self._prepare_sequence_data(
                cells_data, analytics_type, sequence_length
            )
        else:
            X, y = self._prepare_tabular_data(
                cells_data, analytics_type
            )

        if X is None or len(X) == 0:
            return {"status": "error", "message": "No valid training samples"}

        #TODO: normalize
        feature_axes = (0, 1) if X.ndim == 3 else 0
        feature_mean = X.mean(axis=feature_axes, keepdims=True)
        feature_std = X.std(axis=feature_axes, keepdims=True) + 1e-8

        # Apply normalization
        X = (X - feature_mean) / feature_std


        n_samples = len(X)
        n_features = X.shape[-1]

        with mlflow.start_run(run_name=f"{model_name}_training") as run:
            # Create trainer with config
            trainer = create_trainer(model_type, config)

                # Create callback wrapper for status updates
                def training_callback(current_epoch: int, total_epochs: int, loss: Optional[float]):
                    if status_callback:
                        try:
                            status_callback(current_epoch, total_epochs, loss)
                        except Exception as e:
                            logger.warning(f"Status callback error: {e}")

            loss = trainer.train(X=X, y=y, max_epochs=max_epochs, status_callback=training_callback)

            # Save normalization artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                mean_f_name = f"{model_name}_mean.npy"
                std_f_name = f"{model_name}_std.npy"

                mean_path = f"{tmpdir}/{mean_f_name}"
                std_path = f"{tmpdir}/{std_f_name}"

                np.save(mean_path, feature_mean)
                np.save(std_path, feature_std)

                mlflow.log_artifact(mean_path, artifact_path="normalization")
                mlflow.log_artifact(std_path, artifact_path="normalization")

                # Save model config as JSON artifact
                config_path = f"{tmpdir}/model_config.json"
                with open(config_path, 'w') as f:
                    json.dump(config.to_dict(), f, indent=2)
                mlflow.log_artifact(config_path, artifact_path="config")

            # Log parameters including model config
            mlflow.log_params({
                "analytics_type": analytics_type,
                "model_type": model_type,
                "n_samples": n_samples,
                "n_features": n_features,
                "n_cells": len(cells_data),
                "max_epochs": max_epochs,
                # Training config params
                "config.learning_rate": config.training.learning_rate,
                "config.optimizer": config.training.optimizer.value,
                "config.loss_function": config.training.loss_function.value,
                # Architecture config params
                "config.hidden_size": config.architecture.hidden_size,
                "config.num_layers": config.architecture.num_layers,
                "config.dropout": config.architecture.dropout,
                # Sequence config params
                "config.sequence_length": config.sequence.sequence_length,
            })

            mlflow.log_metric("training_loss", float(loss))

            # Log model to MLflow (all trainers use PyTorch)
            mlflow.pytorch.log_model(
                trainer.get_model(),
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
        self.client.set_registered_model_tag(model_name, "horizon", str(horizon))
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
        finally:
            # Always release the training lock
            ModelClass.release_training()

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

    def get_model_info_by_name(self, model_name: str) -> Dict[str, Any]:
        """
        Get training information for a model by name.

        Args:
            model_name: Name of the model

        Returns:
            dict: Model information including training history
        """
        try:
            # Check if model exists
            try:
                _ = self.client.get_registered_model(model_name)
            except Exception:
                raise ValueError(f"Model '{model_name}' not registered")

            # Get latest version
            model_versions = self.client.search_model_versions(f"name='{model_name}'")
            if not model_versions:
                raise ValueError(f"No versions found for model '{model_name}'")

            latest_version = max(model_versions, key=lambda v: int(v.version))

            # Get run info
            run = mlflow.get_run(latest_version.run_id)

            return {
                "model_name": model_name,
                "model_version": latest_version.version,
                "last_training_time": latest_version.creation_timestamp / 1000,
                "training_loss": run.data.metrics.get("training_loss"),
                "samples_used": run.data.params.get("n_samples"),
                "features_used": run.data.params.get("n_features"),
                "run_id": latest_version.run_id
            }

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            raise RuntimeError(f"Error retrieving model info: {str(e)}")

