"""
Model Trainer

Handles training models with real data from storage.
Fetches data, prepares features, trains models, and registers in MLflow.

Author: T. Vicente
"""
import logging
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
from typing import Dict, Any

from src.config.inference_type import get_inference_config
from src.models import models

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains models for analytics types using data from storage"""

    def __init__(self, ml_interface):
        self.ml_interface = ml_interface
        self.client = MlflowClient()

    def train_model(
        self,
        analytics_type: str,
        horizon: int,
        model_type: str,
        max_epochs: int = 100,
        data_limit_per_cell: int = 100
    ) -> Dict[str, Any]:
        """
        Train a model for an analytics type using data from all cells.

        Args:
            analytics_type: Analytics type (e.g., 'latency')
            horizon: Prediction horizon in seconds (e.g., 60)
            model_type: Model type ('xgboost', 'randomforest')
            max_epochs: Maximum training epochs
            data_limit_per_cell: Maximum samples to fetch per cell

        Returns:
            dict: Training results with status, model_name, metrics, etc.
        """
        # Find config matching analytics_type and horizon
        from src.config.inference_type import get_all_inference_types

        config = None
        for cfg in get_all_inference_types().values():
            if cfg.name == analytics_type and cfg.window_duration_seconds == horizon:
                config = cfg
                break

        if not config:
            return {
                "status": "error",
                "message": f"No config found for analytics_type={analytics_type} with horizon={horizon}s"
            }

        model_name = config.get_model_name(model_type)
        logger.info(f"Starting training for {model_name}")

        try:
            # Fetch known cells from storage
            known_cells = self.ml_interface.fetch_known_cells()
            if not known_cells:
                return {
                    "status": "error",
                    "message": "No cells found in storage"
                }

            logger.info(f"Found {len(known_cells)} cells for training")

            # Fetch training data from all cells
            X, y, n_samples, n_features = self._prepare_training_data(
                config, known_cells, data_limit_per_cell
            )

            if X is None or len(X) == 0:
                return {
                    "status": "error",
                    "message": "No training data available"
                }

            # Get model class
            ModelClass = self._get_model_class(model_type)
            if not ModelClass:
                return {
                    "status": "error",
                    "message": f"Model class not found: {model_type}"
                }

            # Train model
            with mlflow.start_run(run_name=f"{model_name}_training") as run:
                model_instance = ModelClass()

                logger.info(
                    f"Training {model_name} with {n_samples} samples, "
                    f"{n_features} features"
                )

                loss = model_instance.train(
                    max_epochs=max_epochs,
                    X=X,
                    y=y
                )

                # Log parameters and metrics
                mlflow.log_param("analytics_type", analytics_type)
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("n_samples", n_samples)
                mlflow.log_param("n_features", n_features)
                mlflow.log_param("n_cells", len(known_cells))
                mlflow.log_param("max_epochs", max_epochs)
                mlflow.log_metric("training_loss", loss)

                # Log model to MLflow
                mlflow.sklearn.log_model(
                    sk_model=model_instance.model,
                    artifact_path="model",
                    registered_model_name=model_name
                )

                run_id = run.info.run_id
                logger.info(f"Model {model_name} trained with run_id {run_id}")

            # Get version number
            model_versions = self.client.search_model_versions(f"name='{model_name}'")
            latest_version = max([int(v.version) for v in model_versions])

            # Transition to Production stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=latest_version,
                stage="Production",
                archive_existing_versions=True  # Archive old versions
            )
            logger.info(f"Transitioned {model_name} version {latest_version} to Production")

            # Tag the model
            self.client.set_registered_model_tag(
                model_name, "analytics_type", analytics_type
            )
            self.client.set_registered_model_tag(
                model_name, "model_type", model_type
            )
            self.client.set_registered_model_tag(
                model_name, "last_trained", str(datetime.now().timestamp())
            )

            return {
                "status": "success",
                "model_name": model_name,
                "model_version": str(latest_version),
                "training_loss": float(loss),
                "samples_used": n_samples,
                "features_used": n_features,
                "run_id": run_id
            }

        except Exception as e:
            logger.error(f"Error training {model_name}: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }

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
        # Find config matching analytics_type and horizon
        from src.config.inference_type import get_all_inference_types

        config = None
        for cfg in get_all_inference_types().values():
            if cfg.name == analytics_type and cfg.window_duration_seconds == horizon:
                config = cfg
                break

        if not config:
            return {"status": "error", "message": f"No config found for analytics_type={analytics_type} with horizon={horizon}s"}

        model_name = config.get_model_name(model_type)

        try:
            # Check if model exists
            try:
                registered_model = self.client.get_registered_model(model_name)
            except:
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

    def _prepare_training_data(self, config, known_cells, data_limit_per_cell: int):
        """Prepare training data from storage for all cells"""
        try:
            # Fetch data using ml_interface
            all_data = self.ml_interface.fetch_training_data_for_cells(
                endpoint=config.storage_endpoint,
                cell_indexes=known_cells,
                window_duration_seconds=config.window_duration_seconds,
                data_limit_per_cell=data_limit_per_cell
            )

            if not all_data:
                logger.warning("No data collected from any cell")
                return None, None, 0, 0

            # Convert to DataFrame for easier processing
            df = pd.DataFrame(all_data)

            # Extract features (exclude metadata and target)
            exclude_cols = {
                'window_start_time', 'window_end_time',
                'window_duration_seconds', 'cell_index',
                'network', 'sample_count'
            }

            # Define target column based on analytics type
            target_mapping = {
                'latency': 'latency_mean',
                'throughput': 'throughput_mean',
                # Add more mappings as needed
            }

            target_col = target_mapping.get(config.name, 'latency_mean')

            if target_col not in df.columns:
                logger.error(f"Target column '{target_col}' not found in data")
                return None, None, 0, 0

            # Get all numerical columns except excluded ones and target
            feature_cols = [
                col for col in df.columns
                if col not in exclude_cols
                and col != target_col
                and not col.startswith(config.name + '_')  # Exclude all target-related columns
                and pd.api.types.is_numeric_dtype(df[col])
            ]

            if len(feature_cols) < 1:
                logger.error("Not enough feature columns")
                return None, None, 0, 0

            X = df[feature_cols].values
            y = df[target_col].values

            n_samples = len(X)
            n_features = X.shape[1]

            logger.info(f"Total collected: {n_samples} samples with {n_features} features from {len(known_cells)} cells")

            return X, y, n_samples, n_features

        except Exception as e:
            logger.error(f"Error fetching training data: {e}", exc_info=True)
            return None, None, 0, 0

    def _get_model_class(self, model_type: str):
        """Get model class by type name"""
        for model_cls in models:
            if model_cls.__name__.lower() == model_type:
                return model_cls
        return None
