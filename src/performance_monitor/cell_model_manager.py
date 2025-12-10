"""
Cell Model Manager

Handles automatic model instantiation for network cells.
When processed data arrives from Kafka, checks if models exist for the cell
and creates new models if needed.

Author: Thiago Vicente
"""

import logging
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd

from src.models import models

# Configure MLflow to use MinIO
os.environ.setdefault('MLFLOW_S3_ENDPOINT_URL', os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000'))
os.environ.setdefault('AWS_ACCESS_KEY_ID', os.getenv('AWS_ACCESS_KEY_ID', 'minio'))
os.environ.setdefault('AWS_SECRET_ACCESS_KEY', os.getenv('AWS_SECRET_ACCESS_KEY', 'minio123'))

logger = logging.getLogger(__name__)


class CellModelManager:
    """Manages ML models per network cell"""

    def __init__(self, ml_interface):
        """
        Initialize CellModelManager

        Args:
            ml_interface: MLInterface instance for MLflow and data storage access
        """
        self.ml_interface = ml_interface
        self.cell_cache = {}  # Cache of known cells with models
        logger.info("CellModelManager initialized")

    def process_network_data(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming network data message from Kafka.
        Called automatically when messages arrive on network.data.processed topic.

        Args:
            message_data: Message data from Kafka consumer

        Returns:
            Processed message data (passthrough for now)
        """
        logger.info(f"=== HANDLER CALLED === Message keys: {list(message_data.keys()) if message_data else 'None'}")
        try:
            content = message_data.get('content', '')
            if not content:
                logger.debug("Empty message content, skipping")
                return message_data

            # Parse JSON content
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse message JSON: {e}")
                return message_data

            # Extract cell identifier
            cell_index = self._extract_cell_index(data)
            if not cell_index:
                logger.warning(f"No cell_index found in message: {data}")
                return message_data

            logger.info(f"Processing data for cell: {cell_index}")

            # Check if model exists for this cell
            model_exists = self._check_model_exists(cell_index)

            if not model_exists:
                logger.info(f"No model found for cell {cell_index}, initiating model creation")
                self._create_model_for_cell(cell_index, data)
            else:
                logger.debug(f"Model already exists for cell {cell_index}")

            return message_data

        except Exception as e:
            logger.error(f"Error processing network data: {e}")
            return message_data

    def _extract_cell_index(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Extract cell identifier from message data.

        Args:
            data: Parsed message data

        Returns:
            Cell ID string or None
        """

        return data.get("cell_index",None)

    def _check_model_exists(self, cell_index: str) -> bool:
        """
        Check if a model exists for the given cell ID using MLflow tags.

        Args:
            cell_index: Cell identifier

        Returns:
            True if model exists, False otherwise
        """
        # Check cache first
        if cell_index in self.cell_cache:
            logger.debug(f"Cell {cell_index} found in cache")
            return True

        if not self.ml_interface.is_mlflow_connected():
            logger.warning("MLflow not connected, cannot check models")
            return False

        try:

            client = MlflowClient()
            registered_models = client.search_registered_models()

            # Search for models with matching cell_index tag
            for model in registered_models or []:
                for version in model.latest_versions or []:
                    try:
                        run = client.get_run(version.run_id)
                        tags = run.data.tags

                        if tags.get('cell_index') == cell_index:
                            logger.info(f"Found existing model for cell {cell_index}: {model.name}")
                            self.cell_cache[cell_index] = {
                                'model_name': model.name,
                                'version': version.version,
                                'found_at': datetime.now().isoformat()
                            }
                            return True
                    except Exception as e:
                        logger.debug(f"Error checking model version: {e}")
                        continue

            return False

        except Exception as e:
            logger.error(f"Error checking model existence for cell {cell_index}: {e}")
            return False

    def _create_model_for_cell(self, cell_index_str: str, data: Dict[str, Any]) -> None:
        """
        Instantiate and register models for the given cell.
        Creates one instance of each model type defined in src.models.

        Args:
            cell_index: Cell identifier
            data: Sample data that triggered the creation
        """
        try:
            logger.info(f"Instantiating models for cell {cell_index_str}")

            if not self.ml_interface.is_mlflow_connected():
                logger.error("MLflow not connected, cannot register models")
                return

            # Mark in cache to avoid duplicate creation attempts
            self.cell_cache[cell_index_str] = {
                'status': 'instantiating',
                'triggered_at': datetime.now().isoformat()
            }

            models_created = []

            # Create one instance of each model type
            for ModelClass in models:
                try:
                    cell_index = int(cell_index_str)
                    model_instance = ModelClass()
                    model_type_name = ModelClass.__name__.lower()
                    model_name = f"cell_{cell_index}_{model_type_name}"

                    model_info = self._register_model_with_mlflow(model_instance, model_name, cell_index)
                    if model_info:
                        models_created.append(model_info)
                        logger.info(f"Created {ModelClass.__name__} model for cell {cell_index}: {model_name}")
                except Exception as e:
                    logger.error(f"Error creating {ModelClass.__name__} for cell {cell_index}: {e}")

            # Update cache
            if models_created:
                self.cell_cache[cell_index] = {
                    'models': models_created,
                    'status': 'created',
                    'created_at': datetime.now().isoformat()
                }
                logger.info(f"Successfully created {len(models_created)} models for cell {cell_index}")
            else:
                logger.error(f"Failed to create any models for cell {cell_index}")
                self.cell_cache[cell_index]['status'] = 'creation_failed'

        except Exception as e:
            logger.error(f"Error creating models for cell {cell_index}: {e}")
            if cell_index in self.cell_cache:
                self.cell_cache[cell_index]['status'] = 'error'
                self.cell_cache[cell_index]['error'] = str(e)

    def _register_model_with_mlflow(self, model, model_name: str, cell_index: str) -> Optional[Dict[str, Any]]:
        """
        Register a model with MLflow, train it with mock data, and tag it with cell_index.
        Uses MLflow's standard sklearn format for compatibility.

        Args:
            model: Model instance (RandomForest or XGBoost)
            model_name: Name for the model in MLflow
            cell_index: Cell identifier

        Returns:
            Model info dict or None
        """
        try:
            with mlflow.start_run(run_name=f"{model_name}_init") as run:
                # Generate mock training data
                n_samples = 100
                n_features = 22  # Matches ProcessedLatency schema: 4*rsrp + 4*sinr + 4*rsrq + 4*latency + 4*cqi + 2*bandwidth

                # Create mock features resembling cell metrics
                # Features: rsrp_mean, rsrp_max, rsrp_min, rsrp_std,
                #           sinr_mean, sinr_max, sinr_min, sinr_std,
                #           rsrq_mean, rsrq_max, rsrq_min, rsrq_std,
                #           latency_mean, latency_max, latency_min, latency_std,
                #           cqi_mean, cqi_max, cqi_min, cqi_std,
                #           primary_bandwidth, ul_bandwidth
                np.random.seed(int(cell_index) % 10000)  # Reproducible per cell
                X = np.random.randn(n_samples, n_features)

                # Mock target variable (e.g., latency prediction)
                y = np.random.rand(n_samples) * 100  # Random latency values 0-100ms

                logger.info(f"Training {model_name} with mock data ({n_samples} samples, {n_features} features)")

                # Train the model
                loss = model.train(
                    min_loss=0.01,
                    max_epochs=10,
                    X=X,
                    y=y
                )

                # Log training metrics
                mlflow.log_param("n_samples", n_samples)
                mlflow.log_param("n_features", n_features)
                mlflow.log_param("training_type", "mock")
                mlflow.log_metric("mock_training_loss", loss)
                mlflow.log_metric("accuracy", 0.85)  # Mock accuracy

                # Log the TRAINED model (accessing the underlying sklearn/xgboost model)
                mlflow.sklearn.log_model(
                    sk_model=model.model,  # Use the trained .model attribute
                    artifact_path="model",
                    registered_model_name=model_name
                )

                logger.info(f"Logged and registered mock-trained model {model_name} with run_id {run.info.run_id}")

            # Set tags on the registered model
            client = MlflowClient()
            client.set_registered_model_tag(model_name, "cell_index", str(cell_index))
            client.set_registered_model_tag(model_name, "model_type", model.__class__.__name__)
            client.set_registered_model_tag(model_name, "status", "mock_trained")
            client.set_registered_model_tag(model_name, "training_data", "synthetic")

            return {
                'model_name': model_name,
                'model_type': model.__class__.__name__,
                'run_id': run.info.run_id,
                'status': 'mock_trained',
                'loss': loss
            }

        except Exception as e:
            logger.error(f"Error logging model {model_name} to MLflow: {e}")
            return None


    def get_cell_model_info(self, cell_index: str) -> Optional[Dict[str, Any]]:
        """
        Get model information for a specific cell.

        Args:
            cell_index: Cell identifier

        Returns:
            Model info dict or None
        """
        return self.cell_cache.get(cell_index)

    def list_cell_models(self) -> List[Dict[str, Any]]:
        """
        List all known cell models.

        Returns:
            List of cell model information
        """
        return [
            {'cell_index': cell_index, **info}
            for cell_index, info in self.cell_cache.items()
        ]

    def clear_cache(self) -> None:
        """Clear the cell model cache"""
        self.cell_cache.clear()
        logger.info("Cell model cache cleared")
