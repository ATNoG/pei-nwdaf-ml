"""
Model Service

Handles model instance creation, deletion, and management in MLflow.
"""
import logging
import json
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import tempfile

from src.utils.features import extract_features
from src.config.inference_type import InferenceConfig, get_all_inference_types, get_inference_config
from src.config.model_config import ModelConfig
from src.models import create_trainer, get_trainer_class
from datetime import datetime
logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self, ml_interface) -> None:
        self._ml_interface = ml_interface

    def _model_exists(self, model_name: str) -> bool:
        """Check if model exists in MLflow registry"""
        try:
            client = MlflowClient()
            try:
                client.get_registered_model(model_name)
                return True
            except Exception:
                return False
        except Exception as e:
            logger.error(f"Error checking model existence: {e}")
            return False

    def get_model_details(self, model_name: str) -> dict:
        """
        Get detailed information about a model including architecture and config.

        Args:
            model_name: Name of the model

        Returns:
            dict with model details

        Raises:
            ValueError: If model not found
        """
        client = MlflowClient()

        try:
            model = client.get_registered_model(model_name)
        except Exception:
            raise ValueError(f"Model '{model_name}' not found in registry")

        tags = model.tags

        # Get latest version
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"No versions found for model '{model_name}'")

        latest = max(versions, key=lambda v: int(v.version))

        # Get run to fetch config artifact
        run = mlflow.get_run(latest.run_id)

        config_dict = None
        try:
            # Download config artifact
            artifact_path = client.download_artifacts(latest.run_id, "config/model_config.json")
            with open(artifact_path, 'r') as f:
                config_dict = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config artifact: {e}")

        return {
            "name": model_name,
            "model_type": tags.get("model_type"),
            "analytics_type": tags.get("analytics_type"),
            "horizon": int(tags.get("horizon", 0)) if tags.get("horizon") else None,
            "version": latest.version,
            "stage": latest.current_stage,
            "config": config_dict,
            "last_training_time": latest.creation_timestamp / 1000 if latest.creation_timestamp else None,
            "training_loss": run.data.metrics.get("training_loss"),
            "run_id": latest.run_id
        }

    def delete_model_instance(self, model_name: str):
        """
        Delete a model instance from MLflow registry
        Args:
            model_name: Name of the model to delete
        """
        # check if model exists
        if not self._model_exists(model_name):
            raise ValueError(f"Model instance [{model_name}] does not exist")

        # find all configurations that have this model as default and reset them
        all_configs = get_all_inference_types()
        configs_updated = []

        for key, config in all_configs.items():
            # config.default_model stores the model type, not the full name
            # so we need to generate the full name and compare
            if config.default_model and config.get_model_name(config.default_model) == model_name:
                config.set_default_model(None)
                configs_updated.append(f"{config.name}_{config.window_duration_seconds}s")
                logger.info(f"Reset default model for config: {config.name} ({config.window_duration_seconds}s)")

        if configs_updated:
            logger.info(f"Updated {len(configs_updated)} configuration(s) that had {model_name} as default: {', '.join(configs_updated)}")

        # delete the model from MLflow registry
        try:
            client = MlflowClient()
            client.delete_registered_model(model_name)
            logger.info(f"Successfully deleted model instance: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to delete model instance: {str(e)}")


    def create_model_instance(
        self,
        horizon: int,
        analytics_type: str,
        model_type: str,
        name: str,
        model_config: ModelConfig|None = None,
    ) -> None:
        """
        Create and register a new instance of a model.

        Args:
            horizon: Window size in seconds for the model
            analytics_type: Type of analytics (e.g., 'latency')
            model_type: Type of model (e.g., 'ann', 'lstm')
            model_config: Optional model configuration. Uses defaults if not provided.
            name: Optional custom model name. Auto-generated if not provided.

        Returns:
            str: The model name that was created

        Raises:
            ValueError: If parameters are invalid or model already exists
        """
        # Validate parameters
        tup = (analytics_type, horizon)
        inf_config = get_inference_config(tup)
        if inf_config is None:
            raise ValueError(f"{analytics_type} for {horizon}s is not accepted")

        # Validate model type
        try:
            get_trainer_class(model_type)
        except ValueError:
            raise ValueError(f"Model [{model_type}] not found")

        # Use provided config or defaults
        config = model_config or ModelConfig.default()

        # Determine model name
        model_name = name

        # Ensure model doesn't exist yet
        if self._model_exists(model_name):
            raise ValueError(f"Model instance '{model_name}' already exists")

        # Create model
        self._instance_model(inf_config, model_type, model_name, config)


    def _instance_model(
        self,
        inf_config: InferenceConfig,
        model_type: str,
        model_name: str,
        config: ModelConfig,
    ):
        """
        Internal method to create and register a model instance.

        Args:
            inf_config: Inference configuration for data endpoints
            model_type: Type of model (e.g., 'ann', 'lstm')
            model_name: Name for the registered model
            config: Model configuration for hyperparameters
        """
        try:
            # Fetch example data to determine feature count
            try:
                example_data = self._ml_interface.request_data_from_storage(
                    endpoint=inf_config.example_endpoint,
                    method="GET"
                )

                if example_data and isinstance(example_data, list) and len(example_data) > 0:
                    sample = example_data[0]
                    features = extract_features(sample, inf_config.name)
                    n_features = len(features)
                    logger.info(f"Detected {n_features} features from example data")
                else:
                    n_features = 22
                    logger.warning(f"Could not fetch example data, using default {n_features} features")

            except Exception as e:
                logger.warning(f"Error fetching example data: {e}, using default 22 features")
                n_features = 22

            sequence_length = config.sequence.sequence_length

            # Create and train model with mock data
            with mlflow.start_run(run_name=f"{model_name}_init") as run:
                n_samples = 100

                # Generate mock training data
                np.random.seed(42)
                X = np.random.randn(n_samples, n_features)
                y = np.random.rand(n_samples) * 100

                logger.info(f"Training {model_name} with mock data ({n_samples} samples, {n_features} features)")

                if sequence_length > 1:
                    X = np.repeat(X[:, np.newaxis, :], sequence_length, axis=1)


                # Create trainer with config and train
                trainer = create_trainer(model_type, config)
                loss = trainer.train(max_epochs=10, X=X, y=y)
                # Log parameters and metrics
                mlflow.log_param("n_samples", n_samples)
                mlflow.log_param("n_features", n_features)
                mlflow.log_param("training_type", "mock_init")
                mlflow.log_param("inference_type", inf_config.name)
                mlflow.log_metric("training_loss", loss)

                # Save model config as artifact
                with tempfile.TemporaryDirectory() as tmpdir:
                    config_path = f"{tmpdir}/model_config.json"
                    with open(config_path, 'w') as f:
                        json.dump(config.to_dict(), f, indent=2)
                    mlflow.log_artifact(config_path, artifact_path="config")

                # Log model to MLflow (all trainers use PyTorch)
                mlflow.pytorch.log_model(
                    pytorch_model=trainer.get_model(),
                    artifact_path="model",
                    registered_model_name=model_name
                )

                logger.info(f"Registered model {model_name} with run_id {run.info.run_id}")

            # Tag the registered model
            client = MlflowClient()
            client.set_registered_model_tag(model_name, "analytics_type", inf_config.name)
            client.set_registered_model_tag(model_name, "model_type", model_type.lower())
            client.set_registered_model_tag(model_name, "auto_created", "true")
            client.set_registered_model_tag(model_name, "horizon", str(inf_config.window_duration_seconds))
            client.set_registered_model_tag(
                model_name, "last_trained", str(datetime.utcnow().timestamp())
            )

            return True

        except Exception as e:
            logger.error(f"Error creating model {model_name}: {e}", exc_info=True)
            return False
