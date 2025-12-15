"""
Model Initializer

Ensures all required models exist on component startup.
Creates and registers missing models in MLflow.
"""
import logging
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

from src.utils.features import extract_features
from src.config.inference_type import InferenceConfig, get_all_inference_types, get_inference_config
from src.models import models_dict

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


    # TODO: handle models with same configuration
    def create_model_instance(self,horizon:int,analytics_type:str,model_type:str):
        """
        Create and register a new instance of a model
        Args:
            inf_config: InferenceConfig for this analytics type
            model_type: Type of model ( ann, lstm, etc. )
            horizon: window size that'll be used for models
            input_sequence_lenght: optional sequence lenght for instance
            name: Name of the model
        """

        # validate parameters
        tup = (analytics_type,horizon)
        config = get_inference_config(tup)
        if config is None:
            raise ValueError(f"{analytics_type} for {horizon}s is not accepted")

        ModelClass = models_dict.get(model_type)
        if ModelClass is None:
            raise ValueError(f"Model [{model_type}] not found")

        ## TODO: remove this and allow models with the same config but different names
        # ensure model doesnt exist yer
        model_name = config.get_model_name(model_type)
        if self._model_exists(model_name):
            raise ValueError("Instance already exists")
        ##

        # create model
        self._instance_model(config,ModelClass, model_name)

    def _instance_model(self,inf_config:InferenceConfig, ModelClass, model_name: str):
        try:

            # Fetch example data to determine feature count
            try:
                example_data = self._ml_interface.request_data_from_storage(
                    endpoint=inf_config.example_endpoint,
                    method="GET"
                )

                if example_data and isinstance(example_data, list) and len(example_data) > 0:
                    # Extract feature fields (exclude metadata)
                    sample = example_data[0]
                    features = extract_features(sample,inf_config.name)
                    n_features = len(features)
                    logger.info(f"Detected {n_features} features from example data")
                else:
                    # Fallback to default
                    n_features = 22
                    logger.warning(f"Could not fetch example data, using default {n_features} features")

            except Exception as e:
                logger.warning(f"Error fetching example data: {e}, using default 22 features")
                n_features = 22

            sequence_length = getattr(ModelClass, "SEQUENCE_LENGTH", 1)

            # Create and train model with mock data
            with mlflow.start_run(run_name=f"{model_name}_init") as run:
                n_samples = 100

                # Generate mock training data
                np.random.seed(42)
                X = np.random.randn(n_samples, n_features)
                y = np.random.rand(n_samples) * 100  # Mock target values

                logger.info(f"Training {model_name} with mock data ({n_samples} samples, {n_features} features)")

                if sequence_length > 1:
                    # For simplicity, repeat the same features for the sequence
                    X = np.repeat(X[:, np.newaxis, :], sequence_length, axis=1)

                # Create and train model instance
                model_instance = ModelClass()
                loss = model_instance.train(
                    max_epochs=10,
                    X=X,
                    y=y
                )

                # Log parameters and metrics
                mlflow.log_param("n_samples", n_samples)
                mlflow.log_param("n_features", n_features)
                mlflow.log_param("training_type", "mock_init")
                mlflow.log_param("inference_type", inf_config.name)
                mlflow.log_metric("training_loss", loss)
                mlflow.log_metric("accuracy", 0.85)  # Mock accuracy

                # Log model based on framework
                if ModelClass.FRAMEWORK == "pytorch":
                    mlflow.pytorch.log_model(
                        pytorch_model=model_instance.model,
                        artifact_path="model",
                        registered_model_name=model_name
                    )
                else:  # sklearn
                    mlflow.sklearn.log_model(
                        sk_model=model_instance.model,
                        artifact_path="model",
                        registered_model_name=model_name
                    )

                logger.info(f"Registered model {model_name} with run_id {run.info.run_id}")

            # Tag the registered model
            client = MlflowClient()
            client.set_registered_model_tag(model_name, "inference_type", inf_config.name)
            client.set_registered_model_tag(model_name, "model_type", ModelClass.__name__.lower())
            client.set_registered_model_tag(model_name, "auto_created", "true")

            return True

        except Exception as e:
            logger.error(f"Error creating model {model_name}: {e}", exc_info=True)
            return False
