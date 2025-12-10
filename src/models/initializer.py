"""
Model Initializer

Ensures all required models exist on component startup.
Creates and registers missing models in MLflow.
"""
import logging
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

from src.config.inference_type import get_all_inference_types
from src.models import models

logger = logging.getLogger(__name__)


def initialize_models(ml_interface) -> dict:
    """
    Initialize all required models for registered inference types.

    Checks MLflow for existing models and creates missing ones.

    Args:
        ml_interface: MLInterface instance

    Returns:
        dict: Summary of initialization (created, existing, failed)
    """
    if not ml_interface.is_mlflow_connected():
        logger.error("MLflow not connected, cannot initialize models")
        return {"status": "error", "message": "MLflow not connected"}

    inference_types = get_all_inference_types()
    model_types = ["xgboost", "randomforest"]

    results = {
        "created": [],
        "existing": [],
        "failed": []
    }

    logger.info(f"Initializing models for {len(inference_types)} inference types")

    for inf_type_name, inf_config in inference_types.items():
        for model_type in model_types:
            model_name = inf_config.get_model_name(model_type)

            try:
                # Check if model exists
                if _model_exists(ml_interface, model_name):
                    logger.info(f"Model already exists: {model_name}")
                    results["existing"].append(model_name)
                else:
                    # Create and register model
                    logger.info(f"Creating model: {model_name}")
                    success = _create_model(
                        ml_interface,
                        inf_config,
                        model_type,
                        model_name
                    )
                    if success:
                        results["created"].append(model_name)
                        logger.info(f"Successfully created: {model_name}")
                    else:
                        results["failed"].append(model_name)
                        logger.error(f"Failed to create: {model_name}")

            except Exception as e:
                logger.error(f"Error initializing {model_name}: {e}")
                results["failed"].append(model_name)

    logger.info(
        f"Model initialization complete: "
        f"{len(results['created'])} created, "
        f"{len(results['existing'])} existing, "
        f"{len(results['failed'])} failed"
    )

    return results


def _model_exists(ml_interface, model_name: str) -> bool:
    """Check if model exists in MLflow registry"""
    try:
        client = MlflowClient()
        try:
            client.get_registered_model(model_name)
            return True
        except:
            return False
    except Exception as e:
        logger.error(f"Error checking model existence: {e}")
        return False


def _create_model(ml_interface, inf_config, model_type: str, model_name: str) -> bool:
    """
    Create and register a model in MLflow.

    Args:
        ml_interface: MLInterface instance
        inf_config: InferenceConfig for this inference type
        model_type: Type of model (xgboost/randomforest)
        model_name: Full model name for registration

    Returns:
        bool: Success status
    """
    try:
        # Get model class
        ModelClass = None
        for model_cls in models:
            if model_cls.__name__.lower() == model_type:
                ModelClass = model_cls
                break

        if not ModelClass:
            logger.error(f"Model class not found for type: {model_type}")
            return False

        # Fetch example data to determine feature count
        try:
            example_data = ml_interface.request_data_from_storage(
                endpoint=inf_config.example_endpoint,
                method="GET"
            )

            if example_data and isinstance(example_data, list) and len(example_data) > 0:
                # Extract feature fields (exclude metadata)
                sample = example_data[0]
                feature_keys = [
                    k for k in sample.keys()
                    if k not in ['window_start_time', 'window_end_time',
                                'window_duration_seconds', 'cell_index',
                                'network', 'sample_count']
                ]
                n_features = len(feature_keys)
                logger.info(f"Detected {n_features} features from example data")
            else:
                # Fallback to default
                n_features = 22
                logger.warning(f"Could not fetch example data, using default {n_features} features")

        except Exception as e:
            logger.warning(f"Error fetching example data: {e}, using default 22 features")
            n_features = 22

        # Create and train model with mock data
        with mlflow.start_run(run_name=f"{model_name}_init") as run:
            n_samples = 100

            # Generate mock training data
            np.random.seed(42)
            X = np.random.randn(n_samples, n_features)
            y = np.random.rand(n_samples) * 100  # Mock target values

            logger.info(f"Training {model_name} with mock data ({n_samples} samples, {n_features} features)")

            # Create and train model instance
            model_instance = ModelClass()
            loss = model_instance.train(
                min_loss=0.01,
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

            # Log model
            mlflow.sklearn.log_model(
                sk_model=model_instance.model,
                artifact_path="model",
                registered_model_name=model_name
            )

            logger.info(f"Registered model {model_name} with run_id {run.info.run_id}")

        # Tag the registered model
        client = MlflowClient()
        client.set_registered_model_tag(model_name, "inference_type", inf_config.name)
        client.set_registered_model_tag(model_name, "model_type", model_type)
        client.set_registered_model_tag(model_name, "auto_created", "true")

        return True

    except Exception as e:
        logger.error(f"Error creating model {model_name}: {e}", exc_info=True)
        return False
