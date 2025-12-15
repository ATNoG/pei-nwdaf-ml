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
from src.config.inference_type import InferenceConfig, get_inference_config
from src.models import models

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


    # TODO: handle models with same configuration
    def create_model_instance(self,horizon:int,analytics_type:str,model_type:str,
        input_sequence_length:int|None = None):
        """
        Create and register a new instance of a model
        Args:
            inf_config: InferenceConfig for this analytics type
            model_type: Type of model ( ann, lstm, etc. )
            horizon: window size that'll be used for models
            input_sequence_lenght: optional sequence lenght for instance
            name: Name of the model
        """

        sequence_length = input_sequence_length if input_sequence_length is not None else 1

        if  input_sequence_length <= 0:
            raise

        # validate parameters
        tup = (analytics_type,horizon)
        config = get_inference_config(tup)
        if config is None:
            raise ValueError(f"{analytics_type} for {horizon}s is not accepted")
