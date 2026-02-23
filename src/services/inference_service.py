"""Service for running inference on trained models."""

import asyncio
import logging

import numpy as np
import mlflow

from src.schemas.model import ArchitectureType, ModelConfig
from src.schemas.inference import ForecastStepPrediction
from src.services.mlflow_service import MLflowService
from src.services.data_storage_client import DataStorageClient
from src.services.data_preparation import calculate_timestamps, prepare_last_sequence
from src.models import MODEL_REGISTRY

logger = logging.getLogger(__name__)


class InferenceService:
    """Service for loading trained models and running predictions."""

    def __init__(
        self,
        mlflow_service: MLflowService,
        data_storage_client: DataStorageClient,
    ):
        self.mlflow_service = mlflow_service
        self.data_storage_client = data_storage_client

    async def predict(self, model_id: str, cell_id: int) -> dict:
        """
        Run inference for a single cell using a trained model.

        Args:
            model_id: UUID of the model.
            cell_id: Cell index to predict for.

        Returns:
            Dict containing model info and structured predictions.

        Raises:
            ValueError: If model not found, not trained, or insufficient data.
            RuntimeError: If prediction fails.
        """
        # Load model detail (config + version info) — sync call, run in thread
        model_detail = await asyncio.to_thread(
            self.mlflow_service.get_model, model_id
        )
        config = model_detail.config

        # Validate model has been trained
        if model_detail.latest_version is None:
            raise ValueError(
                f"Model '{model_id}' has no trained versions. "
                f"Train the model first via POST /v1/training/train"
            )

        # Load trained model from MLflow — sync call, run in thread
        model = await asyncio.to_thread(
            self._load_trained_model,
            model_id=model_id,
            version=model_detail.latest_version,
            config=config,
        )

        # Fetch recent cell data
        lookback_seconds = config.lookback_steps * config.window_duration_seconds
        buffer_seconds = config.window_duration_seconds
        start_ts, end_ts = calculate_timestamps(lookback_seconds + buffer_seconds)

        cell_data = await self.data_storage_client.fetch_cell_data(
            cell_index=cell_id,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            window_duration_seconds=config.window_duration_seconds,
        )

        if not cell_data:
            raise ValueError(
                f"No data available for cell {cell_id} in the "
                f"last {lookback_seconds} seconds"
            )

        # Sort by timestamp
        cell_data.sort(key=lambda x: x.get("window_start_time", 0))

        # Prepare input sequence
        X = prepare_last_sequence(
            cell_data=cell_data,
            input_fields=config.input_fields,
            lookback_steps=config.lookback_steps,
        )

        X = np.nan_to_num(X, nan=0.0)

        # Run prediction
        try:
            raw_predictions = model.predict(X)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

        # Structure predictions
        predictions = self._structure_predictions(
            raw_predictions=raw_predictions,
            output_fields=config.output_fields,
            forecast_steps=config.forecast_steps,
        )

        return {
            "model_id": model_id,
            "model_name": model_detail.name,
            "model_version": model_detail.latest_version,
            "architecture": config.architecture,
            "cell_id": cell_id,
            "lookback_steps": config.lookback_steps,
            "forecast_steps": config.forecast_steps,
            "window_duration_seconds": config.window_duration_seconds,
            "input_fields": config.input_fields,
            "output_fields": config.output_fields,
            "predictions": predictions,
        }

    def _load_trained_model(
        self,
        model_id: str,
        version: int,
        config: ModelConfig,
    ):
        """
        Load a trained model from the MLflow model registry.

        Args:
            model_id: Model ID (MLflow registered model name).
            version: Model version to load.
            config: Model configuration.

        Returns:
            ModelInterface instance with loaded weights.

        Raises:
            ValueError: If architecture is unsupported or model cannot be loaded.
        """
        model_class = MODEL_REGISTRY.get(config.architecture)
        if not model_class:
            raise ValueError(f"Unsupported architecture: {config.architecture}")

        model_uri = f"models:/{model_id}/{version}"
        logger.info(f"Loading model from {model_uri}")

        try:
            loaded_pytorch_model = mlflow.pytorch.load_model(model_uri)
        except Exception as e:
            raise ValueError(
                f"Failed to load model artifact from {model_uri}: {str(e)}"
            )

        # Create wrapper with config
        if config.architecture == ArchitectureType.LSTM:
            model = model_class(
                input_fields=config.input_fields,
                output_fields=config.output_fields,
                window_duration_seconds=config.window_duration_seconds,
                lookback_steps=config.lookback_steps,
                forecast_steps=config.forecast_steps,
                hidden_size=config.hidden_size,
                num_layers=2,
            )
        else:
            model = model_class(
                input_fields=config.input_fields,
                output_fields=config.output_fields,
                window_duration_seconds=config.window_duration_seconds,
                lookback_steps=config.lookback_steps,
                forecast_steps=config.forecast_steps,
                hidden_size=config.hidden_size,
            )

        model.model = loaded_pytorch_model
        return model

    def _structure_predictions(
        self,
        raw_predictions: np.ndarray,
        output_fields: list[str],
        forecast_steps: int,
    ) -> list[ForecastStepPrediction]:
        """
        Convert raw model output into structured predictions.

        The model returns shape (1, forecast_steps * num_output_fields).
        This reshapes it into a list of ForecastStepPrediction objects.

        Args:
            raw_predictions: Raw model output, shape (1, forecast_steps * num_outputs).
            output_fields: List of output field names.
            forecast_steps: Number of forecast steps.

        Returns:
            List of ForecastStepPrediction with field-value mappings per step.
        """
        flat = raw_predictions[0]
        num_outputs = len(output_fields)
        expected_length = forecast_steps * num_outputs

        if len(flat) != expected_length:
            raise RuntimeError(
                f"Model output shape mismatch: got {len(flat)} values, "
                f"expected {expected_length} (forecast_steps={forecast_steps} "
                f"* output_fields={num_outputs})"
            )

        predictions = []
        for step_idx in range(forecast_steps):
            start = step_idx * num_outputs
            end = start + num_outputs
            step_values = flat[start:end]

            values_dict = {
                field: round(float(step_values[i]), 6)
                for i, field in enumerate(output_fields)
            }

            predictions.append(
                ForecastStepPrediction(step=step_idx + 1, values=values_dict)
            )

        return predictions
