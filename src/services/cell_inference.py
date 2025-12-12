"""
Cell Inference Service

Business logic for cell analytics predictions.
Handles data fetching, model selection, and prediction generation.
"""
import logging
import numpy as np

from src.inference.inference import InferenceMaker
from src.config.inference_type import get_inference_config
from src.models import models_dict
from src.schemas.inference import PredictionHorizon

logger = logging.getLogger(__name__)


class CellInferenceService:
    """Service for handling cell analytics predictions"""

    def __init__(self, ml_interface):
        """
        Initialize the service.

        Args:
            ml_interface: MLInterface instance for data access and model loading
        """
        self.ml_interface = ml_interface
        self.inference_maker = InferenceMaker(ml_interface)

    async def predict_cell_analytics(
        self,
        analytics_type: str,
        cell_id: int,
        horizon: int,
        model_type: str|None = None
    ) -> PredictionHorizon:
        """
        Generate prediction for a cell.

        Args:
            analytics_type: Type of analytics (e.g., "latency")
            cell_id: Cell identifier
            horizon: Prediction horizon in seconds
            model_type: Model to use (None = use default from config)

        Returns:
            PredictionHorizon: Prediction result

        Raises:
            ValueError: If config not found or invalid parameters
            RuntimeError: If model loading or prediction fails
        """
        logger.info(f"Analytics request for cell {cell_id}, horizon={horizon}s, model_type={model_type}")

        # Get configuration
        config = get_inference_config((analytics_type, horizon))
        if config is None:
            raise ValueError(f"No config found for {analytics_type} with horizon {horizon}s")

        # Resolve model type
        if model_type is None:
            model_type = config.default_model
            if model_type is None:
                raise ValueError(f"No default model configured for {analytics_type} with horizon {horizon}s")

        # Get model class
        model_class = models_dict.get(model_type.lower())
        if model_class is None:
            raise ValueError(f"Model type not found: {model_type}")

        # Fetch data
        windows_data = await self._fetch_windows(
            config=config,
            cell_id=cell_id,
            num_windows=model_class.SEQUENCE_LENGTH
        )

        # Validate data
        if len(windows_data) < model_class.SEQUENCE_LENGTH:
            raise ValueError(
                f"Not enough data to use model [{model_type}]. "
                f"Need {model_class.SEQUENCE_LENGTH} windows, got {len(windows_data)}"
            )

        # Load model
        model = self.inference_maker._load_inference_type_model(analytics_type, horizon, model_type)
        if model is None or not hasattr(model, "predict"):
            raise RuntimeError(f"Failed to load model for {analytics_type} (horizon={horizon}s) with type {model_type}")

        # Prepare data and predict
        prepared_data, features_list, last_window = self._prepare_data(
            windows_data=windows_data,
            analytics_type=analytics_type,
            model_class=model_class
        )

        result = self._predict(model, prepared_data)

        # Calculate confidence
        confidence = self._calculate_confidence(horizon)

        # Calculate prediction window
        interval_str = self._horizon_to_iso8601(horizon)
        last_window_end = last_window.get("window_end_time", 0)

        return PredictionHorizon(
            used_model=model_type,
            cell_index=cell_id,
            interval=interval_str,
            predicted_value=float(result),
            confidence=round(confidence, 2),
            data=features_list,
            target_start_time=last_window_end,
            target_end_time=last_window_end + horizon,
        )

    async def _fetch_windows(self, config, cell_id: int, num_windows: int) -> list:
        """Fetch data windows from storage"""
        windows_data = await self.ml_interface.fetch_latest_cell_data(
            endpoint=config.storage_endpoint,
            cell_id=cell_id,
            window_duration_seconds=config.window_duration_seconds,
            num_windows=num_windows
        )

        if windows_data is None:
            raise ValueError(f"No data found for cell {cell_id}")

        # Ensure we have a list
        if isinstance(windows_data, dict):
            windows_data = [windows_data]

        return windows_data

    def _prepare_data(self, windows_data: list, analytics_type: str, model_class):
        """
        Prepare data for prediction.

        Returns:
            tuple: (prepared_data, features_list, last_window)
        """
        def extract_features(window_dict):
            return {
                k: v for k, v in window_dict.items()
                if k not in {
                    "window_start_time", "window_end_time",
                    "window_duration_seconds", "cell_index",
                    "network", "sample_count"
                }
                and not k.startswith(analytics_type + '_')
                and v is not None
            }

        features_list = [extract_features(w) for w in windows_data]


        # Handle both single window and sequence cases
        if model_class.SEQUENCE_LENGTH == 1:
            # Single window model
            prepared_data = self.inference_maker._prepare_data_for_prediction(features_list[0])
        else:
            # Sequence model

            if not features_list or not features_list[0]:
                raise ValueError("No valid features found in data")

            # Prepare each window and stack them
            prepared_list = [self.inference_maker._prepare_data_for_prediction(f) for f in features_list]
            prepared_data = np.array(prepared_list)

        last_window = windows_data[-1]

        if not features_list[0]:
            raise ValueError("No valid features extracted")

        # Ensure data is float32 for PyTorch models
        if isinstance(prepared_data, np.ndarray):
            prepared_data = prepared_data.astype(np.float32)
        elif hasattr(prepared_data, 'astype'):  # pandas DataFrame
            prepared_data = prepared_data.astype(np.float32)

        return prepared_data, features_list,last_window

    def _predict(self, model, prepared_data):
        """Make prediction and extract scalar result"""
        result = model.predict(prepared_data)
        result = self.inference_maker._convert_result_for_serialization(result)

        # Extract scalar value from nested lists/arrays
        while isinstance(result, (list, np.ndarray)) and len(result) > 0:
            result = result[0]

        return result

    def _calculate_confidence(self, horizon: int) -> float:
        """
        Calculate confidence score.

        TODO: Implement proper uncertainty quantification
        Currently uses simple heuristic.
        """
        base_confidence = 0.85
        horizon_factor = min(1.0, 60 / horizon)
        return base_confidence * horizon_factor

    def _horizon_to_iso8601(self, horizon: int) -> str:
        """Convert horizon in seconds to ISO 8601 duration"""
        if horizon < 60:
            return f"PT{horizon}S"
        elif horizon < 3600:
            minutes = horizon // 60
            return f"PT{minutes}M"
        elif horizon < 86400:
            hours = horizon // 3600
            return f"PT{hours}H"
        elif horizon < 604800:
            days = horizon // 86400
            return f"P{days}D"
        else:
            weeks = horizon // 604800
            return f"P{weeks}W"
