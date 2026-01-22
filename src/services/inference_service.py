from typing_extensions import Type
import numpy as np
import logging
from typing import List, Dict, Any

from src.models import get_trainer_class
from src.config.inference_type import get_inference_config
from src.config.model_config import ModelConfig
from src.schemas.inference import PredictionHorizon
from src.utils.features import extract_features

logger = logging.getLogger(__name__)


class InferenceService:
    """Generate predictions for a cell using stored data"""

    def __init__(self, ml_interface):
        self.ml_interface = ml_interface
        self._model_cache: Dict[str, Any] = {}

    async def predict_cell_analytics(
        self,
        analytics_type: str,
        cell_index: int,
        horizon: int,
        model_type: str | None = None,
    ):
        # 1. Resolve config
        config = get_inference_config((analytics_type, horizon))
        if not config:
            raise ValueError(f"No config found for {analytics_type} with horizon {horizon}")

        if model_type is None:
            model_type = config.default_model
            if not model_type:
                raise ValueError("No default model configured")

        # 2. Load model (from MLflow or cache)
        try:
            trainer_class = get_trainer_class(model_type)
        except ValueError:
            raise ValueError(f"Model type not found: {model_type}")

        # Get sequence length from default config
        default_config = ModelConfig.default()
        sequence_length = default_config.sequence.sequence_length

        model = self._load_model(
            analytics_type=analytics_type,
            horizon=horizon,
            model_type=model_type,
        )

        if model is None:
            raise RuntimeError(f"Failed to load model {model_type}")

        # 3. Fetch latest windows


        windows = await self.ml_interface.fetch_latest_cell_data(
            endpoint=config.storage_endpoint,
            cell_index=cell_index,
            window_duration_seconds=config.window_duration_seconds,
            num_windows=sequence_length,
        )

        if not windows or len(windows) < sequence_length:
            raise ValueError(
                f"Not enough data: need {sequence_length}, got {len(windows) if windows else 0}"
            )

        model_name = config.get_model_name(model_type)
        feature_mean,feature_std = self._load_normalization(model_name)
        if feature_mean is None or feature_std is None:
            raise RuntimeError(f"Normalization artifacts missing for model {model_name}")

        # 4. Prepare data
        X, features_list, last_window = self._prepare_data(
            windows,
            analytics_type,
            feature_mean,
            feature_std
        )
        interval_str = self._horizon_to_iso8601(horizon)
        last_window_end = last_window.get("window_end_time", 0)

        # 5. Predict
        prediction = model.predict(X)

        return PredictionHorizon(
            used_model=model_type,
            cell_index=cell_index,
            interval=interval_str,
            predicted_value=float(prediction),
            confidence=1,
            data=features_list,
            target_start_time=last_window_end,
            target_end_time=last_window_end + horizon,
        )

    def _prepare_data(
        self,
        windows_data: List[Dict[str, Any]],
        analytics_type: str,
        feature_mean,
        feature_std
    ):
        """
        Returns:
            X: np.ndarray [1, seq_len, num_features]
            features_list: extracted features per window
            last_window: last raw window
        """

        features_list = [extract_features(w,analytics_type) for w in windows_data]

        if not features_list or not features_list[0]:
            raise ValueError("No valid features extracted")

        feature_keys = sorted(features_list[0].keys())

        sequence = []
        for f in features_list:
            sequence.append([f.get(k, 0.0) for k in feature_keys])

        X = np.array(sequence, dtype=np.float32)
        X = np.nan_to_num(X)
        X = X[np.newaxis, :, :]

        # feature_mean and feature_std already have keepdims shape from training
        # For 3D: (1, 1, features), for 2D: (features,) or (1, features)
        # Use directly without adding extra dimensions
        X = (X - feature_mean) / (feature_std + 1e-8)

        return X, features_list, windows_data[-1]

    def _get_model_name(self, inference_type: str, horizon: int, model_type: str) -> str:
        """construct model name for an inference type"""
        key = (inference_type, horizon)
        config = get_inference_config(key)
        if not config:
            raise ValueError(f"config not found: {inference_type} with horizon {horizon}s")
        return config.get_model_name(model_type)

    def _load_model(self, analytics_type: str, horizon: int, model_type: str) -> Any:
        """load model for an inference type"""
        try:
            model_name = self._get_model_name(analytics_type, horizon, model_type)
        except ValueError as e:
            logger.error(str(e))
            return None

        # check cache first
        if model_name in self._model_cache:
            logger.debug(f"Using cached model: {model_name}")
            return self._model_cache[model_name]

        try:
            # try to load from production stage
            model = self.ml_interface.get_model_by_name(model_name, stage='Production')

            if not model:
                # try latest version if production doesn't exist
                model = self.ml_interface.get_model_by_name(model_name, stage=None)

            if model:
                self._model_cache[model_name] = model
                logger.info(f"Loaded model for analytics type {analytics_type} (horizon={horizon}s): {model_name}")
                return model
            else:
                logger.warning(f"model not found for {analytics_type} (horizon={horizon}s): {model_name}")
                return None

        except Exception as e:
            logger.error(f"Error loading model for analytics type {analytics_type}: {e}")
            return None

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

    def _load_normalization(self, model_name: str):
        """
        Download mean/std normalization artifacts from MLflow for a given model.
        Returns:
            mean: np.ndarray
            std: np.ndarray
        """
        import tempfile

        try:
            # Download artifacts to a temp directory
            with tempfile.TemporaryDirectory() as tmpdir:
                mean_path = self.ml_interface.download_model_artifact(
                    model_name=model_name,
                    artifact_path="normalization/{}_mean.npy".format(model_name),
                    dst_path=tmpdir
                )
                std_path = self.ml_interface.download_model_artifact(
                    model_name=model_name,
                    artifact_path="normalization/{}_std.npy".format(model_name),
                    dst_path=tmpdir
                )

                mean = np.load(mean_path)
                std = np.load(std_path)
                return mean, std
        except Exception as e:
            logger.warning(f"Failed to load normalization artifacts for {model_name}: {e}")
            return None, None
