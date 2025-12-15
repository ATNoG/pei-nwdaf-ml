from src.models.model_interface import ModelInterface
from src.config.inference_type import get_inference_config
from typing import Optional, Any, Dict, Union, List
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("Inference")

class InferenceMaker:
    """
    Cooks an inference
    Loads models from MLFlow (communicated with via MLInterface) for producing predictions
    Supports cell-specific models and batch inference
    """

    def __init__(self, ml_interface, selected_model_id: Optional[str] = None, selected_cell_id: Optional[float] = None) -> None:
        """
        Initialize the inference maker

        Args:
            ml_interface: MLInterface instance for communicating with MLFlow
            selected_model_id: Specific model to use, or None for auto-selection
            selected_cell_id: Specific cell to work with (optional)
        """
        self.ml_interface = ml_interface
        self._auto_mode: bool = False    # whether inference should auto-select best model or use provided model
        self._failed_retrieves: int = 0  # times that couldn't retrieve model from repository
        self._current_model = None       # cached loaded model
        self._current_model_id: Optional[str] = None
        self._selected_cell_id = selected_cell_id
        self._model_cache: Dict[str, Any] = {}  # cache for multiple cell models

        self.ml_interface.update_component_status('inference', 'initializing')

        if selected_model_id is None:
            # auto select best model from repository
            self._auto_mode = True
            logger.info("Inference initialized in auto-select mode")
            self.ml_interface.update_component_status('inference', 'idle', mode='auto')
        else:
            # Validate and set specific model
            if self._set_model(selected_model_id):
                logger.info(f"Inference initialized with model: {selected_model_id}")
                self.ml_interface.update_component_status(
                    'inference',
                    'idle',
                    mode='manual',
                    current_model=selected_model_id
                )
            else:
                logger.warning(f"Failed to validate model {selected_model_id}")
                self._auto_mode = True
                self.ml_interface.update_component_status('inference', 'idle', mode='auto')

    def _fetch_best_model(self) -> Optional[str]:
        """
        Returns the model uri string from model registry or none if not found
        """
        try:
            best_model_info = self.ml_interface.get_best_model(metric='accuracy')  # Using dummy metrics for now... inb4 they're fine lol

            if best_model_info:
                model_uri = best_model_info.get('model_uri')
                logger.info(f"Best model found: {best_model_info.get('name')} "
                           f"v{best_model_info.get('version')} "
                           f"(accuracy: {best_model_info.get('metric_value')})")
                return model_uri
            else:
                logger.warning("No best model found in registry")
                return None

        except Exception as e:
            logger.error(f"Error fetching best model: {e}")
            self._failed_retrieves += 1
            return None

    def toggle_auto_select(self, value: Optional[bool] = None) -> None:
        """Toggles auto mode"""
        self._auto_mode = value if value is not None else not self._auto_mode
        mode = 'auto' if self._auto_mode else 'manual'
        logger.info(f"Auto-select mode: {mode}")

        self.ml_interface.update_component_status(
            'inference',
            'idle',
            mode=mode,
            current_model=self._current_model_id
        )

    def _get_cell_model_name(self, cell_index: Union[str, float], model_type: Optional[str] = None) -> str:
        """construct model name for a specific cell"""
        cell_str = str(cell_index).replace('.', '_')
        if model_type:
            return f"cell_{cell_str}_{model_type.lower()}"
        # default to xgboost if no type specified
        return f"cell_{cell_str}_xgboost"

    def _get_inference_type_model_name(self, inference_type: str, horizon: int, model_type: str) -> str:
        """construct model name for an inference type"""
        key = (inference_type, horizon)
        config = get_inference_config(key)
        if not config:
            raise ValueError(f"config not found: {inference_type} with horizon {horizon}s")
        return config.get_model_name(model_type)

    def _load_cell_model(self, cell_index: Union[str, float], model_type: Optional[str] = None) -> Any:
        """load model for a specific cell"""
        model_name = self._get_cell_model_name(cell_index, model_type)

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
                logger.info(f"Loaded model for cell {cell_index}: {model_name}")
                return model
            else:
                logger.warning(f"No model found for cell {cell_index}: {model_name}")
                return None

        except Exception as e:
            logger.error(f"Error loading model for cell {cell_index}: {e}")
            return None

    def _load_inference_type_model(self, inference_type: str, horizon: int, model_type: str) -> Any:
        """load model for an inference type"""
        try:
            model_name = self._get_inference_type_model_name(inference_type, horizon, model_type)
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
                logger.info(f"Loaded model for inference type {inference_type} (horizon={horizon}s): {model_name}")
                return model
            else:
                logger.warning(f"model not found for {inference_type} (horizon={horizon}s): {model_name}")
                return None

        except Exception as e:
            logger.error(f"Error loading model for inference type {inference_type}: {e}")
            return None

    def _load_model(self) -> Any:
        """Loads selected model from MLFlow repository via the interface"""
        try:
            if self._auto_mode:
                target_model_uri = self._fetch_best_model()
                if not target_model_uri:
                    logger.error("Cannot load model: no best model available")
                    self._failed_retrieves += 1
                    return None
            else:
                target_model_uri = self._current_model_id
                if not target_model_uri:
                    logger.error("Cannot load model: no model ID set")
                    return None

            # Check if we already have this model cached
            if self._current_model and self._current_model_id == target_model_uri:
                logger.debug(f"Using cached model: {target_model_uri}")
                return self._current_model

            # Load model through interface
            logger.info(f"Loading model from MLFlow: {target_model_uri}")
            self.ml_interface.update_component_status('inference', 'loading_model')

            model = self.ml_interface.load_model(target_model_uri)

            if model:
                self._current_model = model
                self._current_model_id = target_model_uri
                self._failed_retrieves = 0
                logger.info(f"Successfully loaded model: {target_model_uri}")

                self.ml_interface.update_component_status(
                    'inference',
                    'ready',
                    current_model=target_model_uri,
                    failed_retrieves=0
                )
                return model
            else:
                logger.error(f"Failed to load model: {target_model_uri}")
                self._failed_retrieves += 1
                self.ml_interface.update_component_status(
                    'inference',
                    'error',
                    current_model=None,
                    failed_retrieves=self._failed_retrieves
                )
                return None

        except Exception as e:
            logger.error(f"Exception while loading model: {e}")
            self._failed_retrieves += 1
            self.ml_interface.update_component_status(
                'inference',
                'error',
                error=str(e),
                failed_retrieves=self._failed_retrieves
            )
            return None

    def _set_model(self, model_id: str) -> bool:
        """Sets inference to use a model

        Args:
            model_id: ID/URI of the model (e.g., 'models:/MyModel/Production' or 'models:/MyModel/1')

        Returns:
            bool: Successfully validated and set model
        """
        try:
            # Check if model exists in registry via interface
            logger.info(f"Validating model: {model_id}")

            # Try to get model info from MLFlow
            # For now, we'll attempt to load it as validation
            test_load = self.ml_interface.load_model(model_id)

            if test_load:
                self._current_model_id = model_id
                self._current_model = test_load
                self._auto_mode = False
                logger.info(f"Model validated and set: {model_id}")

                self.ml_interface.update_component_status(
                    'inference',
                    'ready',
                    mode='manual',
                    current_model=model_id
                )
                return True
            else:
                logger.warning(f"Model validation failed: {model_id}")
                return False

        except Exception as e:
            logger.error(f"Error setting model {model_id}: {e}")
            return False

    def set_model_by_name(self, model_name: str, version: Optional[str] = None,
                         stage: Optional[str] = 'Production') -> bool:
        """
        Set model by name and optional version/stage

        Args:
            model_name: Name of the registered model
            version: Specific version number
            stage: Model stage ('Production', 'Staging', etc.)

        Returns:
            bool: Successfully set model?
        """
        try:
            model = self.ml_interface.get_model_by_name(model_name, version=version, stage=stage)

            if model:
                # Construct the URI for tracking
                if stage:
                    model_uri = f"models:/{model_name}/{stage}"
                elif version:
                    model_uri = f"models:/{model_name}/{version}"
                else:
                    model_uri = f"models:/{model_name}/latest"

                self._current_model = model
                self._current_model_id = model_uri
                self._auto_mode = False

                logger.info(f"Model set: {model_uri}")
                self.ml_interface.update_component_status(
                    'inference',
                    'ready',
                    mode='manual',
                    current_model=model_uri
                )
                return True
            else:
                logger.warning(f"Could not load model: {model_name}")
                return False

        except Exception as e:
            logger.error(f"Error setting model by name: {e}")
            return False

    def infer(self, **kwargs) -> Optional[Any]:
        """
        Produces inference using selected model or cell-specific models.

        Supports:
        - Single cell inference with cell_index
        - Batch inference with cell_indices
        - Traditional model-based inference

        Returns:
            Inference result or None if failed.
        """
        try:
            self.ml_interface.update_component_status('inference', 'inferring')

            # check for cell-specific inference
            cell_index = kwargs.get('cell_index')
            cell_indices = kwargs.get('cell_indices')
            model_type = kwargs.get('model_type')
            data = kwargs.get('data')

            # batch inference for multiple cells
            if cell_indices and isinstance(cell_indices, list):
                return self._infer_batch(cell_indices, data, model_type)

            # single cell inference
            if cell_index is not None:
                return self._infer_cell(cell_index, data, model_type)

            # traditional inference (legacy support)
            model = self._load_model()
            if model is None:
                logger.warning("Couldn't load model for inference")
                self.ml_interface.update_component_status('inference', 'error', error='model_load_failed')
                return None

            logger.debug("Performing inference...")

            if isinstance(model, ModelI):
                if data is None:
                    logger.error("No 'data' provided for inference")
                    return None
                result = model.predict(data)
            else:
                if hasattr(model, 'predict'):
                    if data is None:
                        logger.error("No 'data' provided for inference")
                        return None
                    result = model.predict(data)
                else:
                    logger.error("Model does not have infer() or predict() method")
                    return None

            self.ml_interface.log_inference_metrics({'inference_count': 1})

            # Log model_used as a tag or parameter instead of a metric
            if hasattr(self.ml_interface, "log_inference_tag"):
                self.ml_interface.log_inference_tag('model_used', self._current_model_id or 'unknown')
            elif hasattr(self.ml_interface, "log_inference_param"):
                self.ml_interface.log_inference_param('model_used', self._current_model_id or 'unknown')
            else:
                try:
                    import mlflow
                    mlflow.set_tag('model_used', self._current_model_id or 'unknown')
                except ImportError:
                    logger.warning("Could not log 'model_used' as tag: mlflow not available")

            logger.info("Inference completed successfully")
            self.ml_interface.update_component_status(
                'inference',
                'ready',
                current_model=self._current_model_id,
                last_inference='success'
            )

            return result

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            self.ml_interface.update_component_status(
                'inference',
                'error',
                error=str(e),
                last_inference='failed'
            )
            return None

    def _prepare_data_for_prediction(self, data: Any) -> Any:
        """Convert data to a format suitable for model.predict()"""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            # Convert dict to DataFrame (single row)
            return pd.DataFrame([data])
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                # List of dicts -> DataFrame
                return pd.DataFrame(data)
            else:
                # Assume it's already array-like
                return data
        else:
            # Return as-is and let the model handle it
            return data

    def _convert_result_for_serialization(self, result: Any) -> Any:
        """Convert prediction result to JSON-serializable format"""
        if isinstance(result, np.ndarray):
            return result.tolist()
        elif isinstance(result, (np.integer, np.floating)):
            return result.item()
        elif isinstance(result, pd.DataFrame):
            return result.to_dict('records')
        elif isinstance(result, pd.Series):
            return result.tolist()
        else:
            return result

    def _infer_cell(self, cell_index: Union[str, float], data: Any, model_type: Optional[str] = None) -> Optional[Any]:
        """perform inference for a single cell"""
        try:
            model = self._load_cell_model(cell_index, model_type)
            if model is None:
                logger.warning(f"No model available for cell {cell_index}")
                return None

            # Convert data to appropriate format for prediction
            prepared_data = self._prepare_data_for_prediction(data)

            if isinstance(model, ModelI):
                result = model.predict(prepared_data)
            else:
                if hasattr(model, 'predict'):
                    result = model.predict(prepared_data)
                else:
                    logger.error(f"Model for cell {cell_index} has no predict method")
                    return None

            # Convert result to JSON-serializable format
            result = self._convert_result_for_serialization(result)

            model_name = self._get_cell_model_name(cell_index, model_type)
            self.ml_interface.log_inference_metrics({
                'inference_count': 1,
                'cell_index': float(cell_index) if isinstance(cell_index, (int, float)) else hash(str(cell_index))
            })

            logger.info(f"Inference completed for cell {cell_index}")
            self.ml_interface.update_component_status(
                'inference',
                'ready',
                current_model=model_name,
                last_inference='success'
            )

            return result

        except Exception as e:
            logger.error(f"Error in cell inference for {cell_index}: {e}")
            return None

    def _infer_batch(self, cell_indices: List[Union[str, float]], data: Any, model_type: Optional[str] = None) -> Dict[str, Any]:
        """perform batch inference for multiple cells"""
        results = {}

        for cell_index in cell_indices:
            try:
                # if data is a list, match by index; if dict with cell keys, extract
                cell_data = data
                if isinstance(data, list) and len(data) == len(cell_indices):
                    idx = cell_indices.index(cell_index)
                    cell_data = data[idx]
                elif isinstance(data, dict) and str(cell_index) in data:
                    cell_data = data[str(cell_index)]

                result = self._infer_cell(cell_index, cell_data, model_type)
                results[str(cell_index)] = result

            except Exception as e:
                logger.error(f"Error in batch inference for cell {cell_index}: {e}")
                results[str(cell_index)] = None

        logger.info(f"Batch inference completed for {len(cell_indices)} cells")
        return results

    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model"""
        return {
            'model_id': self._current_model_id,
            'auto_mode': self._auto_mode,
            'model_loaded': self._current_model is not None,
            'failed_retrieves': self._failed_retrieves,
            'cached_models': list(self._model_cache.keys()),
            'selected_cell_id': self._selected_cell_id
        }

    def clear_cache(self) -> None:
        """Clear the cached model, forcing reload on next inference"""
        self._current_model = None
        self._model_cache.clear()
        logger.info("Model cache cleared")
        self.ml_interface.update_component_status(
            'inference',
            'idle',
            current_model=self._current_model_id,
            cache_cleared=True
        )
