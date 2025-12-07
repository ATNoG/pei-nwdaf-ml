from src.general.modelI import ModelI
from typing import Optional, Any, Dict
import logging

logger = logging.getLogger("Inference")

class InferenceMaker:
    """
    Cooks an inference
    Loads models from MLFlow (communicated with via MLInterface) for producing predictions
    """
    def __init__(self, ml_interface, selected_model_id: Optional[str] = None) -> None:
        """
        Initialize the inference maker

        Args:
            ml_interface: MLInterface instance for communicating with MLFlow
            selected_model_id: Specific model to use, or None for auto-selection
        """
        self.ml_interface = ml_interface
        self._auto_mode: bool = False   # whether inference should auto-select best model or use provided model
        self._failed_retrieves: int = 0 # times that couldn't retrieve model from repository
        self._current_model = None      # cached loaded model
        self._current_model_id: Optional[str] = None

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
        Produces an inference using the selected model

        Returns:
            Inference result or None if failed
        """
        try:
            self.ml_interface.update_component_status('inference', 'inferring')

            model = self._load_model()
            if model is None:
                logger.warning("Couldn't load model for inference")
                self.ml_interface.update_component_status('inference', 'error', error='model_load_failed')
                return None

            # Perform inference
            logger.debug("Performing inference...")

            if isinstance(model, ModelI):
                result = model.infer(**kwargs)
            else:
                # MLFlow pyfunc models use predict()
                if hasattr(model, 'predict'):
                    # Expect 'data' key in kwargs for MLFlow models
                    data = kwargs.get('data')
                    if data is None:
                        logger.error("No 'data' provided for inference")
                        return None
                    result = model.predict(data)
                else:
                    logger.error("Model does not have infer() or predict() method")
                    return None

            self.ml_interface.log_inference_metrics({
                'inference_count': 1,
                'model_used': self._current_model_id or 'unknown'
            })

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

    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model"""
        return {
            'model_id': self._current_model_id,
            'auto_mode': self._auto_mode,
            'model_loaded': self._current_model is not None,
            'failed_retrieves': self._failed_retrieves
        }

    def clear_cache(self) -> None:
        """Clear the cached model, forcing reload on next inference"""
        self._current_model = None
        logger.info("Model cache cleared")
        self.ml_interface.update_component_status(
            'inference',
            'idle',
            current_model=self._current_model_id,
            cache_cleared=True
        )
