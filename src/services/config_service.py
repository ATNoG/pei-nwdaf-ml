"""
Configuration Service

Business logic for retrieving system configuration.
Handles aggregation of inference types and model types.

"""
import logging
from typing import Any

from src.config.inference_type import get_all_inference_types, get_inference_config
from src.models import get_available_model_types
from src.schemas.config import InferenceTypeConfig, ConfigResponse

logger = logging.getLogger(__name__)


class ConfigService:
    """Service for handling configuration queries"""

    @staticmethod
    def get_system_config() -> ConfigResponse:
        """
        Get all available system configurations.

        Returns:
            ConfigResponse: Available inference types and supported models
        """
        inference_configs = get_all_inference_types()

        # Build inference types list
        inference_types_list = []
        for _, config in inference_configs.items():
            inference_types_list.append(
                InferenceTypeConfig(
                    name=config.name,
                    horizon=config.window_duration_seconds,
                    default_model=config.default_model or "",
                    description=config.description,
                )
            )

        # Get supported model types
        supported_model_types = get_available_model_types()

        return ConfigResponse(
            inference_types=inference_types_list,
            supported_model_types=supported_model_types
        )

    @staticmethod
    def update_default_model(
        analytics_type: str,
        horizon: int,
        model_type: str,
        ml_interface = None
    ) -> dict[str, Any]:
        """
        Update the default model for an inference type configuration.

        Args:
            analytics_type: Analytics type (e.g., 'latency')
            horizon: Prediction horizon in seconds
            model_type: New model type to set as default
            ml_interface: MLInterface instance for persisting to MLflow

        Returns:
            dict: Success message with updated configuration

        Raises:
            ValueError: If config not found or model type invalid
        """
        # Validate config exists
        key = (analytics_type, horizon)
        config = get_inference_config(key)

        if not config:
            raise ValueError(
                f"No configuration found for analytics_type={analytics_type} "
                f"with horizon={horizon}s"
            )

        # Validate model type exists
        available_types = get_available_model_types()
        if model_type.lower() not in available_types:
            raise ValueError(
                f"Invalid model type: {model_type}. "
                f"Supported types: {available_types}"
            )

        # Get model name for the new default                                                                        
        new_model_name = config.get_model_name(model_type)                                                          
        tag_key = "default_for"                                                                                     
        tag_value = f"{analytics_type}_{horizon}"                                                                   
                                                                                                                    
        # Persist to MLflow if ml_interface is provided                                                             
        if ml_interface:                                                                                            
            # Clear old default tag from previous default model (if any)                                            
            old_default_models = ml_interface.get_models_with_tag(tag_key, tag_value)                               
            for old_model in old_default_models:                                                                    
                ml_interface.clear_model_tag(old_model, tag_key)                                                    
                                                                                                                    
            # Set new default tag                                                                                   
            ml_interface.set_model_tag(new_model_name, tag_key, tag_value)                                          
                                                                                                                    
        # Update in-memory config                                                                                   
        old_default = config.default_model                                                                          
        config.set_default_model(model_type)                                                                        
                                                                                                                    
        logger.info(                                                                                                
            f"Updated default model for {analytics_type} (horizon={horizon}s): "                                    
            f"{old_default} -> {model_type}"                                                                        
        )                                                                                                           
                                                                                                                    
        return {                                                                                                    
            "status": "success",                                                                                    
            "message": f"Default model updated for {analytics_type} (horizon={horizon}s)",                          
            "analytics_type": analytics_type,                                                                       
            "horizon": horizon,                                                                                     
            "old_default_model": old_default,                                                                       
            "new_default_model": model_type                                                                         
        } 
                                                                                                
def load_defaults_from_mlflow(ml_interface) -> None:                                                            
    """                                                                                                         
    Load default model settings from MLflow tags on startup.                                                    
                                                                                                                
    Queries MLflow for models with 'default_for' tags and populates                                             
    the in-memory InferenceConfig defaults.                                                                     
                                                                                                                
    Args:                                                                                                       
        ml_interface: MLInterface instance for querying MLflow                                                  
    """                                                                                                         
    if not ml_interface:                                                                                        
        logger.warning("No ml_interface provided - skipping MLflow defaults loading")                           
        return                                                                                                  
                                                                                                                
    inference_configs = get_all_inference_types()                                                               
                                                                                                                
    for (analytics_type, horizon), config in inference_configs.items():                                         
        tag_value = f"{analytics_type}_{horizon}"                                                               
        models = ml_interface.get_models_with_tag("default_for", tag_value)                                     
                                                                                                                
        if models:                                                                                              
            # Extract model_type from model name (e.g., "latency_lstm_60" -> "lstm")                            
            model_name = models[0]                                                                              
            parts = model_name.split('_')                                                                       
            if len(parts) >= 3:                                                                                 
                model_type = parts[-2]  # Second to last part is model type                                     
                config.set_default_model(model_type)                                                            
                logger.info(f"Loaded default model from MLflow: {analytics_type}_{horizon} -> {model_type}")  