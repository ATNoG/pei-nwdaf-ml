"""Configuration endpoint for API metadata"""
from fastapi import APIRouter
from src.config.inference_type import get_all_inference_types
from src.schemas.config import ConfigResponse, InferenceTypeConfig

router = APIRouter()


@router.get("", response_model=ConfigResponse)
async def get_config():
    """
    Get all available configurations.

    Returns supported inference types, their horizons, and available model types.

    Returns:
        ConfigResponse: Available inference types and supported models
    """
    inference_configs = get_all_inference_types()

    # Group horizons by inference type
    inference_types_list = []
    seen_types = set()

    for (type_name, _), config in inference_configs.items():
        if type_name not in seen_types:
            # Collect all horizons for this type
            horizons = [
                h for (t, h) in inference_configs.keys()
                if t == type_name
            ]
            inference_types_list.append(
                InferenceTypeConfig(
                    name=config.name,
                    description=config.description,
                    supported_horizons=sorted(horizons)
                )
            )
            seen_types.add(type_name)

    # Get supported model types from models directory
    from src.models import models
    supported_model_types = [model_cls.__name__.lower() for model_cls in models]

    return ConfigResponse(
        inference_types=inference_types_list,
        supported_model_types=supported_model_types
    )
