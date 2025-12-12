"""Configuration schemas for API responses"""
from pydantic import BaseModel
from typing import List, Optional


class InferenceTypeConfig(BaseModel):
    """Configuration details for an inference type"""
    name: str
    description: Optional[str] = None
    supported_horizons: List[int]


class ConfigResponse(BaseModel):
    """Response containing all available configurations"""
    inference_types: List[InferenceTypeConfig]
    supported_model_types: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "inference_types": [
                    {
                        "name": "latency",
                        "description": "Network latency prediction and analysis",
                        "supported_horizons": [60, 300, 3600]
                    }
                ],
                "supported_model_types": ["xgboost", "randomforest"]
            }
        }
