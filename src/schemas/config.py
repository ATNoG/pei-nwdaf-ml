"""Configuration schemas for API responses"""
from pydantic import BaseModel
from typing import List, Optional


class InferenceTypeConfig(BaseModel):
    """Configuration details for an inference type"""
    name: str
    horizon:int
    default_model:str
    description: Optional[str] = None


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
                        "horizon":60,
                        "default_model": "lstm",
                        "description": "Network latency prediction and analysis",
                    }
                ],
                "supported_model_types": ["lstm"]
            }
        }
