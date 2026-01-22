from pydantic import BaseModel
from typing import Optional

from src.schemas.model_config import ModelConfigSchema


class ModelCreationRequest(BaseModel):
    """Request to create a new model instance"""
    analytics_type: str
    horizon: int
    model_type: str
    name: Optional[str] = None  # Custom model name, auto-generated if not provided
    config: Optional[ModelConfigSchema] = None

    class Config:
        json_schema_extra = {
            "example": {
                "analytics_type": "latency",
                "horizon": 60,
                "model_type": "ann",
                "name": "latency_ann_v2",
                "config": {
                    "architecture": {
                        "hidden_size": 64
                    },
                    "sequence": {
                        "sequence_length": 5
                    }
                }
            }
        }


class ModelCreationResponse(BaseModel):
    """Response when model instance is created"""
    status: str
    model_name: str
    message: str


class ModelDeletionResponse(BaseModel):
    """Response when model instance is deleted"""
    status: str
    model_name: str
    message: str
