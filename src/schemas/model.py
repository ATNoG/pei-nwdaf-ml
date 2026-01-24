from pydantic import BaseModel
from typing import Any
from src.schemas.model_config import ModelConfigSchema


class ModelDetailedInfo(BaseModel):
    """Detailed model information including architecture and config"""
    name: str
    model_type: str
    analytics_type: str
    horizon: int
    version: str|None = None
    stage: str|None = None
    config: dict[str, Any]|None = None
    last_training_time: float|None = None
    training_loss: float|None = None
    run_id: str|None = None


class ModelInfo(BaseModel):
    """Model information from MLFlow registry"""
    name: str
    creation_timestamp: int|None = None
    last_updated_timestamp: int|None = None
    description: str|None = None
    latest_versions: list[dict[str, Any]]|None = None


class ModelCreationRequest(BaseModel):
    """Request to create a new model instance"""
    analytics_type: str
    horizon: int
    model_type: str
    name: str
    config: ModelConfigSchema|None=None

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
