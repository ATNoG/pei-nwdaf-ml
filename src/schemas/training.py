from pydantic import BaseModel
from typing import Optional

from src.schemas.model_config import ModelConfigSchema


class ModelTrainingRequest(BaseModel):
    """Request to train a model by name"""
    model_name: str
    config: Optional[ModelConfigSchema] = None

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "latency_lstm_60",
                "config": {
                    "training": {
                        "learning_rate": 0.001,
                        "optimizer": "adam",
                        "max_epochs": 100
                    },
                    "architecture": {
                        "hidden_size": 64,
                        "dropout": 0.2
                    }
                }
            }
        }


class ModelTrainingStartResponse(BaseModel):
    """Response when training is started"""
    status: str
    model_name: str
    message: str


class ModelTrainingInfo(BaseModel):
    """Training information for a model"""
    model_name: str
    model_version: Optional[str] = None
    last_training_time: Optional[float] = None
    training_loss: Optional[float] = None
    samples_used: Optional[int] = None
    features_used: Optional[int] = None
    run_id: Optional[str] = None
