from pydantic import BaseModel
from typing import Optional


class ModelTrainingRequest(BaseModel):
    """Request to train a model by name using stored config"""
    model_name: str

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "latency_lstm_60"
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
