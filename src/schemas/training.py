from pydantic import BaseModel
from typing import Optional


class ModelTrainingRequest(BaseModel):
    """Request to train a model for an analytics type"""
    analytics_type: str
    horizon: int
    model_type: str = "xgboost"

    class Config:
        json_schema_extra = {
            "example": {
                "analytics_type": "latency",
                "horizon": 60,
                "model_type": "xgboost"
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
