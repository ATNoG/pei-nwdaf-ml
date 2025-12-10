from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Union
from enum import Enum


class PredictionInterval(str, Enum):
    """ISO 8601 duration for prediction intervals"""
    PT5M = "PT5M"      # 5 minutes
    PT1H = "PT1H"      # 1 hour
    P1D = "P1D"        # 1 day
    P1W = "P1W"        # 1 week


# Legacy alias
PredictionHorizon = PredictionInterval


class InferenceRequest(BaseModel):
    """Request model for ML inference"""
    data: Union[Dict[str, Any], List[Dict[str, Any]]]
    cell_index: Optional[Union[str, float]] = None
    cell_indices: Optional[List[Union[str, float]]] = None
    model_type: Optional[str] = None  # e.g., 'xgboost', 'randomforest'
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_stage: Optional[str] = "Production"
    publish_result: bool = False


class TrainingRequest(BaseModel):
    """Request model for ML training"""
    model_name: str
    config: Dict[str, Any]


class InferenceResponse(BaseModel):
    """Response model for inference results"""
    status: str
    model_used: Optional[Union[str, Dict[str, str]]]  # single model or cell->model mapping
    predictions: Any
    published_to_kafka: bool = False
    cell_index: Optional[Union[str, float]] = None


class ModelInfo(BaseModel):
    """Model information from MLFlow registry"""
    name: str
    creation_timestamp: Optional[int] = None
    last_updated_timestamp: Optional[int] = None
    description: Optional[str] = None
    latest_versions: Optional[List[Dict[str, Any]]] = None


class ModelSelectionRequest(BaseModel):
    """Request to manually select a model for inference"""
    model_name: str
    version: Optional[str] = None
    stage: Optional[str] = "Production"


class AutoModeRequest(BaseModel):
    """Request to toggle auto-select mode"""
    auto_mode: bool


class AnalyticsRequest(BaseModel):
    """NWDAF analytics prediction request"""
    analytics_type: str
    cell_id: int

    class Config:
        json_schema_extra = {
            "example": {
                "analytics_type": "latency",
                "cell_id": 26379009
            }
        }


class PredictionHorizon(BaseModel):
    """Single prediction for a time horizon"""
    interval: str
    predicted_value: float
    confidence: float
    data:dict



class AnalyticsTypePrediction(BaseModel):
    """Predictions for all time horizons of an analytics type"""
    analytics_type: str
    predictions: List[PredictionHorizon]


class CellAnalyticsResponse(BaseModel):
    """All analytics predictions for a cell"""
    cell_id: int
    timestamp: float
    analytics: List[AnalyticsTypePrediction]

    class Config:
        json_schema_extra = {
            "example": {
                "cell_id": 26379009,
                "timestamp": 1733828700.0,
                "analytics": [
                    {
                        "analytics_type": "latency",
                        "predictions": [
                            {"interval": "PT1M", "predicted_value": 45.2, "confidence": 0.85},
                            {"interval": "PT1H", "predicted_value": 48.1, "confidence": 0.75},
                            {"interval": "P1D", "predicted_value": 50.3, "confidence": 0.65}
                        ]
                    }
                ]
            }
        }


# Legacy single-prediction response
class AnalyticsResponse(BaseModel):
    """NWDAF analytics prediction response"""
    analytics_type: str
    cell_id: int
    prediction_interval: str
    predicted_value: float
    confidence: float
    timestamp: float
    valid_from: Optional[int] = None
    valid_until: Optional[int] = None

    class Config:
        json_schema_extra = {
            "example": {
                "analytics_type": "latency",
                "cell_id": 26379009,
                "prediction_interval": "PT1M",
                "predicted_value": 45.2,
                "confidence": 0.92,
                "timestamp": 1733828700.0,
                "valid_from": 1733828400,
                "valid_until": 1733828700
            }
        }


# Legacy aliases for backward compatibility
CellInferenceRequest = AnalyticsRequest
CellInferenceResponse = AnalyticsResponse
