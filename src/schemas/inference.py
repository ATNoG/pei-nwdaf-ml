from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Union


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


class CellInferenceRequest(BaseModel):
    """Request model for cell-specific inference that auto-fetches latest data from storage"""
    cell_index: int
    model_type: Optional[str] = "xgboost"  # e.g., 'xgboost', 'randomforest'

    class Config:
        json_schema_extra = {
            "example": {
                "cell_index": 12898855,
                "model_type": "xgboost"
            }
        }


class CellInferenceResponse(BaseModel):
    """Response model for cell-specific inference"""
    cell_index: str
    predictions: Any
    model_used: str
    timestamp: float
    data_window_start: Optional[str] = None
    data_window_end: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "cell_index": "12898855",
                "predictions": [0.85, 0.12, 0.03],
                "model_used": "cell_12898855_xgboost",
                "timestamp": 1702310400.0,
                "data_window_start": "2024-12-10T10:00:00Z",
                "data_window_end": "2024-12-10T10:05:00Z"
            }
        }
