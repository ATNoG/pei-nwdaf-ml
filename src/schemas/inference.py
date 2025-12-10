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
