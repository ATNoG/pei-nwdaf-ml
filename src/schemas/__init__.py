"""
Pydantic schemas for the NWDAF ML component.

This package contains all Pydantic models used for request/response validation
throughout the application.
"""

from src.schemas.data import (
    DataQueryRequest,
    DataStorageRequest,
)

from src.schemas.inference import (
    InferenceRequest,
    TrainingRequest,
    InferenceResponse,
    ModelInfo,
    ModelSelectionRequest,
    AutoModeRequest,
)

from src.schemas.kafka import (
    KafkaMessage,
)

__all__ = [
    # Data schemas
    "DataQueryRequest",
    "DataStorageRequest",
    # Inference schemas
    "InferenceRequest",
    "TrainingRequest",
    "InferenceResponse",
    "ModelInfo",
    "ModelSelectionRequest",
    "AutoModeRequest",
    # Kafka schemas
    "KafkaMessage",
]