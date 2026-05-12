"""Schemas for inference request and response."""

from pydantic import BaseModel, Field

from src.schemas.performance import LocalExplanationResponse
from src.schemas.tags import Tags


class InferenceRequest(BaseModel):
    """Request schema for running inference"""
    output_field: str = Field(..., description="Name of the output field to predict")
    tags: Tags = Field(..., description="Tag filters for fetching data")
    model_id: str | None = Field(None, description="UUID of the trained model to use for prediction")
    explain: bool = Field(
        False,
        description="If true, compute local attributions alongside the prediction. "
                    "Significantly slower, as it loads training-time background from MLflow artifact.",
    )


class ForecastStepPrediction(BaseModel):
    """Prediction values for a single forecast step"""

    step: int = Field(..., ge=1, description="Forecast step number (1 = nearest future window)")
    window_start_time: int = Field(..., description="Prediction window start timestamp (epoch seconds)")
    window_end_time: int = Field(..., description="Prediction window end timestamp (epoch seconds)")
    values: dict[str, float] = Field(...,description="Predicted values keyed by output field name",)


class InferenceResult(BaseModel):
    """Structured prediction output"""

    model_id: str = Field(..., description="Model ID used for prediction")
    model_name: str = Field(..., description="Human-readable model name")
    model_version: int = Field(..., description="Model version used")
    architecture: str = Field(..., description="Model architecture")
    tags: Tags = Field(..., description="Tag filters used to fetch data")
    lookback_steps: int = Field(..., description="Number of historical windows used as input")
    forecast_steps: int = Field(..., description="Number of future windows predicted")
    window_duration_seconds: int = Field(..., description="Duration of each time window in seconds")
    input_data_start: int = Field(..., description="Start timestamp of first input window used (epoch seconds)")
    input_data_end: int = Field(..., description="End timestamp of last input window used (epoch seconds)")
    input_fields: list[str] = Field(..., description="Input fields used for prediction")
    output_fields: list[str] = Field(..., description="Output fields being predicted")
    predictions: list[ForecastStepPrediction] = Field(...,description="Predictions broken down by forecast step and output field")
    explanation: LocalExplanationResponse | None = Field(
        None,
        description="Local explanation. Populated only when explain=true in the request.",
    )
