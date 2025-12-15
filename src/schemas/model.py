from pydantic import BaseModel


class ModelCreationRequest(BaseModel):
    """Request to create a new model instance"""
    analytics_type: str
    horizon: int
    model_type: str

    class Config:
        json_schema_extra = {
            "example": {
                "analytics_type": "latency",
                "horizon": 60,
                "model_type": "ann"
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
