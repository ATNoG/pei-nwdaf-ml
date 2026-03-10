"""Pydantic schemas for field discovery."""

from pydantic import BaseModel, Field


class FieldStatus(BaseModel):
    """Field name with model availability status."""

    name: str = Field(..., description="Field name")
    has_models: bool = Field(..., description="Whether any trained model predicts this field")


class FieldsResponse(BaseModel):
    """Response containing available field names."""

    fields: list[str] = Field(..., description="List of available field names")


class FieldsDetailResponse(BaseModel):
    """Response containing fields with model availability status."""

    fields: list[FieldStatus] = Field(..., description="List of fields with model status")
