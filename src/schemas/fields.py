"""Pydantic schemas for field discovery."""

from pydantic import BaseModel, Field


class FieldsResponse(BaseModel):
    """Response containing available field names."""

    fields: list[str] = Field(..., description="List of available field names")
