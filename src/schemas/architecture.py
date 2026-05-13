from datetime import datetime
from pydantic import BaseModel


class ArchitectureResponse(BaseModel):
    name: str
    uploaded_by: str
    uploaded_at: datetime


class ArchitectureUploadResponse(BaseModel):
    name: str
    uploaded_by: str


class ArchitectureHelpResponse(BaseModel):
    constraints: str
