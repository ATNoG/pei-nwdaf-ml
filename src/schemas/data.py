from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class DataQueryRequest(BaseModel):
    """Request model for querying data from Data Storage"""
    start_time: str = Field(..., description="Window start time (ISO format)")
    end_time: str = Field(..., description="Window end time (ISO format)")
    data_type: str = Field(default="latency", description="Type of data to fetch (currently only 'latency' is supported)")
    filters: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Additional filters (e.g., cell_index, offset, limit)"
    )


class LatencyQueryRequest(BaseModel):
    """Request model specifically for querying latency data from Data Storage /api/v1/processed/latency/ endpoint"""
    start_time: str = Field(..., description="Window start time (ISO format)")
    end_time: str = Field(..., description="Window end time (ISO format)")
    cell_index: int = Field(..., description="Cell index (required)")
    offset: int = Field(default=0, ge=0, description="Number of records to skip")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of records to return")


class DataStorageRequest(BaseModel):
    """Generic request model for Data Storage API"""
    endpoint: str
    method: str = "GET"
    params: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    timeout: int = 30
