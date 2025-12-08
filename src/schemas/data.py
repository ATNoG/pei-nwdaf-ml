from pydantic import BaseModel
from typing import Dict, Any, Optional


class DataQueryRequest(BaseModel):
    """Request model for querying data from Data Storage"""
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    data_type: str = "timeseries"
    filters: Optional[Dict[str, Any]] = None


class DataStorageRequest(BaseModel):
    """Generic request model for Data Storage API"""
    endpoint: str
    method: str = "GET"
    params: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    timeout: int = 30
