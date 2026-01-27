"""
Pydantic schemas for model configuration options API response.
"""

from pydantic import BaseModel
from typing import List, Dict


class ModelConfigOptions(BaseModel):
    """Response schema for available model configuration options"""
    analytics_types: List[str]
    horizons: List[int]
    model_types: List[str]
    optimizers: List[str]
    loss_functions: List[str]
    activations: List[str]
    defaults: Dict[str, any]
