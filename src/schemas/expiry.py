"""Pydantic schemas for expiry policy and audit log."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ExpiryPolicy(BaseModel):
    base_ttl_days: int = Field(default=30, ge=1, description="Days before an untrained model expires")
    tolerance_per_use_days: int = Field(default=1, ge=0, description="Extra days per recorded inference use")
    max_tolerance_days: int = Field(default=90, ge=0, description="Cap on tolerance days regardless of use count")
    sweep_interval_seconds: int = Field(default=3600, ge=60, description="How often the expiry sweep runs")


class ExpiryPolicyUpdate(BaseModel):
    """All fields optional for PATCH semantics."""

    base_ttl_days: Optional[int] = Field(default=None, ge=1)
    tolerance_per_use_days: Optional[int] = Field(default=None, ge=0)
    max_tolerance_days: Optional[int] = Field(default=None, ge=0)
    sweep_interval_seconds: Optional[int] = Field(default=None, ge=60)


class AuditEntry(BaseModel):
    model_id: str
    model_name: str
    expired_at: datetime
    last_used_at: Optional[datetime]
    effective_ttl_days: int
    policy_snapshot: dict

    model_config = {"from_attributes": True}
