from pydantic import BaseModel


class ResourceUsageEntry(BaseModel):
    job_id: str | None = None
    model_id: str
    model_name: str | None = None
    job_type: str
    status: str
    cpu_request: str | None = None
    memory_request: str | None = None
    cpu_limit: str | None = None
    memory_limit: str | None = None
    cpu_usage: str | None = None
    memory_usage: str | None = None


class ResourcesUsageResponse(BaseModel):
    jobs: list[ResourceUsageEntry]


class ModelResourceDefaults(BaseModel):
    model_id: str | None = None
    model_name: str | None = None
    cpu: str
    memory: str
