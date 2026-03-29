from dataclasses import dataclass, field

from pydantic import BaseModel


class JobResources(BaseModel):
    cpu: str
    memory: str


@dataclass
class JobSpec:
    name: str
    namespace: str
    image: str
    env: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    resources: JobResources | None = None
    backoff_limit: int = 0
    ttl_seconds: int = 3600
    active_deadline_seconds: int = 7200
