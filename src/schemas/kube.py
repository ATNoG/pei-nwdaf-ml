from dataclasses import dataclass, field


@dataclass
class JobResources:
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
