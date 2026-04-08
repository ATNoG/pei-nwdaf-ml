"""Service for resource usage and model resource defaults."""

import logging

from sqlalchemy.orm import Session

from src.schemas.resources import ModelResourceDefaults, ResourceUsageEntry

logger = logging.getLogger(__name__)


def _fmt_cpu(raw: str | None) -> str | None:
    """Convert K8s CPU quantity (e.g. '4192721079n', '2') to human-readable cores string."""
    if not raw:
        return None
    if raw.endswith("n"):
        cores = int(raw[:-1]) / 1e9
    elif raw.endswith("m"):
        cores = int(raw[:-1]) / 1000
    else:
        cores = float(raw)
    return f"{cores:.2f} cores"


def _fmt_memory(raw: str | None) -> str | None:
    """Convert K8s memory quantity (e.g. '371608Ki', '2Gi') to human-readable MB/GB string."""
    if not raw:
        return None
    units = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "K": 1000, "M": 1000**2, "G": 1000**3}
    for suffix, factor in units.items():
        if raw.endswith(suffix):
            mb = int(raw[: -len(suffix)]) * factor / 1024**2
            return f"{mb:.0f} MB" if mb < 1024 else f"{mb / 1024:.2f} GB"
    return f"{int(raw) / 1024**2:.0f} MB"


def find_model_config(model_id: str, db: Session):
    """Return (config, job_type) searching both forecast and anomaly tables."""
    from src.db.anomaly_model_config import AnomalyModelConfigDB
    from src.db.model_config import ModelConfigDB

    cfg = db.query(ModelConfigDB).filter(ModelConfigDB.model_id == model_id).first()
    if cfg:
        return cfg, "forecast"
    cfg = (
        db.query(AnomalyModelConfigDB)
        .filter(AnomalyModelConfigDB.model_id == model_id)
        .first()
    )
    if cfg:
        return cfg, "anomaly"
    return None, None


def get_pod_metrics(namespace: str) -> dict[str, tuple[str | None, str | None]]:
    """
    Query metrics-server for all pods in namespace.
    Returns dict of pod_name -> (cpu_usage, memory_usage), empty on failure.
    """
    try:
        from kubernetes import client as k8s_client

        # Ensure KubeClient has applied its configuration (sets the default ApiClient)
        from src.services.kube_training import get_kube_training_service
        get_kube_training_service()

        api = k8s_client.CustomObjectsApi()
        result = api.list_namespaced_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            namespace=namespace,
            plural="pods",
        )
        metrics: dict[str, tuple[str | None, str | None]] = {}
        for item in result.get("items", []):
            pod_name = item["metadata"]["name"]
            containers = item.get("containers", [])
            if containers:
                usage = containers[0].get("usage", {})
                metrics[pod_name] = (usage.get("cpu"), usage.get("memory"))
        return metrics
    except Exception as e:
        logger.warning("Failed to fetch pod metrics: %s", e)
        return {}


def get_job_alloc(
    kube, job_id: str, namespace: str
) -> tuple[str | None, str | None, str | None, str | None]:
    """Return (cpu_request, memory_request, cpu_limit, memory_limit) from K8s job spec."""
    if not kube:
        return None, None, None, None
    try:
        k8s_job = kube._kube_client.get_job(f"ml-train-{job_id}", namespace)
        if not k8s_job:
            return None, None, None, None
        res = k8s_job.spec.template.spec.containers[0].resources
        requests = res.requests or {}
        limits = res.limits or {}
        return (
            requests.get("cpu"),
            requests.get("memory"),
            limits.get("cpu"),
            limits.get("memory"),
        )
    except Exception:
        return None, None, None, None


def get_job_usage(
    job_id: str, pod_metrics: dict[str, tuple[str | None, str | None]]
) -> tuple[str | None, str | None]:
    """Match a job_id to a pod and return its live (cpu_usage, memory_usage)."""
    prefix = f"ml-train-{job_id}"
    for pod_name, (cpu, mem) in pod_metrics.items():
        if pod_name.startswith(prefix):
            return cpu, mem
    return None, None


def list_active_job_usage(db: Session) -> list[ResourceUsageEntry]:
    """Return resource allocation and live usage for all running/queued training jobs."""
    from src.core.config import settings
    from src.db.anomaly_training_job import AnomalyTrainingJobDB
    from src.db.training_job import TrainingJobDB
    from src.services.kube_training import get_kube_training_service

    kube = get_kube_training_service()
    active = ("running", "queued")
    namespace = settings.TRAIN_KUBE_NAMESPACE
    pod_metrics = get_pod_metrics(namespace)

    entries: list[ResourceUsageEntry] = []

    for job in db.query(TrainingJobDB).filter(TrainingJobDB.status.in_(active)).all():
        cpu_req, mem_req, cpu_lim, mem_lim = get_job_alloc(kube, job.job_id, namespace)
        cpu_use, mem_use = get_job_usage(job.job_id, pod_metrics)
        entries.append(ResourceUsageEntry(
            job_id=job.job_id, model_id=job.model_id, job_type="forecast",
            status=job.status,
            cpu_request=_fmt_cpu(cpu_req), memory_request=_fmt_memory(mem_req),
            cpu_limit=_fmt_cpu(cpu_lim), memory_limit=_fmt_memory(mem_lim),
            cpu_usage=_fmt_cpu(cpu_use), memory_usage=_fmt_memory(mem_use),
        ))

    for job in (
        db.query(AnomalyTrainingJobDB)
        .filter(AnomalyTrainingJobDB.status.in_(active))
        .all()
    ):
        cpu_req, mem_req, cpu_lim, mem_lim = get_job_alloc(kube, job.job_id, namespace)
        cpu_use, mem_use = get_job_usage(job.job_id, pod_metrics)
        entries.append(ResourceUsageEntry(
            job_id=job.job_id, model_id=job.model_id, job_type="anomaly",
            status=job.status,
            cpu_request=_fmt_cpu(cpu_req), memory_request=_fmt_memory(mem_req),
            cpu_limit=_fmt_cpu(cpu_lim), memory_limit=_fmt_memory(mem_lim),
            cpu_usage=_fmt_cpu(cpu_use), memory_usage=_fmt_memory(mem_use),
        ))

    return entries


def set_model_defaults(model_id: str, cpu: str, memory: str, db: Session) -> ModelResourceDefaults:
    cfg, _ = find_model_config(model_id, db)
    if not cfg:
        raise ValueError(f"Model '{model_id}' not found")
    cfg.default_cpu = cpu
    cfg.default_memory = memory
    db.commit()
    return ModelResourceDefaults(model_id=cfg.model_id, model_name=cfg.name, cpu=cpu, memory=memory)


def get_model_defaults(model_id: str, db: Session) -> ModelResourceDefaults | None:
    cfg, _ = find_model_config(model_id, db)
    if not cfg:
        raise ValueError(f"Model '{model_id}' not found")
    if cfg.default_cpu and cfg.default_memory:
        return ModelResourceDefaults(model_id=cfg.model_id, model_name=cfg.name, cpu=cfg.default_cpu, memory=cfg.default_memory)
    return None
