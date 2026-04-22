from src.core.config import settings
from src.schemas.kube import JobResources, JobSpec
from src.services.kube_client import KubeClient


_instance: "KubeTrainingService | None" = None


def get_kube_training_service() -> "KubeTrainingService | None":
    global _instance
    if settings.TRAIN_USE_KUBE and _instance is None:
        _instance = KubeTrainingService(KubeClient())
    return _instance


class KubeTrainingService:
    def __init__(self, kube_client: KubeClient):
        self._kube_client = kube_client

    _DEFAULT_RESOURCES = JobResources(cpu="2", memory="2Gi")

    def to_kube(self, model_id: str, job_id: str, resources: JobResources | None, training_type: str):
        spec = JobSpec(
            name=f"ml-train-{model_id}",
            namespace=settings.TRAIN_KUBE_NAMESPACE,
            image=settings.TRAIN_KUBE_IMAGE,
            labels={"app": "ml-train", "model-id": model_id},
            env={
                "TRAINING_JOB_ID": job_id,
                "TRAINING_TYPE": training_type,
                "DATABASE_URL": settings.DATABASE_URL,
                "MLFLOW_TRACKING_URI": settings.MLFLOW_TRACKING_URI,
                "MLFLOW_S3_ENDPOINT_URL": settings.MLFLOW_S3_ENDPOINT_URL,
                "AWS_ACCESS_KEY_ID": settings.AWS_ACCESS_KEY_ID,
                "AWS_SECRET_ACCESS_KEY": settings.AWS_SECRET_ACCESS_KEY,
                "AWS_DEFAULT_REGION": settings.AWS_DEFAULT_REGION,
                "DATA_STORAGE_API_URL": settings.DATA_STORAGE_API_URL,
                "DATA_STORAGE_EXAMPLE_ENDPOINT": settings.DATA_STORAGE_EXAMPLE_ENDPOINT,
                "DATA_STORAGE_DATA_ENDPOINT": settings.DATA_STORAGE_DATA_ENDPOINT,
                "DATA_STORAGE_CELL_ENDPOINT": settings.DATA_STORAGE_CELL_ENDPOINT,
                "DATA_STORAGE_EXCLUDED_FIELDS": settings.DATA_STORAGE_EXCLUDED_FIELDS,
                "LOG_LEVEL": settings.LOG_LEVEL,
                "ENCRYPTION_ENABLED": str(settings.ENCRYPTION_ENABLED).lower(),
            },
            resources=resources or self._DEFAULT_RESOURCES,
        )
        self._kube_client.create_job(spec)

    def cancel(self, model_id: str):
        self._kube_client.delete_job(f"ml-train-{model_id}", settings.TRAIN_KUBE_NAMESPACE)

    def is_job_dead(self, model_id: str) -> bool:
        """Returns True if the K8s job has failed or no longer exists."""
        job = self._kube_client.get_job(f"ml-train-{model_id}", settings.TRAIN_KUBE_NAMESPACE)
        if job is None:
            return True
        conditions = job.status.conditions or []
        return any(c.type == "Failed" and c.status == "True" for c in conditions)
