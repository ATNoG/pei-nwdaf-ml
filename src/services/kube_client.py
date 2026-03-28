import base64
import logging
import tempfile

from kubernetes import client
from kubernetes.client.rest import ApiException

from src.core.config import settings
from src.schemas.kube import JobResources, JobSpec

logger = logging.getLogger(__name__)


class KubeClient:
    def __init__(self):
        self._configure()
        self._batch = client.BatchV1Api()

    def create_job(self, spec: JobSpec) -> str:
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=spec.name, namespace=spec.namespace, labels=spec.labels
            ),
            spec=client.V1JobSpec(
                backoff_limit=spec.backoff_limit,
                ttl_seconds_after_finished=spec.ttl_seconds,
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(labels=spec.labels),
                    spec=client.V1PodSpec(
                        restart_policy="Never",
                        containers=[
                            client.V1Container(
                                name="worker",
                                image=spec.image,
                                image_pull_policy="IfNotPresent",
                                env=[
                                    client.V1EnvVar(name=k, value=v)
                                    for k, v in spec.env.items()
                                ],
                                resources=self._build_resources(spec.resources),
                            )
                        ],
                    ),
                ),
            ),
        )
        self._batch.create_namespaced_job(namespace=spec.namespace, body=job)
        logger.info(f"Created K8s Job '{spec.name}' in namespace '{spec.namespace}'")
        return spec.name

    def get_job(self, name: str, namespace: str) -> client.V1Job | None:
        try:
            return self._batch.read_namespaced_job(name=name, namespace=namespace)
        except ApiException as e:
            if e.status == 404:
                return None
            raise

    def delete_job(self, name: str, namespace: str) -> None:
        try:
            self._batch.delete_namespaced_job(
                name=name,
                namespace=namespace,
                body=client.V1DeleteOptions(propagation_policy="Foreground"),
            )
            logger.info(f"Deleted K8s Job '{name}'")
        except ApiException as e:
            if e.status != 404:
                raise

    def _configure(self) -> None:
        configuration = client.Configuration()
        configuration.host = settings.TRAIN_KUBE_HOST
        configuration.api_key = {"authorization": f"Bearer {settings.TRAIN_KUBE_TOKEN}"}

        if settings.TRAIN_KUBE_CA_CERT:
            ca_bytes = base64.b64decode(settings.TRAIN_KUBE_CA_CERT)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".crt")
            tmp.write(ca_bytes)
            tmp.flush()
            tmp.close()
            configuration.ssl_ca_cert = tmp.name
        else:
            configuration.verify_ssl = False

        client.Configuration.set_default(configuration)

    @staticmethod
    def _build_resources(
        resources: JobResources | None,
    ) -> client.V1ResourceRequirements | None:
        if resources is None:
            return None
        quota = {"cpu": resources.cpu, "memory": resources.memory}
        return client.V1ResourceRequirements(requests=quota, limits=quota)
