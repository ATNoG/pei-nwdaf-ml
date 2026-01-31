"""Integration test fixtures with real services via testcontainers."""

import pytest
import time
import httpx
from testcontainers.postgres import PostgresContainer
from testcontainers.core.container import DockerContainer
from testcontainers.core.network import Network
from testcontainers.core.waiting_utils import wait_for_logs
from mlflow import MlflowClient


@pytest.fixture(scope="session")
def docker_network():
    """Shared Docker network for inter-container communication."""
    with Network() as network:
        yield network


@pytest.fixture(scope="session")
def postgres_container(docker_network):
    """PostgreSQL container for MLflow backend store."""
    postgres = PostgresContainer("postgres:15")
    postgres.with_network(docker_network)
    postgres.with_network_aliases("postgres")

    with postgres:
        yield postgres


@pytest.fixture(scope="session")
def minio_container(docker_network):
    """MinIO container for MLflow artifact store."""
    minio = DockerContainer("minio/minio:latest")
    minio.with_network(docker_network)
    minio.with_network_aliases("minio")
    minio.with_exposed_ports(9000)
    minio.with_env("MINIO_ROOT_USER", "minio")
    minio.with_env("MINIO_ROOT_PASSWORD", "minio123")
    minio.with_command("server /data")

    with minio:
        # Wait for MinIO to be ready
        wait_for_logs(minio, "MinIO Object Storage Server", timeout=30)
        time.sleep(2)  # Extra wait for API to be fully ready
        yield minio


@pytest.fixture(scope="session")
def mlflow_container(docker_network, postgres_container, minio_container):
    """MLflow tracking server container."""
    # Use container network aliases for inter-container communication
    postgres_url = f"postgresql://test:test@postgres:5432/test"

    mlflow = DockerContainer("ghcr.io/mlflow/mlflow:latest")
    mlflow.with_network(docker_network)
    mlflow.with_exposed_ports(5000)
    mlflow.with_env("MLFLOW_BACKEND_STORE_URI", postgres_url)
    mlflow.with_env("MLFLOW_DEFAULT_ARTIFACT_ROOT", "s3://mlflow")
    mlflow.with_env("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    mlflow.with_env("AWS_ACCESS_KEY_ID", "minio")
    mlflow.with_env("AWS_SECRET_ACCESS_KEY", "minio123")
    mlflow.with_command(
        [
            "mlflow",
            "server",
            "--backend-store-uri",
            postgres_url,
            "--default-artifact-root",
            "s3://mlflow",
            "--host",
            "0.0.0.0",
            "--port",
            "5000",
        ]
    )

    with mlflow:
        # Wait for MLflow to be ready by checking logs first
        wait_for_logs(mlflow, "Application startup complete", timeout=120)

        # Then verify HTTP endpoint works
        host = mlflow.get_container_host_ip()
        port = mlflow.get_exposed_port(5000)
        mlflow_url = f"http://{host}:{port}"

        # Poll MLflow health endpoint with all possible connection errors
        for i in range(30):  # 30 attempts = ~30 seconds
            try:
                response = httpx.get(f"{mlflow_url}/health", timeout=2.0)
                if response.status_code == 200:
                    time.sleep(1)  # Extra wait for stability
                    break
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ReadError):
                pass
            time.sleep(1)
        else:
            raise TimeoutError("MLflow health check did not pass in 30 seconds")

        yield mlflow


@pytest.fixture(scope="session")
def mlflow_tracking_uri(mlflow_container):
    """MLflow tracking URI for integration tests."""
    host = mlflow_container.get_container_host_ip()
    port = mlflow_container.get_exposed_port(5000)
    return f"http://{host}:{port}"


@pytest.fixture
def mlflow_client(mlflow_tracking_uri):
    """Real MLflow client connected to testcontainer."""
    import mlflow

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient(mlflow_tracking_uri)

    yield client

    # Cleanup: delete all registered models after each test
    try:
        for model in client.search_registered_models():
            client.delete_registered_model(model.name)
    except Exception:
        pass  # Ignore cleanup errors
