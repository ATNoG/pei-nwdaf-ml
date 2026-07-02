import asyncio
import base64
import logging
import os
import tempfile

import httpx
import numpy as np

from src.core.config import settings

logger = logging.getLogger(__name__)

# Per-event-loop AsyncClient cache. See data_storage_client.py for full
# rationale: a singleton httpx.AsyncClient pins asyncio primitives to the
# loop where it was created, so reusing it from a worker thread's
# asyncio.run() loop raises "Event loop is closed" / "bound to a different
# event loop".
_async_clients: dict[int, httpx.AsyncClient] = {}
_sync_client: httpx.Client | None = None

# Cache: model_uri -> (arch_bytes, model_bytes)
_bytes_cache: dict[str, tuple[bytes, bytes]] = {}
_MAX_BYTES_CACHE = 20


def _get_async_client() -> httpx.AsyncClient:
    loop = asyncio.get_event_loop()
    key = id(loop)
    client = _async_clients.get(key)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
        _async_clients[key] = client
    return client


def _get_sync_client() -> httpx.Client:
    global _sync_client
    if _sync_client is None or _sync_client.is_closed:
        _sync_client = httpx.Client(
            timeout=60.0,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _sync_client


def _fetch_artifact_bytes(model_uri: str) -> bytes:
    """Download model artifact bytes from MLflow/MinIO without torch.load."""
    import mlflow
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=tmpdir,
        )
        for root, _, files in os.walk(local_path):
            for f in files:
                if f.endswith(".pth"):
                    with open(os.path.join(root, f), "rb") as fp:
                        return fp.read()
    raise ValueError(f"No .pth file found in artifact {model_uri}")


def _get_bytes(architecture_id: str, model_uri: str) -> tuple[bytes, bytes]:
    global _bytes_cache
    if model_uri not in _bytes_cache:
        if len(_bytes_cache) >= _MAX_BYTES_CACHE:
            oldest = next(iter(_bytes_cache))
            del _bytes_cache[oldest]
        from src.services.architecture_service import ArchitectureService
        arch_bytes = ArchitectureService().get_raw_bytes(architecture_id)
        model_bytes = _fetch_artifact_bytes(model_uri)
        _bytes_cache[model_uri] = (arch_bytes, model_bytes)
        logger.info("Cached bytes for %s", model_uri)
    return _bytes_cache[model_uri]


def _payload(architecture_id: str, model_uri: str, config, X: np.ndarray) -> dict:
    arch_bytes, model_bytes = _get_bytes(architecture_id, model_uri)
    arr = np.ascontiguousarray(X, dtype=np.float32)
    return {
        "architecture_id": architecture_id,
        "arch_bytes": base64.b64encode(arch_bytes).decode(),
        "model_bytes": base64.b64encode(model_bytes).decode(),
        "config": config.model_dump(),
        "X_bytes": base64.b64encode(arr.tobytes()).decode(),
        "X_shape": list(arr.shape),
    }


def _payload_encrypted(architecture_id: str, model_uri: str, config, pages: list[bytes]) -> dict:
    arch_bytes, model_bytes = _get_bytes(architecture_id, model_uri)
    return {
        "architecture_id": architecture_id,
        "arch_bytes": base64.b64encode(arch_bytes).decode(),
        "model_bytes": base64.b64encode(model_bytes).decode(),
        "config": config.model_dump(),
        "encrypted_pages": [base64.b64encode(p).decode() for p in pages],
    }


def sync_safe_predict(architecture: str, model_uri: str, config, X: np.ndarray) -> np.ndarray:
    """Sync version — use inside sync callbacks (e.g. KernelSHAP predict_fn)."""
    client = _get_sync_client()
    try:
        r = client.post(
            f"{settings.INFERENCE_WORKER_URL}/predict",
            json=_payload(architecture, model_uri, config, X),
        )
        r.raise_for_status()
        body = r.json()
        return np.frombuffer(base64.b64decode(body["output_bytes"]), dtype=np.float32).reshape(body["output_shape"])
    except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as e:
        logger.error("Inference worker unreachable: %s", e)
        raise RuntimeError("Inference service unavailable. Try again later.")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Prediction failed: {e.response.json().get('detail', str(e))}")


async def safe_predict(
    architecture: str,
    model_uri: str,
    config,
    data: list[bytes] | np.ndarray,
) -> tuple[np.ndarray, list[dict] | None]:
    """Send prediction request to isolated inference worker.

    data: list[bytes] → encrypted path (worker decrypts with model private key)
    data: np.ndarray  → plaintext path (KernelSHAP / unencrypted inference)

    Returns (predictions, used_windows). used_windows is None for the numpy path.
    """
    if isinstance(data, list):
        payload = _payload_encrypted(architecture, model_uri, config, data)
    else:
        payload = _payload(architecture, model_uri, config, data)

    client = _get_async_client()
    try:
        r = await client.post(
            f"{settings.INFERENCE_WORKER_URL}/predict",
            json=payload,
        )
        r.raise_for_status()
        body = r.json()
        predictions = np.frombuffer(base64.b64decode(body["output_bytes"]), dtype=np.float32).reshape(body["output_shape"])
        return predictions, body.get("used_windows")
    except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as e:
        logger.error("Inference worker unreachable: %s", e)
        raise RuntimeError("Inference service unavailable. Try again later.")
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e))
        raise RuntimeError(f"Prediction failed: {detail}")
