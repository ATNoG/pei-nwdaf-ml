import base64
import logging

import httpx
import numpy as np

from src.core.config import settings

logger = logging.getLogger(__name__)

_async_client: httpx.AsyncClient | None = None
_sync_client: httpx.Client | None = None


def _get_async_client() -> httpx.AsyncClient:
    global _async_client
    if _async_client is None or _async_client.is_closed:
        _async_client = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _async_client


def _get_sync_client() -> httpx.Client:
    global _sync_client
    if _sync_client is None or _sync_client.is_closed:
        _sync_client = httpx.Client(
            timeout=60.0,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _sync_client


def _payload(architecture: str, model_uri: str, config, X: np.ndarray) -> dict:
    arr = np.ascontiguousarray(X, dtype=np.float32)
    return {
        "architecture": architecture,
        "model_uri": model_uri,
        "config": config.model_dump(),
        "X_bytes": base64.b64encode(arr.tobytes()).decode(),
        "X_shape": list(arr.shape),
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


async def safe_predict(architecture: str, model_uri: str, config, X: np.ndarray) -> np.ndarray:
    """Send prediction request to isolated inference worker container.

    Worker loads and caches the model by model_uri. All dangerous operations
    (exec, torch.load, predict) run in a container with cap_drop=[ALL].
    """
    client = _get_async_client()
    try:
        r = await client.post(
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
        detail = e.response.json().get("detail", str(e))
        raise RuntimeError(f"Prediction failed: {detail}")
