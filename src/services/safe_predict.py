import logging

import httpx
import numpy as np

from src.core.config import settings

logger = logging.getLogger(__name__)


def _payload(architecture: str, model_uri: str, config, X: np.ndarray) -> dict:
    return {
        "architecture": architecture,
        "model_uri": model_uri,
        "config": config.model_dump(),
        "X": X.tolist(),
    }


def sync_safe_predict(architecture: str, model_uri: str, config, X: np.ndarray) -> np.ndarray:
    """Sync version — use inside sync callbacks (e.g. KernelSHAP predict_fn)."""
    with httpx.Client(timeout=60.0) as client:
        try:
            r = client.post(
                f"{settings.INFERENCE_WORKER_URL}/predict",
                json=_payload(architecture, model_uri, config, X),
            )
            r.raise_for_status()
            return np.array(r.json()["output"], dtype=np.float32)
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
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            r = await client.post(
                f"{settings.INFERENCE_WORKER_URL}/predict",
                json=_payload(architecture, model_uri, config, X),
            )
            r.raise_for_status()
            return np.array(r.json()["output"], dtype=np.float32)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as e:
            logger.error("Inference worker unreachable: %s", e)
            raise RuntimeError("Inference service unavailable. Try again later.")
        except httpx.HTTPStatusError as e:
            detail = e.response.json().get("detail", str(e))
            raise RuntimeError(f"Prediction failed: {detail}")
