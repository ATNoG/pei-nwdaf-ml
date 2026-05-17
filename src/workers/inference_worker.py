"""Isolated inference worker — exec + torch.load + predict with cap_drop=[ALL].

Receives arch_bytes + model_bytes from mlservice (no MinIO/MLflow access needed).
"""

import asyncio
import base64
import logging

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.schemas.model import ModelConfig
from src.services.architecture_service import ArchitectureService

# Pre-built ArchitectureService instance with no MinIO connection (only _make_namespace + load_from_bytes needed)
_arch_svc = ArchitectureService.__new__(ArchitectureService)

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Inference Worker", docs_url=None, redoc_url=None)


class PredictRequest(BaseModel):
    architecture_id: str
    arch_bytes: str    # base64-encoded Python source
    model_bytes: str   # base64-encoded .pth file
    config: dict
    X_bytes: str
    X_shape: list[int]


class PredictResponse(BaseModel):
    output_bytes: str
    output_shape: list[int]


def _load_and_predict(req: PredictRequest, config: ModelConfig) -> np.ndarray:
    import torch
    from io import BytesIO

    arch_code = base64.b64decode(req.arch_bytes)
    model_data = base64.b64decode(req.model_bytes)

    with _arch_svc.load_from_bytes(req.architecture_id, arch_code) as (cls, _):
        loaded = torch.load(BytesIO(model_data), map_location="cpu", weights_only=False)  # nosec B614

        model = cls(
            input_fields=config.input_fields,
            output_fields=config.output_fields,
            window_duration_seconds=config.window_duration_seconds,
            lookback_steps=config.lookback_steps,
            forecast_steps=config.forecast_steps,
            hidden_size=config.hidden_size,
        )
        model.model = loaded

        X = np.frombuffer(base64.b64decode(req.X_bytes), dtype=np.float32).reshape(req.X_shape)
        X = np.nan_to_num(X, nan=0.0)
        return model.predict(X)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        config = ModelConfig(**req.config)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid config: {e}")

    try:
        raw = await asyncio.to_thread(_load_and_predict, req, config)
        raw = np.ascontiguousarray(raw, dtype=np.float32)
        return PredictResponse(
            output_bytes=base64.b64encode(raw.tobytes()).decode(),
            output_shape=list(raw.shape),
        )
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8061)
