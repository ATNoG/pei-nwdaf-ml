"""Isolated inference worker — exec + torch.load + predict with cap_drop=[ALL]."""

import asyncio
import base64
import logging
import os

import mlflow
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.core.config import settings
from src.schemas.model import ModelConfig
from src.services.model_loader import load_trained_model

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Inference Worker", docs_url=None, redoc_url=None)

_model_cache: dict[str, object] = {}
_MAX_CACHE = 20


class PredictRequest(BaseModel):
    architecture: str
    model_uri: str
    config: dict
    X_bytes: str
    X_shape: list[int]


class PredictResponse(BaseModel):
    output_bytes: str
    output_shape: list[int]


@app.on_event("startup")
async def startup():
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.MLFLOW_S3_ENDPOINT_URL
    os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.AWS_SECRET_ACCESS_KEY
    os.environ["AWS_DEFAULT_REGION"] = settings.AWS_DEFAULT_REGION
    logger.info("Inference worker ready")


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        config = ModelConfig(**req.config)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid config: {e}")

    if req.model_uri not in _model_cache:
        if len(_model_cache) >= _MAX_CACHE:
            oldest = next(iter(_model_cache))
            del _model_cache[oldest]
        try:
            model = await asyncio.to_thread(
                load_trained_model, req.architecture, req.model_uri, config
            )
            _model_cache[req.model_uri] = model
            logger.info("Cached model %s", req.model_uri)
        except Exception as e:
            logger.error("Failed to load %s: %s", req.model_uri, e)
            raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

    model = _model_cache[req.model_uri]

    try:
        X = np.frombuffer(base64.b64decode(req.X_bytes), dtype=np.float32).reshape(req.X_shape)
        X = np.nan_to_num(X, nan=0.0)
        raw = await asyncio.to_thread(model.predict, X)
        raw = np.ascontiguousarray(raw, dtype=np.float32)
        return PredictResponse(
            output_bytes=base64.b64encode(raw.tobytes()).decode(),
            output_shape=list(raw.shape),
        )
    except Exception as e:
        _model_cache.pop(req.model_uri, None)
        logger.error("Prediction failed for %s: %s", req.model_uri, e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8061)
