"""NWDAF ML Service - FastAPI application entrypoint."""

import logging
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI

from src.core.config import settings

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting NWDAF ML Service...")
    logger.info(f"MLflow tracking URI: {settings.MLFLOW_TRACKING_URI}")
    logger.info(f"Data Storage API: {settings.DATA_STORAGE_API_URL}")

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    yield

    logger.info("Shutting down NWDAF ML Service...")


# Create FastAPI app
app = FastAPI(
    title="NWDAF ML Service",
    description="Model instantiation, training, and inference service",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# Include API routers
from src.routers import router

app.include_router(router)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "service": "NWDAF ML Service",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    """Health check endpoint."""
    # TODO: Add actual health checks (MLflow, data-storage connectivity)
    return {
        "status": "healthy",
        "mlflow_uri": settings.MLFLOW_TRACKING_URI,
        "data_storage_api": settings.DATA_STORAGE_API_URL,
    }


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {settings.API_HOST}:{settings.API_PORT}")

    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )
