"""NWDAF ML Service - FastAPI application entrypoint."""

import logging
import os
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI

from src.core.config import settings
from src.db.database import init_db
import src.db.expiry_policy  # noqa: F401 — register table with SQLAlchemy Base
import src.db.expiry_audit  # noqa: F401 — register table with SQLAlchemy Base
from src.routers import router
from src.services.expiry_service import ExpiryService, ExpiryScheduler

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
    logger.info(f"Database URL: {settings.DATABASE_URL}")

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.MLFLOW_S3_ENDPOINT_URL
    os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.AWS_SECRET_ACCESS_KEY
    os.environ["AWS_DEFAULT_REGION"] = settings.AWS_DEFAULT_REGION

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    # Initialize database tables
    logger.info("Initializing database tables...")
    init_db()
    logger.info("Database initialized successfully")

    # Start model expiry scheduler
    expiry_service = ExpiryService()
    app.state.expiry_service = expiry_service
    expiry_scheduler = ExpiryScheduler(expiry_service)
    expiry_scheduler.start()
    app.state.expiry_scheduler = expiry_scheduler

    yield

    logger.info("Shutting down NWDAF ML Service...")
    expiry_scheduler = getattr(app.state, "expiry_scheduler", None)
    if expiry_scheduler:
        expiry_scheduler.stop()


# Create FastAPI app
app = FastAPI(
    title="NWDAF ML Service",
    description="Model instantiation, training, and inference service",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


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
        "status": "healthy"
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
