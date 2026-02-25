"""NWDAF ML Service - FastAPI application entrypoint."""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI

from src.core.config import settings
from src.db.database import init_db
from src.routers import router

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


def _build_performance_service(db):
    """
    Construct a PerformanceService outside of FastAPI's DI system.

    Used by the background monitoring loop which has no request context.
    Mirrors the manual construction in get_performance_service() in the router.
    """
    from mlflow import MlflowClient
    from src.services.config_service import MLConfigService
    from src.services.mlflow_service import MLflowService
    from src.services.performance_service import PerformanceService
    from src.services.data_storage_client import DataStorageClient

    client = MlflowClient()
    config_svc = MLConfigService(db)
    mlflow_svc = MLflowService(client, config_svc)
    return PerformanceService(mlflow_svc, config_svc, DataStorageClient(), db)


async def _has_new_data(data_storage_client, last_ts: int) -> bool:
    """
    Return True if any data windows have arrived since last_ts.

    Checks a small sample of cells to avoid a full scan on every cycle.
    """
    try:
        cells = await data_storage_client.get_known_cells()
    except Exception as e:
        logger.warning(f"Auto-monitor: failed to fetch known cells: {e}")
        return False

    now = int(time.time())
    for cell in cells[:3]:
        try:
            data = await data_storage_client.fetch_cell_data(cell, last_ts, now, 60)
            if data:
                return True
        except Exception:
            continue
    return False


async def _monitoring_loop() -> None:
    """
    Background task that periodically re-scores the best model for every
    monitored output field and logs a warning when degradation is detected.

    Controlled by MONITORING_TRIGGER_MODE:
      "time"  — run on every interval regardless of new data
      "data"  — run only when new data has arrived since last cycle
      "both"  — run on interval only if new data has arrived
    """
    from src.db.database import SessionLocal
    from src.services.data_storage_client import DataStorageClient

    logger.info(
        "Auto-monitor started (mode=%s, interval=%ds, factor=%.2f)",
        settings.MONITORING_TRIGGER_MODE,
        settings.MONITORING_INTERVAL_SECONDS,
        settings.MONITORING_DEGRADATION_FACTOR,
    )

    last_check_ts = int(time.time())

    while True:
        await asyncio.sleep(settings.MONITORING_INTERVAL_SECONDS)

        # Data-income gate — skip cycle if no new windows have arrived
        if settings.MONITORING_TRIGGER_MODE in ("data", "both"):
            data_client = DataStorageClient()
            if not await _has_new_data(data_client, last_check_ts):
                logger.debug("Auto-monitor: no new data since last check, skipping cycle")
                last_check_ts = int(time.time())
                continue

        last_check_ts = int(time.time())

        db = SessionLocal()
        try:
            svc = _build_performance_service(db)
            fields = svc.get_monitored_fields()

            if not fields:
                logger.debug("Auto-monitor: no monitored fields found, skipping cycle")
                continue

            logger.info("Auto-monitor: checking %d field(s): %s", len(fields), fields)

            for field in fields:
                try:
                    result = await svc.monitor_best_model(field, trigger="auto_monitor")
                    if result.score is None:
                        continue
                    baseline = svc.get_baseline_score(field)
                    if baseline and svc._score_is_worse(
                        result.score,
                        baseline,
                        result.metric,
                        settings.MONITORING_DEGRADATION_FACTOR,
                    ):
                        logger.warning(
                            "Auto-monitor: model '%s' %s=%.4f for field '%s' "
                            "has degraded beyond %.0f%% of baseline %.4f",
                            result.model_id,
                            result.metric,
                            result.score,
                            field,
                            settings.MONITORING_DEGRADATION_FACTOR * 100,
                            baseline,
                        )
                    else:
                        logger.info(
                            "Auto-monitor: field '%s' — model '%s' %s=%.4f (baseline=%.4f)",
                            field,
                            result.model_id,
                            result.metric,
                            result.score,
                            baseline if baseline is not None else float("nan"),
                        )
                except Exception as e:
                    logger.error("Auto-monitor: error monitoring field '%s': %s", field, e)
        finally:
            db.close()

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

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    logger.info("Initializing database tables...")
    init_db()
    logger.info("Database initialized successfully")

    monitor_task = None
    if settings.MONITORING_ENABLED:
        monitor_task = asyncio.create_task(_monitoring_loop())
        logger.info("Auto-monitor task started")

    yield

    if monitor_task:
        monitor_task.cancel()
        logger.info("Auto-monitor task stopped")

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
