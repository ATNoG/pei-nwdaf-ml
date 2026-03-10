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
from src.notification import AlertLevel, notification_center
from src.routers import router
from src.services.inference_pipeline import setup_inference_pipeline

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

# Debugging purposes only: suppress overly verbose logs from dependencies
for noisy in ("httpx", "httpcore", "botocore", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def _build_performance_service(db):
    """
    Construct a PerformanceService outside of FastAPI's DI system.

    Used by the background monitoring loop which has no request context.
    Mirrors the manual construction in get_performance_service() in the router.
    """
    from mlflow import MlflowClient

    from src.services.config_service import MLConfigService
    from src.services.data_storage_client import DataStorageClient
    from src.services.mlflow_service import MLflowService
    from src.services.performance_service import PerformanceService

    client = MlflowClient()
    config_svc = MLConfigService(db)
    mlflow_svc = MLflowService(client, config_svc)
    return PerformanceService(mlflow_svc, config_svc, DataStorageClient(), db)


def _build_training_service(db):
    """
    Construct a TrainingService outside of FastAPI's DI system.

    Used by the background monitoring loop to trigger retraining when
    performance degradation is detected.
    """
    from mlflow import MlflowClient

    from src.services.config_service import MLConfigService
    from src.services.data_storage_client import DataStorageClient
    from src.services.mlflow_service import MLflowService
    from src.services.training_service import TrainingService

    client = MlflowClient()
    config_svc = MLConfigService(db)
    mlflow_svc = MLflowService(client, config_svc)
    return TrainingService(mlflow_svc, DataStorageClient(), config_svc, db)


def _trigger_field_retraining(field, model_configs, training_svc, loop):
    """
    Queue retraining jobs for all model configs belonging to a degraded field.

    Uses the lookback_seconds from each model's last successful training job,
    falling back to 86400s if no completed job exists.

    Returns a list of queued job_ids (configs that failed to queue are skipped).
    """
    from src.db.training_job import TrainingJobDB
    from src.routers.v1.training_router import _run_training_sync, training_executor

    job_ids = []
    for cfg in model_configs:
        last_job = (
            training_svc.db.query(TrainingJobDB)
            .filter(
                TrainingJobDB.model_id == cfg.model_id,
                TrainingJobDB.status == "completed",
            )
            .order_by(TrainingJobDB.created_at.desc())
            .first()
        )
        lookback = last_job.lookback_seconds if last_job else 86400
        try:
            job_info = training_svc.create_training_job(cfg.model_id, lookback)
            loop.run_in_executor(
                training_executor, _run_training_sync, job_info["job_id"]
            )
            job_ids.append(job_info["job_id"])
            logger.info(
                "Auto-monitor: queued retraining job %s for model %s (field '%s')",
                job_info["job_id"],
                cfg.model_id,
                field,
            )
        except Exception as e:
            logger.error(
                "Auto-monitor: failed to queue retraining for model %s: %s",
                cfg.model_id,
                e,
            )
    return job_ids


async def _has_new_data(data_storage_client, last_ts: int) -> bool:
    """
    Return True if any data windows have arrived since last_ts.

    Checks a small sample of cells to avoid a full scan on every cycle.
    """
    try:
        cells = await data_storage_client.get_known_cells()
    except Exception as e:
        logger.warning("Auto-monitor: failed to fetch known cells: %s", e)
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
    monitored output field.

    Each field moves through a simple state machine:
      MONITORING  — score the best model; on degradation trigger retraining
      RETRAINING  — wait for all queued training jobs to finish
      EVALUATING  — re-elect a best model via evaluate_field()

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
    # Per-field state machine (persists across cycles, resets on restart)
    field_states: dict[str, str] = (
        {}
    )  # field → "monitoring" | "retraining" | "evaluating"
    field_jobs: dict[str, list[str]] = {}  # field → job_ids currently being tracked

    while True:
        await asyncio.sleep(settings.MONITORING_INTERVAL_SECONDS)

        # Data-income gate — skip cycle if no new windows have arrived
        if settings.MONITORING_TRIGGER_MODE in ("data", "both"):
            data_client = DataStorageClient()
            if not await _has_new_data(data_client, last_check_ts):
                logger.debug(
                    "Auto-monitor: no new data since last check, skipping cycle"
                )
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
                state = field_states.get(field, "monitoring")

                # RETRAINING: wait for all queued jobs to reach a terminal state
                if state == "retraining":
                    from src.db.training_job import TrainingJobDB

                    jobs = [
                        db.get(TrainingJobDB, jid) for jid in field_jobs.get(field, [])
                    ]
                    active = [
                        j for j in jobs if j and j.status in ("queued", "running")
                    ]

                    if active:
                        logger.info(
                            "----- [RETRAINING] '%s' — %d/%d job(s) still running.",
                            field,
                            len(active),
                            len(jobs),
                        )
                        continue  # check again next cycle

                    failed = [j for j in jobs if j and j.status == "failed"]
                    completed = [j for j in jobs if j and j.status == "completed"]

                    for j in failed:
                        logger.error(
                            "----- [RETRAINING] '%s' — job %s FAILED: %s",
                            field,
                            j.job_id,
                            j.error_message,
                        )

                    if completed:
                        field_states[field] = "evaluating"
                        logger.info(
                            "----- [RETRAINING] '%s' — all done (%d completed, %d failed) → moving to EVALUATING",
                            field,
                            len(completed),
                            len(failed),
                        )
                        await notification_center.notify(
                            f"Retraining completed for '{field}': "
                            f"{len(completed)} succeeded, {len(failed)} failed. "
                            f"Evaluating best model.",
                            AlertLevel.INFO,
                        )
                    else:
                        logger.error(
                            "----- [RETRAINING] '%s' — all jobs failed → returning to MONITORING",
                            field,
                        )
                        await notification_center.notify(
                            f"Retraining failed for '{field}': all jobs failed.",
                            AlertLevel.WARNING,
                        )
                        field_states[field] = "monitoring"
                        field_jobs.pop(field, None)

                # EVALUATING: re-elect a best model then return to monitoring
                elif state == "evaluating":
                    logger.info(
                        "----- [EVALUATING] '%s' — scoring all retrained models.", field
                    )
                    try:
                        best = svc.get_best_model(field)
                        metric = best.metric if best and best.metric else "rmse"
                        result = await svc.evaluate_field(field, metric)
                        best_score = next(
                            (m.score for m in result.models if m.is_best), float("nan")
                        )

                        logger.info(
                            "----- [EVALUATING] '%s' — new best: %s  %s=%.4f",
                            field,
                            result.best_model_id,
                            metric,
                            best_score,
                        )
                        prev_id = best.model_id if best else None
                        if result.best_model_id != prev_id:
                            await notification_center.notify(
                                f"Best model changed for '{field}': "
                                f"{prev_id} → {result.best_model_id} "
                                f"({metric}={best_score:.4f})",
                                AlertLevel.WARNING,
                            )
                        else:
                            await notification_center.notify(
                                f"Model re-evaluated for '{field}': "
                                f"{result.best_model_id} remains best "
                                f"({metric}={best_score:.4f})",
                                AlertLevel.INFO,
                            )
                    except Exception as e:
                        logger.error(
                            "----- [EVALUATING] '%s' — evaluation FAILED: %s",
                            field,
                            e,
                        )
                    finally:
                        field_states[field] = "monitoring"
                        field_jobs.pop(field, None)
                        logger.info("----- [MONITORING] '%s' — resumed", field)

                # MONITORING: score the best model; trigger retraining on degradation
                else:
                    logger.info("----- [MONITORING] '%s' — scoring best model.", field)
                    try:
                        result = await svc.monitor_best_model(
                            field, trigger="auto_monitor"
                        )
                        if result.score is None:
                            logger.info(
                                "----- [MONITORING] '%s' — no score (no data?), skipping",
                                field,
                            )
                            continue

                        baseline = svc.get_baseline_score(field)
                        if baseline and svc._score_is_worse(
                            result.score,
                            baseline,
                            result.metric,
                            settings.MONITORING_DEGRADATION_FACTOR,
                        ):
                            logger.warning(
                                "----- [MONITORING] '%s' — DEGRADED  %s=%.4f  baseline=%.4f  "
                                "threshold=%.4f → triggering retraining",
                                field,
                                result.metric,
                                result.score,
                                baseline,
                                baseline * settings.MONITORING_DEGRADATION_FACTOR,
                            )
                            await notification_center.notify(
                                f"Performance degradation detected for '{field}': "
                                f"{result.metric}={result.score:.4f} "
                                f"(baseline={baseline:.4f}, "
                                f"threshold={baseline * settings.MONITORING_DEGRADATION_FACTOR:.4f}). "
                                f"Triggering retraining.",
                                AlertLevel.CRITICAL,
                            )
                            model_configs = [
                                c
                                for c in svc.ml_config_service.list_all()
                                if field in c.output_fields
                            ]

                            training_svc = _build_training_service(db)
                            loop = asyncio.get_event_loop()
                            job_ids = _trigger_field_retraining(
                                field, model_configs, training_svc, loop
                            )

                            if job_ids:
                                field_states[field] = "retraining"
                                field_jobs[field] = job_ids
                                logger.info(
                                    "----- [RETRAINING] '%s' — %d job(s) queued",
                                    field,
                                    len(job_ids),
                                )
                            else:
                                logger.error(
                                    "----- [MONITORING] '%s' — degradation detected but no jobs could be queued",
                                    field,
                                )
                        else:
                            logger.info(
                                "----- [MONITORING] '%s' — OK  %s=%.4f  baseline=%.4f",
                                field,
                                result.metric,
                                result.score,
                                baseline if baseline is not None else float("nan"),
                            )
                    except Exception as e:
                        logger.error("----- [MONITORING] '%s' — error: %s", field, e)
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

    if settings.KAFKA_ENABLED:
        setup_inference_pipeline()

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
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {settings.API_HOST}:{settings.API_PORT}")

    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )
