"""NWDAF ML Service - FastAPI application entrypoint."""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import mlflow
from fastapi import FastAPI

from src.core.config import settings
from src.core.monitoring_state import (
    clear_field_jobs,
    get_field_jobs,
    get_field_state,
    set_field_jobs,
    set_field_state,
    set_last_checked,
)
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
            # dispatch resolves default resources from model config automatically
            training_svc.dispatch(job_info["job_id"], resources=None)
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
    """Return True if any data windows have arrived since last_ts."""
    try:
        now = int(time.time())
        data = await data_storage_client.fetch_data(
            start_timestamp=last_ts,
            end_timestamp=now,
            window_duration_seconds=60,
        )
        return bool(data)
    except Exception as e:
        logger.warning("Auto-monitor: failed to check for new data: %s", e)
        return False
    return False


async def _monitoring_loop() -> None:
    """
    Background task that periodically re-scores the best model for every
    monitored output field.

    Each field moves through a simple state machine:
      MONITORING  - score the best model; on degradation trigger retraining
      RETRAINING  - wait for all queued training jobs to finish
      EVALUATING  - re-elect a best model via evaluate_field()

    Controlled by MONITORING_TRIGGER_MODE:
      "time"  - run on every interval regardless of new data
      "data"  - run only when new data has arrived since last cycle
      "both"  - run on interval only if new data has arrived
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

        # Data-income gate - skip cycle if no new windows have arrived
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
                event_type, field_name = field.split(":", 1)
                state = get_field_state(field)

                # RETRAINING: wait for all queued jobs to reach a terminal state
                if state == "retraining":
                    from src.db.training_job import TrainingJobDB

                    jobs = [db.get(TrainingJobDB, jid) for jid in get_field_jobs(field)]
                    active = [
                        j for j in jobs if j and j.status in ("queued", "running")
                    ]

                    if active:
                        logger.info(
                            "----- [RETRAINING] '%s' - %d/%d job(s) still running.",
                            field_name,
                            len(active),
                            len(jobs),
                        )
                        continue  # check again next cycle

                    failed = [j for j in jobs if j and j.status == "failed"]
                    completed = [j for j in jobs if j and j.status == "completed"]

                    for j in failed:
                        logger.error(
                            "----- [RETRAINING] '%s' - job %s FAILED: %s",
                            field_name,
                            j.job_id,
                            j.error_message,
                        )

                    if completed:
                        set_field_state(field, "evaluating")
                        logger.info(
                            "----- [RETRAINING] '%s' - all done (%d completed, %d failed) → moving to EVALUATING",
                            field_name,
                            len(completed),
                            len(failed),
                        )
                        await notification_center.notify(
                            f"Retraining completed for '{field_name}': "
                            f"{len(completed)} succeeded, {len(failed)} failed. "
                            f"Evaluating best model.",
                            AlertLevel.INFO,
                        )
                    else:
                        logger.error(
                            "----- [RETRAINING] '%s' - all jobs failed → returning to MONITORING",
                            field_name,
                        )
                        await notification_center.notify(
                            f"Retraining failed for '{field_name}': all jobs failed.",
                            AlertLevel.WARNING,
                        )
                        set_field_state(field, "monitoring")
                        clear_field_jobs(field)

                # EVALUATING: re-elect a best model then return to monitoring
                elif state == "evaluating":
                    logger.info(
                        "----- [EVALUATING] '%s' - scoring all retrained models.", field_name
                    )
                    try:
                        best = svc.get_best_model(field_name)
                        metric = best.metric if best and best.metric else "rmse"
                        result = await svc.evaluate_field(field_name, metric)
                        best_score = next(
                            (m.score for m in result.models if m.is_best), float("nan")
                        )

                        logger.info(
                            "----- [EVALUATING] '%s' - new best: %s  %s=%.4f",
                            field_name,
                            result.best_model_id,
                            metric,
                            best_score,
                        )
                        prev_id = best.model_id if best else None
                        if result.best_model_id != prev_id:
                            await notification_center.notify(
                                f"Best model changed for '{field_name}': "
                                f"{prev_id} → {result.best_model_id} "
                                f"({metric}={best_score:.4f})",
                                AlertLevel.WARNING,
                            )
                        else:
                            await notification_center.notify(
                                f"Model re-evaluated for '{field_name}': "
                                f"{result.best_model_id} remains best "
                                f"({metric}={best_score:.4f})",
                                AlertLevel.INFO,
                            )
                    except Exception as e:
                        logger.error(
                            "----- [EVALUATING] '%s' - evaluation FAILED: %s",
                            field_name,
                            e,
                        )
                    finally:
                        set_field_state(field, "monitoring")
                        clear_field_jobs(field)
                        logger.info("----- [MONITORING] '%s' - resumed", field_name)

                # MONITORING: score the best model; trigger retraining on degradation
                else:
                    logger.info("----- [MONITORING] '%s' - scoring best model.", field_name)
                    try:
                        result = await svc.monitor_best_model(
                            field_name, trigger="auto_monitor"
                        )
                        if result.score is None:
                            logger.info(
                                "----- [MONITORING] '%s' - no score (no data?), skipping",
                                field_name,
                            )
                            continue

                        baseline = svc.get_baseline_score(field_name)
                        if baseline and svc._score_is_worse(
                            result.score,
                            baseline,
                            result.metric,
                            settings.MONITORING_DEGRADATION_FACTOR,
                        ):
                            logger.warning(
                                "----- [MONITORING] '%s' - DEGRADED  %s=%.4f  baseline=%.4f  "
                                "threshold=%.4f → triggering retraining",
                                field_name,
                                result.metric,
                                result.score,
                                baseline,
                                baseline * settings.MONITORING_DEGRADATION_FACTOR,
                            )
                            await notification_center.notify(
                                f"Performance degradation detected for '{field_name}': "
                                f"{result.metric}={result.score:.4f} "
                                f"(baseline={baseline:.4f}, "
                                f"threshold={baseline * settings.MONITORING_DEGRADATION_FACTOR:.4f}). "
                                f"Triggering retraining.",
                                AlertLevel.CRITICAL,
                            )
                            model_configs = [
                                c
                                for c in svc.ml_config_service.list_all()
                                if field_name in c.output_fields
                            ]

                            training_svc = _build_training_service(db)
                            loop = asyncio.get_event_loop()
                            job_ids = _trigger_field_retraining(
                                field_name, model_configs, training_svc, loop
                            )

                            if job_ids:
                                set_field_state(field, "retraining")
                                set_field_jobs(field, job_ids)
                                logger.info(
                                    "----- [RETRAINING] '%s' - %d job(s) queued",
                                    field_name,
                                    len(job_ids),
                                )
                            else:
                                logger.error(
                                    "----- [MONITORING] '%s' - degradation detected but no jobs could be queued",
                                    field_name,
                                )
                        else:
                            set_last_checked(field, datetime.now(tz=timezone.utc))
                            logger.info(
                                "----- [MONITORING] '%s' - OK  %s=%.4f  baseline=%.4f",
                                field_name,
                                result.metric,
                                result.score,
                                baseline if baseline is not None else float("nan"),
                            )
                    except Exception as e:
                        logger.error("----- [MONITORING] '%s' - error: %s", field_name, e)
        finally:
            db.close()


async def _kube_reconciliation_loop():
    """Detect K8s jobs that failed before the worker could update the DB and mark them as FAILED."""
    from sqlalchemy import update as sa_update

    from src.db.anomaly_model_config import AnomalyModelConfigDB
    from src.db.anomaly_training_job import AnomalyTrainingJobDB
    from src.db.database import SessionLocal
    from src.db.model_config import ModelConfigDB
    from src.db.training_job import TrainingJobDB
    from src.schemas.training import TrainingJobStatus
    from src.services.kube_training import get_kube_training_service

    active = [TrainingJobStatus.RUNNING, TrainingJobStatus.QUEUED]

    while True:
        await asyncio.sleep(30)
        kube = get_kube_training_service()
        if not kube:
            continue
        db = SessionLocal()
        try:
            for job in (
                db.query(TrainingJobDB).filter(TrainingJobDB.status.in_(active)).all()
            ):
                if kube.is_job_dead(job.model_id):
                    job.status = TrainingJobStatus.FAILED
                    job.error_message = "K8s job failed or not found"
                    job.completed_at = datetime.now()
                    db.execute(
                        sa_update(ModelConfigDB)
                        .where(ModelConfigDB.model_id == job.model_id)
                        .values(is_training=False)
                    )
                    logger.warning(
                        "Reconciliation: forecast job %s marked FAILED", job.job_id
                    )

            for job in (
                db.query(AnomalyTrainingJobDB)
                .filter(AnomalyTrainingJobDB.status.in_(active))
                .all()
            ):
                if kube.is_job_dead(job.model_id):
                    job.status = TrainingJobStatus.FAILED
                    job.error_message = "K8s job failed or not found"
                    job.completed_at = datetime.now()
                    db.execute(
                        sa_update(AnomalyModelConfigDB)
                        .where(AnomalyModelConfigDB.model_id == job.model_id)
                        .values(is_training=False)
                    )
                    logger.warning(
                        "Reconciliation: anomaly job %s marked FAILED", job.job_id
                    )

            db.commit()
        except Exception as e:
            logger.error("Kube reconciliation error: %s", e)
        finally:
            db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting NWDAF ML Service...")
    logger.info(f"MLflow tracking URI: {settings.MLFLOW_TRACKING_URI}")
    logger.info(f"Data Storage API: {settings.DATA_STORAGE_API_URL}")
    logger.info(f"Database URL: {settings.DATABASE_URL}")
    logger.info(f"Policy Service: {settings.POLICY_SERVICE_URL} (enabled={settings.POLICY_ENABLED})")

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.MLFLOW_S3_ENDPOINT_URL
    os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.AWS_SECRET_ACCESS_KEY
    os.environ["AWS_DEFAULT_REGION"] = settings.AWS_DEFAULT_REGION

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    logger.info("Initializing database tables...")
    init_db()
    logger.info("Database initialized successfully")

    if settings.ADD_TEST_MODELS:
        from src.db.database import SessionLocal as _SL
        from src.services.architecture_service import ArchitectureService as _AS
        _db = _SL()
        try:
            _AS().seed_builtins(_db)
        finally:
            _db.close()

    if settings.KAFKA_ENABLED:
        setup_inference_pipeline()

    # Release any training locks left over from a previous crash or restart
    from src.db.database import SessionLocal
    from src.db.model_config import ModelConfigDB

    db = SessionLocal()
    try:
        released = (
            db.query(ModelConfigDB)
            .filter(ModelConfigDB.is_training == True)
            .update({"is_training": False})
        )
        db.commit()
        if released:
            logger.warning(
                "Released %d stale training lock(s) from previous run", released
            )
    finally:
        db.close()

    monitor_task = None
    if settings.MONITORING_ENABLED:
        monitor_task = asyncio.create_task(_monitoring_loop())
        logger.info("Auto-monitor task started")

    # Policy Service integration
    policy_client = None
    policy_heartbeat_task = None
    if settings.POLICY_ENABLED:
        try:
            from src.services.config_service import MLConfigService
            from policy_client import PolicyClient

            policy_client = PolicyClient(
                service_url=settings.POLICY_SERVICE_URL,
                component_id=settings.POLICY_COMPONENT_ID,
                enable_policy=True,
                fail_open=settings.POLICY_FAILOPEN,
            )

            # Register ML service as a component
            registered = await policy_client.register_component(
                component_type="ml_agent",
                role=settings.POLICY_ROLENAME,
            )

            if registered:
                await policy_client.start_heartbeat()
                logger.info(f"Policy client registered: {settings.POLICY_COMPONENT_ID}")

                # Reuse mlflow client for model registration to avoid resource leaks
                from mlflow import MlflowClient
                mlflow_client_for_policy = MlflowClient()

                # Register all existing ML models
                db = SessionLocal()
                try:
                    config_service = MLConfigService(db)
                    models = config_service.list_all()
                    for model in models:
                        result = await policy_client.register_ml_model(
                            model_id=model.model_id,
                            model_name=model.name,
                            input_fields=model.input_fields,
                            output_fields=model.output_fields,
                            data_type="network_prediction",  # Default data type
                            architecture=model.architecture,
                            window_duration_seconds=model.window_duration_seconds,
                        )
                        if result:
                            logger.info(f"Registered ML model with policy: {model.name}")
                        else:
                            logger.warning(f"Failed to register model {model.name} with policy (returned False)")
                finally:
                    db.close()

                # Store on app state for dependency injection
                app.state.policy_client = policy_client

                # Track registered models to avoid re-registering
                _registered_models = set()

                # Start background task to register new models
                async def _model_registration_loop():
                    """Periodically check for new models and register them."""
                    nonlocal _registered_models
                    while True:
                        await asyncio.sleep(60)  # Check every minute
                        db = SessionLocal()
                        try:
                            config_service = MLConfigService(db)
                            models = config_service.list_all()
                            for model in models:
                                # Only register models we haven't seen yet
                                if model.model_id not in _registered_models:
                                    result = await policy_client.register_ml_model(
                                        model_id=model.model_id,
                                        model_name=model.name,
                                        input_fields=model.input_fields,
                                        output_fields=model.output_fields,
                                        data_type="network_prediction",
                                        architecture=model.architecture,
                                        window_duration_seconds=model.window_duration_seconds,
                                    )
                                    if result:
                                        _registered_models.add(model.model_id)
                                        logger.info(f"Registered new ML model with policy: {model.name}")
                        finally:
                            db.close()

                policy_heartbeat_task = asyncio.create_task(_model_registration_loop())
                logger.info("Model registration loop started")
            else:
                logger.warning("Failed to register ML service with policy")

        except ImportError:
            logger.warning("Policy client SDK not available - policy integration disabled")
        except Exception as e:
            logger.error(f"Failed to initialize policy client: {e}")
    reconcile_task = None
    if settings.TRAIN_USE_KUBE:
        reconcile_task = asyncio.create_task(_kube_reconciliation_loop())
        logger.info("Kube reconciliation task started")

    yield

    if monitor_task:
        monitor_task.cancel()
        logger.info("Auto-monitor task stopped")

    # Stop policy heartbeat
    if policy_client:
        await policy_client.stop_heartbeat()
        logger.info("Policy heartbeat stopped")

    if policy_heartbeat_task:
        policy_heartbeat_task.cancel()
        try:
            await policy_heartbeat_task
        except asyncio.CancelledError:
            pass
        logger.info("Model registration loop stopped")
    if reconcile_task:
        reconcile_task.cancel()
        logger.info("Kube reconciliation task stopped")

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
