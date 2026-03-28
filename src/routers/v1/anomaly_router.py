"""Router for anomaly detection endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.core.dependencies import get_db
from src.schemas.anomaly import (
    AnomalyDetectionRequest,
    AnomalyDetectionResult,
    AnomalyDetectionSummary,
    AnomalyModelCreate,
    AnomalyModelDetail,
    AnomalyModelMeta,
    AnomalyModelSummary,
    AnomalyTrainingJobDetail,
    AnomalyTrainingJobSummary,
    AnomalyTrainingRequest,
    AnomalyTrainingResponse,
)
from src.services.anomaly_config_service import AnomalyConfigService
from src.services.anomaly_detection_service import AnomalyDetectionService
from src.services.anomaly_training_service import AnomalyTrainingService
from src.services.data_storage_client import DataStorageClient

logger = logging.getLogger(__name__)
router = APIRouter()


def get_anomaly_config_service(db: Session = Depends(get_db)) -> AnomalyConfigService:
    return AnomalyConfigService(db)


def get_anomaly_training_service(
    db: Session = Depends(get_db),
) -> AnomalyTrainingService:
    return AnomalyTrainingService(
        data_storage_client=DataStorageClient(),
        anomaly_config_service=AnomalyConfigService(db),
        db=db,
    )


def get_anomaly_detection_service(
    db: Session = Depends(get_db),
) -> AnomalyDetectionService:
    return AnomalyDetectionService(
        data_storage_client=DataStorageClient(),
        anomaly_config_service=AnomalyConfigService(db),
    )


# ── Model CRUD ───────────────────────────────────────────────────────


@router.post("/models", response_model=AnomalyModelDetail, status_code=201)
async def create_anomaly_model(
    body: AnomalyModelCreate,
    config_service: AnomalyConfigService = Depends(get_anomaly_config_service),
    data_storage_client: DataStorageClient = Depends(DataStorageClient),
) -> AnomalyModelDetail:
    """Create a new anomaly detection model."""
    try:
        # Validate input fields
        is_valid, invalid_fields = await data_storage_client.validate_fields(
            body.config.input_fields
        )
        if not is_valid:
            available = await data_storage_client.get_available_fields()
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid field names provided",
                    "invalid_fields": sorted(invalid_fields),
                    "available_fields": sorted(available),
                },
            )

        from datetime import datetime
        from uuid import uuid4

        from mlflow import MlflowClient

        model_id = str(uuid4())
        created_at = datetime.now()

        # Create MLflow registered model
        client = MlflowClient()
        client.create_registered_model(model_id)
        client.set_registered_model_tag(model_id, "name", body.name)
        client.set_registered_model_tag(model_id, "type", "anomaly")

        from src.db.anomaly_model_config import AnomalyModelConfigDB

        config_db = AnomalyModelConfigDB(
            model_id=model_id,
            name=body.name,
            input_fields=body.config.input_fields,
            window_duration_seconds=body.config.window_duration_seconds,
            hidden_size=body.config.hidden_size,
            threshold_percentile=body.config.threshold_percentile,
            threshold_value=None,
            created_at=created_at,
        )
        config_service.create(config_db)

        return AnomalyModelDetail(
            id=model_id,
            name=body.name,
            config=body.config,
            threshold_value=None,
            created_at=created_at,
            latest_version=None,
            last_trained_at=None,
            mlflow_run_id=None,
            training_loss=None,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create anomaly model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=list[AnomalyModelSummary])
async def list_anomaly_models(
    config_service: AnomalyConfigService = Depends(get_anomaly_config_service),
) -> list[AnomalyModelSummary]:
    """List all anomaly detection models."""
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    configs = config_service.list_all()
    client = MlflowClient()
    summaries = []
    for cfg in configs:
        latest_version = None
        try:
            rm = client.get_registered_model(cfg.model_id)
            if rm.latest_versions:
                latest_version = int(rm.latest_versions[0].version)
        except MlflowException:
            pass
        summaries.append(
            AnomalyModelSummary(
                id=cfg.model_id,
                name=cfg.name,
                created_at=cfg.created_at,
                latest_version=latest_version,
            )
        )
    return summaries


@router.get("/models/{model_id}", response_model=AnomalyModelDetail)
async def get_anomaly_model(
    model_id: str,
    config_service: AnomalyConfigService = Depends(get_anomaly_config_service),
) -> AnomalyModelDetail:
    """Get anomaly model details."""
    from datetime import datetime

    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    cfg = config_service.get_config(model_id)
    if not cfg:
        raise HTTPException(
            status_code=404, detail=f"Anomaly model '{model_id}' not found"
        )

    config = config_service.config_from_db(cfg)

    latest_version = None
    last_trained_at = None
    mlflow_run_id = None
    training_loss = None

    try:
        client = MlflowClient()
        rm = client.get_registered_model(model_id)
        if rm.latest_versions:
            mv = rm.latest_versions[0]
            latest_version = int(mv.version)
            if mv.run_id:
                try:
                    run = client.get_run(mv.run_id)
                    last_trained_at = datetime.fromtimestamp(run.info.end_time / 1000)
                    mlflow_run_id = mv.run_id
                    training_loss = run.data.metrics.get("final_loss")
                except Exception:
                    pass
    except MlflowException:
        pass

    return AnomalyModelDetail(
        id=model_id,
        name=cfg.name,
        config=config,
        threshold_value=cfg.threshold_value,
        created_at=cfg.created_at,
        latest_version=latest_version,
        last_trained_at=last_trained_at,
        mlflow_run_id=mlflow_run_id,
        training_loss=training_loss,
    )


@router.delete("/models/{model_id}", status_code=204)
async def delete_anomaly_model(
    model_id: str,
    config_service: AnomalyConfigService = Depends(get_anomaly_config_service),
):
    """Delete an anomaly model."""
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    try:
        config_service.delete_model(model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        MlflowClient().delete_registered_model(model_id)
    except MlflowException:
        pass


# ── Training ─────────────────────────────────────────────────────────


@router.post("/training/train", response_model=AnomalyTrainingResponse, status_code=202)
async def train_anomaly_model(
    request: AnomalyTrainingRequest,
    training_service: AnomalyTrainingService = Depends(get_anomaly_training_service),
) -> AnomalyTrainingResponse:
    """Start training an anomaly detection model."""
    try:
        job_info = training_service.create_training_job(
            request.model_id, request.lookback_seconds
        )
        training_service.dispatch(job_info["job_id"], request.resources)
        logger.info(f"Anomaly training job {job_info['job_id']} dispatched")

        return AnomalyTrainingResponse(
            job_id=job_info["job_id"],
            model_id=job_info["model_id"],
            status=job_info["status"],
            message="Anomaly training job queued. Check status at GET /anomaly/training/jobs/{job_id}",
            created_at=job_info["created_at"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create anomaly training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/jobs", response_model=list[AnomalyTrainingJobSummary])
async def list_anomaly_training_jobs(
    model_id: str | None = None,
    status: str | None = None,
    training_service: AnomalyTrainingService = Depends(get_anomaly_training_service),
) -> list[AnomalyTrainingJobSummary]:
    """List anomaly training jobs."""
    jobs = training_service.list_jobs(model_id=model_id, status=status)
    return [
        AnomalyTrainingJobSummary(
            job_id=j.job_id,
            model_id=j.model_id,
            status=j.status,
            created_at=j.created_at,
            started_at=j.started_at,
        )
        for j in jobs
    ]


@router.get("/training/jobs/{job_id}", response_model=AnomalyTrainingJobDetail)
async def get_anomaly_training_job(
    job_id: str,
    training_service: AnomalyTrainingService = Depends(get_anomaly_training_service),
) -> AnomalyTrainingJobDetail:
    """Get anomaly training job details."""
    job = training_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Anomaly job {job_id} not found")

    return AnomalyTrainingJobDetail(
        job_id=job.job_id,
        model_id=job.model_id,
        status=job.status,
        mlflow_run_id=job.mlflow_run_id,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
    )


@router.delete("/training/jobs/{job_id}", status_code=204)
async def cancel_anomaly_training_job(
    job_id: str,
    training_service: AnomalyTrainingService = Depends(get_anomaly_training_service),
):
    """Cancel an anomaly training job."""
    try:
        training_service.cancel_job(job_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to cancel anomaly job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Detection ────────────────────────────────────────────────────────


@router.post("/detect", response_model=AnomalyDetectionResult)
async def run_anomaly_detection(
    request: AnomalyDetectionRequest,
    detection_service: AnomalyDetectionService = Depends(get_anomaly_detection_service),
) -> AnomalyDetectionResult:
    """Run anomaly detection for all IPs in a cell using a single model."""
    try:
        return await detection_service.detect(
            request.cell_id, request.model_id, request.lookback_seconds
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/all", response_model=AnomalyDetectionSummary)
async def run_all_anomaly_detection(
    request: AnomalyDetectionRequest,
    detection_service: AnomalyDetectionService = Depends(get_anomaly_detection_service),
) -> AnomalyDetectionSummary:
    """Run anomaly detection for all IPs in a cell using all trained models."""
    trained_models = [
        cfg
        for cfg in detection_service.anomaly_config_service.list_all()
        if cfg.threshold_value is not None
    ]
    if not trained_models:
        raise HTTPException(
            status_code=404, detail="No trained anomaly models available"
        )

    models_meta: dict[str, AnomalyModelMeta] = {}
    ip_anomalies: dict[str, dict[str, str]] = {}

    for model_cfg in trained_models:
        try:
            result = await detection_service.detect(
                request.cell_id, model_cfg.model_id, request.lookback_seconds
            )
            models_meta[result.model_id] = AnomalyModelMeta(
                name=result.model_name,
                fields=result.input_fields,
                threshold=result.threshold_value,
                window_duration_seconds=result.window_duration_seconds,
            )
            for ip_result in result.results:
                ip = ip_result.ip_src
                ip_anomalies.setdefault(ip, {})[
                    result.model_id
                ] = f"{ip_result.num_anomalies}/{ip_result.num_windows}"
        except Exception as e:
            logger.warning(f"Model {model_cfg.name} skipped: {e}")

    return AnomalyDetectionSummary(
        cell_id=request.cell_id,
        models=models_meta,
        ip_anomalies=ip_anomalies,
    )
