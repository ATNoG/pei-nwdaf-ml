"""Service for training anomaly detection models."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from uuid import uuid4

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from sqlalchemy import update
from sqlalchemy.orm import Session

from src.db.anomaly_model_config import AnomalyModelConfigDB
from src.db.anomaly_training_job import AnomalyTrainingJobDB
from src.models.autoencoder import Autoencoder
from src.schemas.training import TrainingJobStatus
from src.services.anomaly_config_service import AnomalyConfigService
from src.services.data_preparation import calculate_timestamps, extract_fields
from src.services.data_storage_client import DataStorageClient

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="anomaly_training")


def _run_anomaly_training_sync(job_id: str):
    from src.core.config import settings
    from src.db.database import SessionLocal
    from src.services.anomaly_config_service import AnomalyConfigService

    db = SessionLocal()
    try:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        service = AnomalyTrainingService(
            data_storage_client=DataStorageClient(),
            anomaly_config_service=AnomalyConfigService(db),
            db=db,
        )
        asyncio.run(service.execute_training(job_id))
    finally:
        db.close()


class AnomalyTrainingService:
    """Service for orchestrating anomaly model training with MLflow tracking."""

    def __init__(
        self,
        data_storage_client: DataStorageClient,
        anomaly_config_service: AnomalyConfigService,
        db: Session,
    ):
        self.data_storage_client = data_storage_client
        self.anomaly_config_service = anomaly_config_service
        self.db = db

    def create_training_job(self, model_id: str, lookback_seconds: int) -> dict:
        """Create a training job for an anomaly model."""
        config = self.anomaly_config_service.get_config(model_id)
        if not config:
            raise ValueError(f"Anomaly model {model_id} not found")

        if not self._acquire_training_lock(model_id):
            raise ValueError(f"Anomaly model {model_id} is already training")

        job_id = str(uuid4())
        now = datetime.now()

        try:
            job = AnomalyTrainingJobDB(
                job_id=job_id,
                model_id=model_id,
                status=TrainingJobStatus.QUEUED,
                lookback_seconds=lookback_seconds,
                mlflow_run_id=None,
                created_at=now,
                started_at=None,
                completed_at=None,
                error_message=None,
            )
            self.db.add(job)
            self.db.commit()
            self.db.refresh(job)
        except Exception:
            self._release_training_lock(model_id)
            raise

        logger.info(f"Created anomaly training job {job_id} for model {model_id}")
        return {
            "job_id": job_id,
            "model_id": model_id,
            "status": TrainingJobStatus.QUEUED,
            "created_at": now,
        }

    async def execute_training(self, job_id: str):
        """Execute the actual training for an anomaly detection job."""
        try:
            job = self.get_job(job_id)
            if not job:
                logger.error(f"Anomaly job {job_id} not found")
                return

            result = self.db.execute(
                update(AnomalyTrainingJobDB)
                .where(AnomalyTrainingJobDB.job_id == job_id)
                .where(AnomalyTrainingJobDB.status == TrainingJobStatus.QUEUED)
                .values(status=TrainingJobStatus.RUNNING, started_at=datetime.now())
            )
            self.db.commit()
            if result.rowcount == 0:
                logger.info(f"Anomaly job {job_id} was not in QUEUED state (likely cancelled), aborting")
                return

            try:
                config_db = self.anomaly_config_service.get_config(job.model_id)
                config = self.anomaly_config_service.config_from_db(config_db)

                start_ts, end_ts = calculate_timestamps(job.lookback_seconds)
                logger.info(
                    f"Anomaly job {job_id}: Fetching data from {start_ts} to {end_ts}"
                )

                # Fetch data for all cells
                cell_data_dict = await self._fetch_all_cell_data(
                    start_ts, end_ts, config.window_duration_seconds
                )
                if not cell_data_dict:
                    raise ValueError("No data available from any cell")

                # Build flat feature vectors - each window is one sample
                all_features = []
                for cell_index, cell_data in cell_data_dict.items():
                    try:
                        inputs, _ = extract_fields(cell_data, config.input_fields, [])
                        all_features.extend(inputs)
                        logger.info(
                            f"Anomaly job {job_id}: Cell {cell_index} - {len(inputs)} windows"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Anomaly job {job_id}: Failed for cell {cell_index}: {e}"
                        )

                if not all_features:
                    raise ValueError("No cells had sufficient data")

                X_train = np.nan_to_num(np.array(all_features, dtype=np.float32))
                logger.info(
                    f"Anomaly job {job_id}: Training data shape {X_train.shape}"
                )

                # Contamination filter: if a model already exists, exclude above-threshold samples
                X_train = self._contamination_filter(job.model_id, config_db, X_train)

                # Train and log
                run_id = await self._train_and_log(
                    model_id=job.model_id,
                    job_id=job_id,
                    X_train=X_train,
                    config=config,
                    config_db=config_db,
                )

                job.status = TrainingJobStatus.COMPLETED
                job.completed_at = datetime.now()
                self.db.commit()
                logger.info(
                    f"Anomaly job {job_id}: Training completed with run {run_id}"
                )

            finally:
                self._release_training_lock(job.model_id)

        except Exception as e:
            logger.error(f"Anomaly job {job_id}: Training failed: {e}")
            job = self.get_job(job_id)
            if job:
                job.status = TrainingJobStatus.FAILED
                job.error_message = str(e)
                job.completed_at = datetime.now()
                self.db.commit()

    def get_job(self, job_id: str) -> AnomalyTrainingJobDB | None:
        return self.db.get(AnomalyTrainingJobDB, job_id)

    def list_jobs(
        self, model_id: str | None = None, status: str | None = None
    ) -> list[AnomalyTrainingJobDB]:
        query = self.db.query(AnomalyTrainingJobDB)
        if model_id:
            query = query.filter(AnomalyTrainingJobDB.model_id == model_id)
        if status:
            query = query.filter(AnomalyTrainingJobDB.status == status)
        return query.all()

    def dispatch(self, job_id: str, resources) -> None:
        from src.services.kube_training import get_kube_training_service
        kube = get_kube_training_service()
        if kube:
            kube.to_kube(job_id, resources, "anomaly")
        else:
            asyncio.get_running_loop().run_in_executor(_executor, _run_anomaly_training_sync, job_id)

    def cancel_job(self, job_id: str):
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Anomaly job {job_id} not found")
        if job.status not in [TrainingJobStatus.QUEUED, TrainingJobStatus.RUNNING]:
            raise ValueError(f"Anomaly job {job_id} is not in a cancellable state")

        job.status = TrainingJobStatus.CANCELLED
        job.completed_at = datetime.now()
        self.db.commit()
        logger.info(f"Anomaly job {job_id} marked as cancelled")

        if job.mlflow_run_id:
            try:
                client = MlflowClient()
                client.set_terminated(job.mlflow_run_id, status="KILLED")
            except Exception as e:
                logger.error(f"Failed to terminate MLflow run: {e}")

        from src.services.kube_training import get_kube_training_service
        kube = get_kube_training_service()
        if kube:
            try:
                kube.cancel(job_id)
            except Exception as e:
                logger.error(f"Failed to delete K8s job for {job_id}: {e}")
            finally:
                self._release_training_lock(job.model_id)
        # else: thread pool worker releases the lock via execute_training's finally block

    # ── Internal helpers ──────────────────────────────────────────────

    def _acquire_training_lock(self, model_id: str) -> bool:
        result = self.db.execute(
            update(AnomalyModelConfigDB)
            .where(AnomalyModelConfigDB.model_id == model_id)
            .where(AnomalyModelConfigDB.is_training == False)  # noqa: E712
            .values(is_training=True)
        )
        self.db.commit()
        return result.rowcount > 0

    def _release_training_lock(self, model_id: str):
        self.db.execute(
            update(AnomalyModelConfigDB)
            .where(AnomalyModelConfigDB.model_id == model_id)
            .values(is_training=False)
        )
        self.db.commit()

    async def _fetch_all_cell_data(
        self,
        start_timestamp: int,
        end_timestamp: int,
        window_duration_seconds: int,
    ) -> dict[int, list[dict]]:
        cells = await self.data_storage_client.get_known_cells()
        logger.info(f"Found {len(cells)} cells to fetch data from")

        cell_data: dict[int, list[dict]] = {}
        for cell_index in cells:
            try:
                data = await self.data_storage_client.fetch_cell_data(
                    cell_index=cell_index,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    window_duration_seconds=window_duration_seconds,
                    ip_src="*",
                )
                if data:
                    data.sort(key=lambda x: x.get("window_start_time", 0))
                    cell_data[cell_index] = data
            except Exception as e:
                logger.error(f"Failed to fetch data for cell {cell_index}: {e}")
        return cell_data

    def _contamination_filter(
        self,
        model_id: str,
        config_db: AnomalyModelConfigDB,
        X: np.ndarray,
    ) -> np.ndarray:
        """Remove samples that exceed the current threshold (if a model exists)."""
        if config_db.threshold_value is None:
            return X

        try:
            client = MlflowClient()
            versions = client.get_latest_versions(model_id, stages=["None"])
            if not versions:
                return X

            latest = max(versions, key=lambda v: int(v.version))
            model_uri = f"models:/{model_id}/{latest.version}"
            loaded_net = mlflow.pytorch.load_model(model_uri)

            ae = Autoencoder(
                input_size=X.shape[1],
                hidden_size=config_db.hidden_size,
            )
            ae.model = loaded_net

            # Load scaler and apply before scoring
            scaler_mean, scaler_std = self._load_scaler(model_id, latest.run_id)
            X_scaled = self._apply_scaler(X, scaler_mean, scaler_std)
            errors = ae.score(X_scaled)

            mask = errors <= config_db.threshold_value
            kept = int(mask.sum())
            logger.info(
                f"Contamination filter: kept {kept}/{len(X)} samples "
                f"(threshold={config_db.threshold_value:.6f})"
            )

            # If filtering would remove all samples, skip filtering entirely.
            # This can happen when retraining with different data granularity
            # (e.g., switching from aggregated to per-IP data).
            if kept == 0:
                logger.warning(
                    "Contamination filter would remove all samples - skipping filter. "
                    "This may indicate a change in data distribution or granularity."
                )
                return X

            return X[mask]
        except Exception as e:
            logger.warning(f"Contamination filter skipped: {e}")
            return X

    @staticmethod
    def _fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute mean and std for standardisation."""
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0  # avoid division by zero for constant features
        return mean, std

    @staticmethod
    def _apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return ((X - mean) / std).astype(np.float32)

    async def _train_and_log(
        self,
        model_id: str,
        job_id: str,
        X_train: np.ndarray,
        config,
        config_db: AnomalyModelConfigDB,
    ) -> str:
        import os
        import tempfile

        # Validate minimum samples
        if len(X_train) == 0:
            raise ValueError("No training samples available after filtering")
        if len(X_train) < 10:
            logger.warning(
                f"Very few training samples ({len(X_train)}), model may not generalise well"
            )

        num_features = X_train.shape[1]

        with mlflow.start_run(run_name=f"{model_id}_anomaly_training") as run:
            run_id = run.info.run_id

            # Link run to job immediately
            job = self.get_job(job_id)
            if job:
                job.mlflow_run_id = run_id
                self.db.commit()

            mlflow.log_params(
                {
                    "model_id": model_id,
                    "architecture": "autoencoder",
                    "num_samples": len(X_train),
                    "num_features": num_features,
                    "hidden_size": config.hidden_size,
                    "threshold_percentile": config.threshold_percentile,
                    "window_duration_seconds": config.window_duration_seconds,
                    "input_fields": config.input_fields,
                }
            )

            # Fit scaler and normalise training data
            scaler_mean, scaler_std = self._fit_scaler(X_train)
            X_scaled = self._apply_scaler(X_train, scaler_mean, scaler_std)

            # Log scaler as artifact so detection can load it
            with tempfile.TemporaryDirectory() as tmpdir:
                scaler_path = os.path.join(tmpdir, "scaler.npz")
                np.savez(scaler_path, mean=scaler_mean, std=scaler_std)
                mlflow.log_artifact(scaler_path, artifact_path="scaler")

            # Always train from scratch - scaler + threshold must match
            ae = Autoencoder(input_size=num_features, hidden_size=config.hidden_size)

            def status_callback(epoch, max_epochs, loss):
                logger.info(f"Epoch {epoch}/{max_epochs}: loss={loss:.6f}")
                mlflow.log_metric("training_loss", loss, step=epoch)
                current_job = self.get_job(job_id)
                if current_job and current_job.status == TrainingJobStatus.CANCELLED:
                    raise InterruptedError("Training cancelled by user")

            final_loss = ae.train(
                X_scaled,
                max_epochs=100,
                status_callback=status_callback,
            )
            mlflow.log_metric("final_loss", final_loss)

            # Compute threshold on scaled training data
            errors = ae.score(X_scaled)
            threshold_value = float(np.percentile(errors, config.threshold_percentile))
            mlflow.log_metric("threshold_value", threshold_value)
            logger.info(
                f"Anomaly threshold: {threshold_value:.6f} "
                f"(p{config.threshold_percentile})"
            )

            # Store threshold in DB
            self.anomaly_config_service.update_threshold(model_id, threshold_value)

            # Store scaled training data for explainability (KernelSHAP background + PI)
            with tempfile.TemporaryDirectory() as tmpdir:
                bg_path = os.path.join(tmpdir, "background.npz")
                np.savez(bg_path, X_background=X_scaled)
                mlflow.log_artifact(bg_path, artifact_path="background")
            logger.info(f"Stored anomaly background ({len(X_scaled)} samples) as MLflow artifact")

            # Log model to MLflow
            mlflow.pytorch.log_model(
                pytorch_model=ae.model,
                artifact_path="model",
                registered_model_name=None,
            )

            # Register model version
            model_uri = f"runs:/{run_id}/model"
            result = mlflow.register_model(model_uri, model_id)
            mlflow.set_tag("model_version", result.version)

            logger.info(f"Anomaly model {model_id} v{result.version} logged")
            return run_id

    @staticmethod
    def _load_scaler(model_id: str, run_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Load scaler params (mean, std) from an MLflow run artifact."""
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="scaler/scaler.npz"
        )
        data = np.load(artifact_path)
        return data["mean"], data["std"]
