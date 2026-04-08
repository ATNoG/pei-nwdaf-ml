"""Service for training ML models."""

import logging
from datetime import datetime
from uuid import uuid4
import numpy as np
from sqlalchemy import update
from sqlalchemy.orm import Session

from src.db.model_config import ModelConfigDB
from src.db.training_job import TrainingJobDB
from src.services.data_storage_client import DataStorageClient
from src.services.mlflow_service import MLflowService
from src.services.config_service import MLConfigService
from src.services.data_preparation import (
    calculate_timestamps,
    extract_fields,
    prepare_sequences,
)
from src.schemas.model import ArchitectureType
from src.schemas.training import TrainingJobStatus
from src.models import MODEL_REGISTRY
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class TrainingService:
    """Service for orchestrating model training with MLflow tracking."""

    def __init__(
        self,
        mlflow_service: MLflowService,
        data_storage_client: DataStorageClient,
        ml_config_service: MLConfigService,
        db: Session,
    ):
        self.mlflow_service = mlflow_service
        self.data_storage_client = data_storage_client
        self.ml_config_service = ml_config_service
        self.db = db

    def create_training_job(self, model_id: str, lookback_seconds: int) -> dict:
        """
        Create a training job and store in database.

        Args:
            model_id: UUID of model to train
            lookback_seconds: How far back to fetch data (seconds)

        Returns:
            Job information dict with job_id, model_id, status, created_at

        Raises:
            ValueError: If model not found or already training
        """
        # Validate model exists
        model_detail = self.mlflow_service.get_model(model_id)
        if not model_detail:
            raise ValueError(f"Model {model_id} not found")

        # Atomically acquire training lock
        if not self._acquire_training_lock(model_id):
            raise ValueError(f"Model {model_id} is already training")

        # Generate job ID
        job_id = str(uuid4())
        now = datetime.now()

        try:
            # Create job record
            job = TrainingJobDB(
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

            # Save to database
            self.db.add(job)
            self.db.commit()
            self.db.refresh(job)
        except Exception as e:
            # Release lock if job creation fails
            self._release_training_lock(model_id)
            raise

        logger.info(f"Created training job {job_id} for model {model_id}")

        return {
            "job_id": job_id,
            "model_id": model_id,
            "status": TrainingJobStatus.QUEUED,
            "created_at": now,
        }

    async def execute_training(self, job_id: str):
        """
        Execute the actual training for a job.

        Args:
            job_id: Training job ID to execute
        """
        try:
            # Get job details
            job = self.get_job(job_id)
            if not job:
                logger.error(f"Job {job_id} not found")
                return

            # Check if job was cancelled before starting
            if job.status == TrainingJobStatus.CANCELLED:
                logger.info(f"Job {job_id} was cancelled before execution")
                return

            # Update job status to running
            job.status = TrainingJobStatus.RUNNING
            job.started_at = datetime.now()
            self.db.commit()

            # Lock is already acquired in create_training_job
            try:
                # Get model config
                model_detail = self.mlflow_service.get_model(job.model_id)
                config = model_detail.config

                # Calculate timestamps
                start_ts, end_ts = self._calculate_timestamps(job.lookback_seconds)
                logger.info(f"Job {job_id}: Fetching data from {start_ts} to {end_ts}")

                # Fetch data for all cells
                # Derive component_id for policy enforcement
                component_id = f"ml-{model_detail.name}" if model_detail.name else None
                cell_data_dict = await self._fetch_all_cell_data(
                    start_ts, end_ts, config.window_duration_seconds, component_id
                )

                if not cell_data_dict:
                    raise ValueError("No data available from any cell")

                # Prepare sequences from each cell
                all_X = []
                all_y = []
                successful_cells = 0
                failed_cells = 0

                for cell_index, cell_data in cell_data_dict.items():
                    try:
                        X, y = self._prepare_sequences_per_cell(
                            cell_data,
                            config.input_fields,
                            config.output_fields,
                            config.lookback_steps,
                            config.forecast_steps,
                        )

                        if len(X) == 0:
                            raise ValueError(
                                f"Insufficient data: got {len(cell_data)} windows, "
                                f"need at least {config.lookback_steps + config.forecast_steps}"
                            )

                        all_X.append(X)
                        all_y.append(y)
                        successful_cells += 1
                        logger.info(f"Job {job_id}: Cell {cell_index} - created {len(X)} sequences")

                    except Exception as e:
                        failed_cells += 1
                        logger.warning(
                            f"Job {job_id}: Failed to prepare data for cell {cell_index}: {str(e)}"
                        )
                        continue

                # Check if we have any successful cells
                if successful_cells == 0:
                    raise ValueError(
                        f"No cells had sufficient data. "
                        f"Failed cells: {failed_cells}. "
                        f"Try increasing lookback_seconds."
                    )

                # Combine sequences from all successful cells
                X_train = np.concatenate(all_X, axis=0)
                y_train = np.concatenate(all_y, axis=0)

                logger.info(
                    f"Job {job_id}: Combined training data - "
                    f"{len(X_train)} sequences from {successful_cells} cells "
                    f"({failed_cells} cells failed)"
                )

                # Train model and log to MLflow
                # Note: mlflow_run_id is set inside _train_and_log immediately after run starts
                mlflow_run_id = await self._train_and_log(
                    model_id=job.model_id,
                    job_id=job_id,
                    architecture=config.architecture,
                    X_train=X_train,
                    y_train=y_train,
                    config_dict={
                        "hidden_size": config.hidden_size,
                        "lookback_steps": config.lookback_steps,
                        "forecast_steps": config.forecast_steps,
                        "input_fields": config.input_fields,
                        "output_fields": config.output_fields,
                        "window_duration_seconds": config.window_duration_seconds,
                    },
                )

                # Mark job as completed (only if not cancelled)
                job = self.get_job(job_id)
                if job and job.status != TrainingJobStatus.CANCELLED:
                    job.status = TrainingJobStatus.COMPLETED
                    job.completed_at = datetime.now()
                    self.db.commit()
                    logger.info(f"Job {job_id}: Training completed successfully with run {mlflow_run_id}")

            finally:
                # Always release lock
                self._release_training_lock(job.model_id)

        except Exception as e:
            # If job was cancelled, InterruptedError will be raised — don't overwrite status
            job = self.get_job(job_id)
            if job and job.status == TrainingJobStatus.CANCELLED:
                logger.info(f"Job {job_id}: Training interrupted by cancellation")
                return
            logger.error(f"Job {job_id}: Training failed with error: {str(e)}")
            if job:
                job.status = TrainingJobStatus.FAILED
                job.error_message = str(e)
                job.completed_at = datetime.now()
                self.db.commit()


    def get_job(self, job_id: str) -> TrainingJobDB | None:
        """Get training job by ID."""
        return self.db.get(TrainingJobDB, job_id)

    def list_jobs(self, model_id: str | None = None, status: str | None = None) -> list[TrainingJobDB]:
        """List training jobs with optional filters."""
        query = self.db.query(TrainingJobDB)
        if model_id:
            query = query.filter(TrainingJobDB.model_id == model_id)
        if status:
            query = query.filter(TrainingJobDB.status == status)
        return query.all()

    def cancel_job(self, job_id: str):
        """
        Cancel a training job.

        Args:
            job_id: Job ID to cancel
        """
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        if job.status not in [TrainingJobStatus.QUEUED, TrainingJobStatus.RUNNING]:
            raise ValueError(f"Job {job_id} is not in a cancellable state")

        # Update job status
        job.status = TrainingJobStatus.CANCELLED
        job.completed_at = datetime.now()
        self.db.commit()

        logger.info(f"Job {job_id} marked as cancelled")

        # If job was running, clean up resources
        if job.mlflow_run_id:
            try:
                # Stop MLflow run
                client = MlflowClient()
                client.set_terminated(job.mlflow_run_id, status="KILLED")
                logger.info(f"Terminated MLflow run {job.mlflow_run_id}")
            except Exception as e:
                logger.error(f"Failed to terminate MLflow run: {str(e)}")

        # Release training lock
        self._release_training_lock(job.model_id)
        logger.info(f"Released training lock for model {job.model_id}")

    def _acquire_training_lock(self, model_id: str) -> bool:
        """
        Atomically acquire training lock for a model.

        Args:
            model_id: Model ID to lock

        Returns:
            True if lock acquired, False if already locked
        """
        # Atomic update: only set is_training=True if currently False
        result = self.db.execute(
            update(ModelConfigDB)
            .where(ModelConfigDB.model_id == model_id)
            .where(ModelConfigDB.is_training == False)  # noqa: E712
            .values(is_training=True)
        )
        self.db.commit()

        acquired = result.rowcount > 0
        if acquired:
            logger.info(f"Acquired training lock for model {model_id}")
        else:
            logger.warning(f"Failed to acquire training lock for model {model_id} (already locked)")

        return acquired

    def _release_training_lock(self, model_id: str):
        """
        Release training lock for a model.

        Args:
            model_id: Model ID to unlock
        """
        self.db.execute(
            update(ModelConfigDB)
            .where(ModelConfigDB.model_id == model_id)
            .values(is_training=False)
        )
        self.db.commit()
        logger.info(f"Released training lock for model {model_id}")

    def _calculate_timestamps(self, lookback_seconds: int) -> tuple[int, int]:
        """Calculate start and end timestamps from lookback duration."""
        return calculate_timestamps(lookback_seconds)

    async def _fetch_all_cell_data(
        self,
        start_timestamp: int,
        end_timestamp: int,
        window_duration_seconds: int,
        component_id: str | None = None,
    ) -> dict[int, list[dict]]:
        """
        Fetch data for all known cells.

        Args:
            start_timestamp: Start time
            end_timestamp: End time
            window_duration_seconds: Window duration from model config
            component_id: Optional component ID to pass for policy enforcement

        Returns:
            Dict mapping cell_index to list of data windows
        """
        # Get all known cells
        cells = await self.data_storage_client.get_known_cells(component_id=component_id)
        logger.info(f"Found {len(cells)} cells to fetch data from")

        # Fetch data for each cell
        cell_data = {}
        for cell_index in cells:
            try:
                data = await self.data_storage_client.fetch_cell_data(
                    cell_index=cell_index,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    window_duration_seconds=window_duration_seconds,
                    component_id=component_id,
                )
                if data:
                    # Sort by timestamp to ensure proper sequence order
                    data.sort(key=lambda x: x.get("window_start_time", 0))
                    cell_data[cell_index] = data
                    logger.info(f"Fetched {len(data)} windows for cell {cell_index}")
            except Exception as e:
                logger.error(f"Failed to fetch data for cell {cell_index}: {str(e)}")
                # Continue with other cells even if one fails

        return cell_data

    def _extract_fields(
        self,
        data: list[dict],
        input_fields: list[str],
        output_fields: list[str],
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Extract input and output field values from data windows."""
        return extract_fields(data, input_fields, output_fields)

    def _prepare_sequences_per_cell(
        self,
        cell_data: list[dict],
        input_fields: list[str],
        output_fields: list[str],
        lookback_steps: int,
        forecast_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create training sequences from cell data using sliding windows."""
        return prepare_sequences(
            cell_data, input_fields, output_fields, lookback_steps, forecast_steps
        )

    async def _train_and_log(
        self,
        model_id: str,
        job_id: str,
        architecture: ArchitectureType,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config_dict: dict,
    ) -> str:
        """
        Train model and log to MLflow.

        Args:
            model_id: Model ID (used as MLflow model name)
            job_id: Training job ID (for cancellation checks)
            architecture: Model architecture type
            X_train: Training input data
            y_train: Training output data
            config_dict: Model configuration

        Returns:
            MLflow run ID

        Raises:
            InterruptedError: If training is cancelled
        """
        with mlflow.start_run(run_name=f"{model_id}_training") as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run {run_id} for model {model_id}")

            # CRITICAL: Store run_id immediately so it can be cancelled
            job = self.get_job(job_id)
            if job:
                job.mlflow_run_id = run_id
                self.db.commit()
                logger.info(f"Job {job_id}: MLflow run {run_id} linked to job")

            # Log parameters
            mlflow.log_params({
                "model_id": model_id,
                "architecture": architecture.value,
                "num_sequences": len(X_train),
                "num_input_features": X_train.shape[2],
                "num_output_features": y_train.shape[2],
                **config_dict
            })

            # Build or load model for incremental training
            model = self._build_model(
                model_id=model_id,
                architecture=architecture,
                input_fields=config_dict["input_fields"],
                output_fields=config_dict["output_fields"],
                window_duration_seconds=config_dict["window_duration_seconds"],
                lookback_steps=config_dict["lookback_steps"],
                forecast_steps=config_dict["forecast_steps"],
                hidden_size=config_dict["hidden_size"],
            )

            # Train model (with cancellation support)
            final_loss = self._train_pytorch_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                job_id=job_id,
                epochs=100,
                batch_size=32,
                learning_rate=0.001,
            )

            # Log final metrics
            mlflow.log_metric("final_loss", final_loss)

            # Log model to MLflow
            model_uri = f"runs:/{run_id}/model"
            mlflow.pytorch.log_model(
                pytorch_model=model.model,
                artifact_path="model",
                registered_model_name=None
            )

            # Register or update model
            version = self._register_or_update_model(model_id, model_uri)
            mlflow.set_tag("model_version", version)

            logger.info(f"Model {model_id} v{version} logged successfully")

            return run_id

    def _register_or_update_model(self, model_id: str, model_uri: str) -> str:
        """
        Register new model or add version to existing model.

        Args:
            model_id: Model ID (MLflow model name)
            model_uri: MLflow model URI

        Returns:
            Model version number as string
        """
        result = mlflow.register_model(model_uri, model_id)
        version = result.version
        logger.info(f"Registered model {model_id} version {version}")
        return str(version)

    def _build_model(
        self,
        model_id: str,
        architecture: ArchitectureType,
        input_fields: list[str],
        output_fields: list[str],
        window_duration_seconds: int,
        lookback_steps: int,
        forecast_steps: int,
        hidden_size: int,
    ):
        """
        Build or load PyTorch model for incremental training.

        Tries to load the latest version from MLflow for incremental training.
        If no version exists, creates a new model from scratch.

        Args:
            model_id: Model ID (for loading from MLflow)
            architecture: Model architecture (LSTM or ANN)
            input_fields: List of input field names
            output_fields: List of output field names
            window_duration_seconds: Data window granularity in seconds
            lookback_steps: Lookback window size
            forecast_steps: Forecast window size
            hidden_size: Hidden layer size

        Returns:
            PyTorch model instance (loaded or new)

        Raises:
            ValueError: If architecture not supported
        """
        model_class = MODEL_REGISTRY.get(architecture)
        if not model_class:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Try to load latest model for incremental training
        try:
            client = MlflowClient()
            latest_versions = client.get_latest_versions(model_id, stages=["Production", "None"])

            if latest_versions:
                latest_version = max(latest_versions, key=lambda v: int(v.version))
                model_uri = f"models:/{model_id}/{latest_version.version}"

                logger.info(
                    f"Loading model {model_id} version {latest_version.version} "
                    f"for incremental training"
                )

                # Load the PyTorch model from MLflow
                loaded_model = mlflow.pytorch.load_model(model_uri)

                # Wrap it back in our model interface with configuration
                if architecture == ArchitectureType.LSTM:
                    model = model_class(
                        input_fields=input_fields,
                        output_fields=output_fields,
                        window_duration_seconds=window_duration_seconds,
                        lookback_steps=lookback_steps,
                        forecast_steps=forecast_steps,
                        hidden_size=hidden_size,
                        num_layers=2
                    )
                else:  # ANN
                    model = model_class(
                        input_fields=input_fields,
                        output_fields=output_fields,
                        window_duration_seconds=window_duration_seconds,
                        lookback_steps=lookback_steps,
                        forecast_steps=forecast_steps,
                        hidden_size=hidden_size
                    )
                model.model = loaded_model

                logger.info("Successfully loaded model for incremental training")
                return model

        except Exception as e:
            logger.info(f"No existing model found ({str(e)}), creating new model from scratch")

        # Create new model if loading failed or no versions exist
        if architecture == ArchitectureType.LSTM:
            model = model_class(
                input_fields=input_fields,
                output_fields=output_fields,
                window_duration_seconds=window_duration_seconds,
                lookback_steps=lookback_steps,
                forecast_steps=forecast_steps,
                hidden_size=hidden_size,
                num_layers=2
            )
        else:  # ANN
            model = model_class(
                input_fields=input_fields,
                output_fields=output_fields,
                window_duration_seconds=window_duration_seconds,
                lookback_steps=lookback_steps,
                forecast_steps=forecast_steps,
                hidden_size=hidden_size
            )
        logger.info(
            f"Built new {architecture.value} model with {len(input_fields)} inputs, "
            f"{len(output_fields)} outputs, lookback={lookback_steps}, forecast={forecast_steps}"
        )

        return model

    def _train_pytorch_model(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        job_id: str,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> float:
        """
        Train PyTorch model with cancellation support.

        Args:
            model: PyTorch model instance
            X_train: Training input data
            y_train: Training target data
            job_id: Training job ID (for cancellation checks)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            Final training loss

        Raises:
            InterruptedError: If training is cancelled
        """
        def status_callback(epoch, max_epochs, loss, **kwargs):
            """Callback for logging progress and checking cancellation."""
            # Log progress every 10 epochs
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{max_epochs}: loss={loss:.6f}")
                mlflow.log_metric("training_loss", loss, step=epoch)

            # Check if job was cancelled (check every epoch)
            job = self.get_job(job_id)
            if job and job.status == TrainingJobStatus.CANCELLED:
                logger.warning(f"Job {job_id} cancelled at epoch {epoch}")
                raise InterruptedError("Training cancelled by user")

        logger.info(f"Starting training: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")

        # Prepare data for training
        X_prepared = np.nan_to_num(np.array(X_train, dtype=np.float32))
        y_prepared = np.nan_to_num(np.array(y_train, dtype=np.float32))

        # Ensure proper dimensions
        if X_prepared.ndim == 2:
            X_prepared = X_prepared[:, np.newaxis, :]
        if y_prepared.ndim == 2:
            y_prepared = y_prepared[:, np.newaxis, :]

        # Flatten y for training: (batch, forecast_steps, features) -> (batch, forecast_steps * features)
        y_prepared = y_prepared.reshape(y_prepared.shape[0], -1)

        final_loss = model.train(
            X=X_prepared,
            y=y_prepared,
            max_epochs=epochs,
            status_callback=status_callback
        )

        logger.info(f"Training completed with final loss: {final_loss:.6f}")
        return final_loss
