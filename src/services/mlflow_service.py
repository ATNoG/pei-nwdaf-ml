"""MLflow service for managing models."""

from datetime import datetime
from uuid import uuid4

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from src.services.config_service import MLConfigService
from src.db.model_config import ModelConfigDB
from src.schemas.model import ArchitectureType, ModelConfig, ModelDetail, ModelSummary


class MLflowService:
    """MLflow service for managing model registry and metadata."""

    def __init__(self, mlflow_client: MlflowClient, ml_config_service: MLConfigService):
        self.client = mlflow_client
        self.ml_config_service = ml_config_service

    def create_model(self, name: str, config: ModelConfig) -> ModelDetail:
        """
        Create a new model in MLflow registry and store config in PostgreSQL.

        Generates a unique ID, creates MLflow registered model, and stores config in DB.
        """
        # Generate unique ID
        model_id = str(uuid4())
        created_at = datetime.now()

        try:
            # Create registered model in MLflow
            self.client.create_registered_model(model_id)

            # Store only name tag in MLflow for discoverability
            self.client.set_registered_model_tag(model_id, "name", name)

            # Store full config in PostgreSQL
            model_config = ModelConfigDB(
                model_id=model_id,
                name=name,
                architecture=config.architecture.value,
                input_fields=config.input_fields,
                output_fields=config.output_fields,
                window_duration_seconds=config.window_duration_seconds,
                lookback_steps=config.lookback_steps,
                forecast_steps=config.forecast_steps,
                hidden_size=config.hidden_size,
                created_at=created_at,
            )
            self.ml_config_service.create(model_config)

            return ModelDetail(
                id=model_id,
                name=name,
                config=config,
                created_at=created_at,
                latest_version=None,
                last_trained_at=None,
                mlflow_run_id=None,
                training_loss=None,
            )
        except Exception:
            # Clean up MLflow model if DB insert failed
            try:
                self.client.delete_registered_model(model_id)
            except Exception:
                pass
            raise

    def get_model(self, model_id: str) -> ModelDetail:
        """Get model details by ID from PostgreSQL and MLflow."""
        # Get config from PostgreSQL
        model_config = self.ml_config_service.get_config(model_id)
        if not model_config:
            raise ValueError(f"Model '{model_id}' not found")

        # Get MLflow metadata
        try:
            registered_model = self.client.get_registered_model(model_id)
        except MlflowException:
            raise ValueError(f"Model '{model_id}' not found in MLflow")

        # Reconstruct ModelConfig from database
        config = self.ml_config_service.config_from_db(model_config)

        # Get latest version info if it exists
        latest_version = None
        last_trained_at = None
        mlflow_run_id = None
        training_loss = None

        if registered_model.latest_versions:
            latest_mv = registered_model.latest_versions[0]
            latest_version = int(latest_mv.version)

            # Get run info if available
            if latest_mv.run_id:
                try:
                    run = self.client.get_run(latest_mv.run_id)
                    last_trained_at = datetime.fromtimestamp(run.info.end_time / 1000)
                    mlflow_run_id = latest_mv.run_id
                    training_loss = run.data.metrics.get("final_loss")
                except Exception:
                    pass

        return ModelDetail(
            id=model_id,
            name=model_config.name,
            config=config,
            created_at=model_config.created_at,
            latest_version=latest_version,
            last_trained_at=last_trained_at,
            mlflow_run_id=mlflow_run_id,
            training_loss=training_loss,
        )

    def list_models(self) -> list[ModelSummary]:
        """List all registered models from PostgreSQL and MLflow."""
        # Get all configs from PostgreSQL
        model_configs = self.ml_config_service.list_all()

        summaries = []
        for model_config in model_configs:
            # Get MLflow metadata for version info
            latest_version = None
            try:
                rm = self.client.get_registered_model(model_config.model_id)
                if rm.latest_versions:
                    latest_version = int(rm.latest_versions[0].version)
            except MlflowException:
                # Model exists in DB but not in MLflow (inconsistent state)
                pass

            summaries.append(
                ModelSummary(
                    id=model_config.model_id,
                    name=model_config.name,
                    architecture=ArchitectureType(model_config.architecture),
                    created_at=model_config.created_at,
                    latest_version=latest_version,
                )
            )

        return summaries

    def delete_model(self, model_id: str) -> None:
        """Delete a model from both PostgreSQL and MLflow registry."""
        # Delete from PostgreSQL
        self.ml_config_service.delete_model(model_id)

        # Delete from MLflow
        try:
            self.client.delete_registered_model(model_id)
        except MlflowException:
            # Model exists in DB but not in MLflow - already deleted from DB
            pass
