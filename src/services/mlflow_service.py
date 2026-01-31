"""MLflow service for managing models."""

import json
from datetime import datetime
from uuid import uuid4

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.schemas.model import ArchitectureType, ModelConfig, ModelDetail, ModelSummary


class MLflowService:
    """MLflow service for managing model registry and metadata."""

    def __init__(self, mlflow_client: MlflowClient):
        self.client = mlflow_client

    def create_model(self, name: str, config: ModelConfig) -> ModelDetail:
        """
        Create a new model in MLflow registry.

        Generates a unique ID and stores the model config as tags.
        """
        # Check if name already exists
        existing = self.client.search_registered_models(filter_string=f"name='{name}'")
        if existing:
            raise ValueError(f"Model with name '{name}' already exists")

        # Generate unique ID
        model_id = str(uuid4())

        # Create registered model with ID as MLflow name
        registered_model = self.client.create_registered_model(model_id)

        # Store name and config as tags
        tags = {
            "name": name,
            "config:architecture": config.architecture.value,
            "config:input_fields": json.dumps(config.input_fields),
            "config:output_fields": json.dumps(config.output_fields),
            "config:window_duration_seconds": str(config.window_duration_seconds),
            "config:lookback_steps": str(config.lookback_steps),
            "config:forecast_steps": str(config.forecast_steps),
            "config:hidden_size": str(config.hidden_size),
        }

        for key, value in tags.items():
            self.client.set_registered_model_tag(model_id, key, value)

        return ModelDetail(
            id=model_id,
            name=name,
            config=config,
            created_at=datetime.fromtimestamp(registered_model.creation_timestamp / 1000),
            latest_version=None,
            last_trained_at=None,
            mlflow_run_id=None,
            training_loss=None,
        )

    def get_model(self, model_id: str) -> ModelDetail:
        """Get model details by ID."""
        try:
            registered_model = self.client.get_registered_model(model_id)
        except MlflowException:
            raise ValueError(f"Model '{model_id}' not found")

        # Reconstruct config from tags
        config = self._get_model_config(model_id)

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
                    training_loss = run.data.metrics.get("loss")
                except Exception:
                    pass

        name = registered_model.tags.get("name", model_id)

        return ModelDetail(
            id=model_id,
            name=name,
            config=config,
            created_at=datetime.fromtimestamp(registered_model.creation_timestamp / 1000),
            latest_version=latest_version,
            last_trained_at=last_trained_at,
            mlflow_run_id=mlflow_run_id,
            training_loss=training_loss,
        )

    def list_models(self) -> list[ModelSummary]:
        """List all registered models."""
        registered_models = self.client.search_registered_models()

        summaries = []
        for rm in registered_models:
            latest_version = None
            if rm.latest_versions:
                latest_version = int(rm.latest_versions[0].version)


            # Get name and architecture from tags
            name = rm.tags.get("name", rm.name)
            architecture_str = rm.tags.get("config:architecture", "ann")
            architecture = ArchitectureType(architecture_str)

            summaries.append(
                ModelSummary(
                    id=rm.name,  # rm.name is the UUID we generated
                    name=name,
                    architecture=architecture,
                    created_at=datetime.fromtimestamp(rm.creation_timestamp / 1000),
                    latest_version=latest_version,
                )
            )

        return summaries

    def delete_model(self, model_id: str) -> None:
        """Delete a model from the registry."""
        try:
            self.client.delete_registered_model(model_id)
        except MlflowException:
            raise ValueError(f"Model '{model_id}' not found")

    def _get_model_config(self, model_id: str) -> ModelConfig:
        """Reconstruct ModelConfig from MLflow tags."""
        rm = self.client.get_registered_model(model_id)
        tags = rm.tags

        return ModelConfig(
            architecture=ArchitectureType(tags.get("config:architecture", "ann")),
            input_fields=json.loads(tags.get("config:input_fields", "[]")),
            output_fields=json.loads(tags.get("config:output_fields", "[]")),
            window_duration_seconds=int(tags.get("config:window_duration_seconds", "60")),
            lookback_steps=int(tags.get("config:lookback_steps", "30")),
            forecast_steps=int(tags.get("config:forecast_steps", "5")),
            hidden_size=int(tags.get("config:hidden_size", "32")),
        )
