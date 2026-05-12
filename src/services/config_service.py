"""Service for managing model configurations in PostgreSQL."""

from sqlalchemy.orm import Session

from src.db.model_config import ModelConfigDB
from src.schemas.model import ModelConfig


class MLConfigService:
    """Service for model configuration database operations."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, model_config: ModelConfigDB) -> None:
        """Create a new model configuration in the database."""
        self.db.add(model_config)
        self.db.commit()
        self.db.refresh(model_config)

    def get_config(self, model_id: str) -> ModelConfigDB | None:
        """Get model configuration by ID."""
        return self.db.query(ModelConfigDB).filter(ModelConfigDB.model_id == model_id).first()

    def list_all(self) -> list[ModelConfigDB]:
        """List all model configurations."""
        return self.db.query(ModelConfigDB).all()

    def delete_model(self, model_id: str) -> None:
        """Delete a model configuration from the database."""
        model_config = self.get_config(model_id)
        if not model_config:
            raise ValueError(f"Model '{model_id}' not found")

        self.db.delete(model_config)
        self.db.commit()

    def config_from_db(self, model_config: ModelConfigDB) -> ModelConfig:
        """Convert database model to ModelConfig schema."""
        return ModelConfig(
            architecture=model_config.architecture,
            input_fields=model_config.input_fields,
            output_fields=model_config.output_fields,
            window_duration_seconds=model_config.window_duration_seconds,
            lookback_steps=model_config.lookback_steps,
            forecast_steps=model_config.forecast_steps,
            hidden_size=model_config.hidden_size,
        )
