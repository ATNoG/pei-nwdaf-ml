"""Service for managing model configurations in PostgreSQL."""

from sqlalchemy.orm import Session

from src.db.model_config import ModelConfigDB
from src.schemas.model import ArchitectureType, ModelConfig


class MLConfigService:
    """Service for model configuration database operations."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, db_config: ModelConfigDB) -> None:
        """Create a new model configuration in the database."""
        self.db.add(db_config)
        self.db.commit()
        self.db.refresh(db_config)

    def get_config(self, model_id: str) -> ModelConfigDB | None:
        """Get model configuration by ID."""
        return self.db.query(ModelConfigDB).filter(ModelConfigDB.model_id == model_id).first()

    def list_all(self) -> list[ModelConfigDB]:
        """List all model configurations."""
        return self.db.query(ModelConfigDB).all()

    def delete_model(self, model_id: str) -> None:
        """Delete a model configuration from the database."""
        db_config = self.get_config(model_id)
        if not db_config:
            raise ValueError(f"Model '{model_id}' not found")

        self.db.delete(db_config)
        self.db.commit()

    def config_from_db(self, db_config: ModelConfigDB) -> ModelConfig:
        """Convert database model to ModelConfig schema."""
        return ModelConfig(
            architecture=ArchitectureType(db_config.architecture),
            input_fields=db_config.input_fields,
            output_fields=db_config.output_fields,
            window_duration_seconds=db_config.window_duration_seconds,
            lookback_steps=db_config.lookback_steps,
            forecast_steps=db_config.forecast_steps,
            hidden_size=db_config.hidden_size,
        )
