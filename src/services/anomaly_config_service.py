"""Service for managing anomaly model configurations in PostgreSQL."""

from sqlalchemy.orm import Session

from src.db.anomaly_model_config import AnomalyModelConfigDB
from src.schemas.anomaly import AnomalyModelConfig


class AnomalyConfigService:
    """Service for anomaly model configuration database operations."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, model_config: AnomalyModelConfigDB) -> None:
        """Create a new anomaly model configuration in the database."""
        self.db.add(model_config)
        self.db.commit()
        self.db.refresh(model_config)

    def get_config(self, model_id: str) -> AnomalyModelConfigDB | None:
        """Get anomaly model configuration by ID."""
        return (
            self.db.query(AnomalyModelConfigDB)
            .filter(AnomalyModelConfigDB.model_id == model_id)
            .first()
        )

    def list_all(self) -> list[AnomalyModelConfigDB]:
        """List all anomaly model configurations."""
        return self.db.query(AnomalyModelConfigDB).all()

    def delete_model(self, model_id: str) -> None:
        """Delete an anomaly model configuration from the database."""
        model_config = self.get_config(model_id)
        if not model_config:
            raise ValueError(f"Anomaly model '{model_id}' not found")
        self.db.delete(model_config)
        self.db.commit()

    def update_threshold(self, model_id: str, threshold_value: float) -> None:
        """Update the computed threshold value after training."""
        model_config = self.get_config(model_id)
        if not model_config:
            raise ValueError(f"Anomaly model '{model_id}' not found")
        model_config.threshold_value = threshold_value
        self.db.commit()

    def config_from_db(self, model_config: AnomalyModelConfigDB) -> AnomalyModelConfig:
        """Convert database model to AnomalyModelConfig schema."""
        return AnomalyModelConfig(
            input_fields=model_config.input_fields,
            window_duration_seconds=model_config.window_duration_seconds,
            hidden_size=model_config.hidden_size,
            threshold_percentile=model_config.threshold_percentile,
        )
