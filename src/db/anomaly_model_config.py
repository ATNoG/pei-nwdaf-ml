"""Database model for anomaly detection model configurations."""

from datetime import datetime

from sqlalchemy import JSON, Boolean, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.database import Base


class AnomalyModelConfigDB(Base):
    """Database model for storing anomaly detection model configurations."""

    __tablename__ = "anomaly_model_configs"

    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    input_fields: Mapped[list] = mapped_column(JSON, nullable=False)
    window_duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    hidden_size: Mapped[int] = mapped_column(Integer, nullable=False)
    threshold_percentile: Mapped[float] = mapped_column(Float, nullable=False)
    threshold_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    is_training: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    def __repr__(self) -> str:
        return f"<AnomalyModelConfigDB(model_id={self.model_id}, name={self.name})>"
