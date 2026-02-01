"""Database models for application data."""

from datetime import datetime

from sqlalchemy import JSON, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.database import Base


class ModelConfigDB(Base):
    """Database model for storing model configurations."""

    __tablename__ = "model_configs"

    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    architecture: Mapped[str] = mapped_column(String, nullable=False)
    input_fields: Mapped[list] = mapped_column(JSON, nullable=False)
    output_fields: Mapped[list] = mapped_column(JSON, nullable=False)
    window_duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    lookback_steps: Mapped[int] = mapped_column(Integer, nullable=False)
    forecast_steps: Mapped[int] = mapped_column(Integer, nullable=False)
    hidden_size: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)

    def __repr__(self) -> str:
        return f"<ModelConfigDB(model_id={self.model_id}, name={self.name}, architecture={self.architecture})>"
