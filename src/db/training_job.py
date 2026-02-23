"""Database model for training jobs."""

from datetime import datetime
from sqlalchemy import String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from src.db.database import Base


class TrainingJobDB(Base):
    """Database model for tracking training jobs."""

    __tablename__ = "training_jobs"

    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_id: Mapped[str] = mapped_column(String, ForeignKey("model_configs.model_id", ondelete="CASCADE"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    lookback_seconds: Mapped[int] = mapped_column(nullable=False)
    mlflow_run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)

    def __repr__(self) -> str:
        return f"<TrainingJobDB(job_id={self.job_id}, model_id={self.model_id}, status={self.status})>"
