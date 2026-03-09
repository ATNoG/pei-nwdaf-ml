"""Database model for anomaly detection training jobs."""

from datetime import datetime

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.database import Base


class AnomalyTrainingJobDB(Base):
    """Database model for tracking anomaly detection training jobs."""

    __tablename__ = "anomaly_training_jobs"

    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("anomaly_model_configs.model_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    lookback_seconds: Mapped[int] = mapped_column(nullable=False)
    mlflow_run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)

    def __repr__(self) -> str:
        return f"<AnomalyTrainingJobDB(job_id={self.job_id}, model_id={self.model_id}, status={self.status})>"
