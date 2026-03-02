"""Database model for score history tracking."""

from datetime import datetime

from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.database import Base


class ScoreHistoryDB(Base):
    """Database model for tracking performance score measurements over time."""

    __tablename__ = "score_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("model_configs.model_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    field_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    metric: Mapped[str] = mapped_column(String, nullable=False)  # "rmse" | "mae" | "mape" | "r2"
    measured_at: Mapped[datetime] = mapped_column(nullable=False)
    trigger: Mapped[str] = mapped_column(String, nullable=False)  # "evaluate" | "monitor" | "auto_monitor"

    def __repr__(self) -> str:
        return (
            f"<ScoreHistoryDB(id={self.id}, model_id={self.model_id}, "
            f"field={self.field_name}, {self.metric}={self.score:.4f}, trigger={self.trigger})>"
        )
