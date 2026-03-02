"""Database model for the model expiry audit log."""

from datetime import datetime, timezone

from sqlalchemy import Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.database import Base


class ExpiryAuditDB(Base):
    """Immutable audit record written each time a model is expired by the sweep."""

    __tablename__ = "nwdaf_expiry_audit"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    expired_at: Mapped[datetime] = mapped_column(
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    last_used_at: Mapped[datetime | None] = mapped_column(nullable=True)
    effective_ttl_days: Mapped[int] = mapped_column(Integer, nullable=False)
    policy_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<ExpiryAuditDB(id={self.id}, model_id={self.model_id}, "
            f"model_name={self.model_name}, expired_at={self.expired_at})>"
        )
