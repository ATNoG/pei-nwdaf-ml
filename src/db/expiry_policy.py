"""Database model for the model expiry policy."""

from datetime import datetime, timezone

from sqlalchemy import Integer
from sqlalchemy.orm import Mapped, mapped_column

from src.db.database import Base


class ExpiryPolicyDB(Base):
    """Persisted expiry policy. Append-only: latest row by id wins."""

    __tablename__ = "nwdaf_expiry_policy"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    base_ttl_days: Mapped[int] = mapped_column(Integer, nullable=False)
    tolerance_per_use_days: Mapped[int] = mapped_column(Integer, nullable=False)
    max_tolerance_days: Mapped[int] = mapped_column(Integer, nullable=False)
    sweep_interval_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return (
            f"<ExpiryPolicyDB(id={self.id}, base_ttl_days={self.base_ttl_days}, "
            f"sweep_interval_seconds={self.sweep_interval_seconds})>"
        )
