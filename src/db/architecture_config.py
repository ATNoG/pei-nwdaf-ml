from datetime import datetime

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.database import Base


class ArchitectureConfigDB(Base):
    __tablename__ = "architecture_configs"

    architecture_id: Mapped[str] = mapped_column(String, primary_key=True)
    uploaded_by: Mapped[str] = mapped_column(String, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(nullable=False)

    def __repr__(self) -> str:
        return f"<ArchitectureConfigDB(architecture_id={self.architecture_id}, uploaded_by={self.uploaded_by})>"
