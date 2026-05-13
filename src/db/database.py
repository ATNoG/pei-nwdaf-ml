"""Database configuration and session management."""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from src.core.config import settings

# Create SQLAlchemy engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base for models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get database session.

    Usage:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database session.

    Usage:
        with get_db_context() as db:
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables in the database."""
    from src.db.architecture_config import ArchitectureConfigDB  # noqa: F401

    Base.metadata.create_all(bind=engine)
    # Add new nullable columns to existing tables (idempotent)
    with engine.connect() as conn:
        for table, col in [
            ("model_configs", "default_cpu"),
            ("model_configs", "default_memory"),
            ("anomaly_model_configs", "default_cpu"),
            ("anomaly_model_configs", "default_memory"),
        ]:
            try:
                conn.execute(
                    __import__("sqlalchemy").text(
                        f"ALTER TABLE {table} ADD COLUMN {col} VARCHAR"
                    )
                )
                conn.commit()
            except Exception:
                conn.rollback()
