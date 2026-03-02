"""TTL-based model expiry service with PostgreSQL-backed policy and audit log."""

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional

from mlflow.tracking import MlflowClient

from src.db.database import get_db_context
from src.db.expiry_audit import ExpiryAuditDB
from src.db.expiry_policy import ExpiryPolicyDB
from src.db.model_config import ModelConfigDB
from src.schemas.expiry import ExpiryPolicy, ExpiryPolicyUpdate

logger = logging.getLogger(__name__)

_DEFAULT_POLICY = ExpiryPolicy(
    base_ttl_days=int(os.getenv("EXPIRY_BASE_TTL_DAYS", "30")),
    tolerance_per_use_days=int(os.getenv("EXPIRY_TOLERANCE_PER_USE_DAYS", "1")),
    max_tolerance_days=int(os.getenv("EXPIRY_MAX_TOLERANCE_DAYS", "90")),
    sweep_interval_seconds=int(os.getenv("EXPIRY_SWEEP_INTERVAL_SECONDS", "3600")),
)


def _load_policy_from_db() -> Optional[ExpiryPolicy]:
    """Read the latest policy row from the database. Returns None if no row exists."""
    try:
        with get_db_context() as db:
            row = db.query(ExpiryPolicyDB).order_by(ExpiryPolicyDB.id.desc()).first()
            if row is None:
                return None
            return ExpiryPolicy(
                base_ttl_days=row.base_ttl_days,
                tolerance_per_use_days=row.tolerance_per_use_days,
                max_tolerance_days=row.max_tolerance_days,
                sweep_interval_seconds=row.sweep_interval_seconds,
            )
    except Exception as e:
        logger.warning(f"Could not load expiry policy from DB: {e}")
        return None


def _save_policy_to_db(policy: ExpiryPolicy) -> None:
    """Persist a new policy row (append-only; latest id wins on read)."""
    try:
        with get_db_context() as db:
            row = ExpiryPolicyDB(
                base_ttl_days=policy.base_ttl_days,
                tolerance_per_use_days=policy.tolerance_per_use_days,
                max_tolerance_days=policy.max_tolerance_days,
                sweep_interval_seconds=policy.sweep_interval_seconds,
                updated_at=datetime.now(timezone.utc),
            )
            db.add(row)
            db.commit()
    except Exception as e:
        logger.error(f"Could not save expiry policy to DB: {e}")


class ExpiryService:
    """
    Manages the lifecycle of registered ML models.

    Models that have not been trained (or created) for longer than their
    effective TTL are automatically deleted from both MLflow and PostgreSQL.

    Effective TTL per model:
        base_ttl_days + min(use_count * tolerance_per_use_days, max_tolerance_days)

    Note: use_count is reserved for future inference-tracking integration.
    Currently 0, so only base_ttl_days applies.
    """

    def __init__(self) -> None:
        loaded = _load_policy_from_db()
        if loaded is None:
            logger.info("No expiry policy in DB — seeding from env var defaults")
            self._policy = _DEFAULT_POLICY
            _save_policy_to_db(self._policy)
        else:
            logger.info("Loaded expiry policy from DB")
            self._policy = loaded

    def get_policy(self) -> ExpiryPolicy:
        return self._policy

    def update_policy(self, update: ExpiryPolicyUpdate) -> ExpiryPolicy:
        """Apply a partial update to the current policy and persist it."""
        current = self._policy
        self._policy = ExpiryPolicy(
            base_ttl_days=update.base_ttl_days if update.base_ttl_days is not None else current.base_ttl_days,
            tolerance_per_use_days=update.tolerance_per_use_days if update.tolerance_per_use_days is not None else current.tolerance_per_use_days,
            max_tolerance_days=update.max_tolerance_days if update.max_tolerance_days is not None else current.max_tolerance_days,
            sweep_interval_seconds=update.sweep_interval_seconds if update.sweep_interval_seconds is not None else current.sweep_interval_seconds,
        )
        _save_policy_to_db(self._policy)
        return self._policy

    def replace_policy(self, policy: ExpiryPolicy) -> ExpiryPolicy:
        """Fully replace the current policy and persist it."""
        self._policy = policy
        _save_policy_to_db(self._policy)
        return self._policy

    def run_sweep(self) -> int:
        """
        Scan all registered models and delete those whose effective TTL has expired.

        Uses last_trained_at (from latest MLflow run) or created_at as the baseline.
        Writes an audit record for every deleted model.

        Returns the number of models deleted.
        """
        policy = self._policy
        now = datetime.now(timezone.utc).timestamp()
        deleted = 0

        try:
            client = MlflowClient()
            with get_db_context() as db:
                model_configs = db.query(ModelConfigDB).all()

                for db_config in model_configs:
                    try:
                        model_id = db_config.model_id

                        # Resolve last-activity timestamp from MLflow
                        last_activity: Optional[datetime] = None
                        try:
                            rm = client.get_registered_model(model_id)
                            if rm.latest_versions:
                                latest_mv = rm.latest_versions[0]
                                if latest_mv.run_id:
                                    run = client.get_run(latest_mv.run_id)
                                    if run.info.end_time:
                                        last_activity = datetime.fromtimestamp(
                                            run.info.end_time / 1000, tz=timezone.utc
                                        )
                        except Exception:
                            pass

                        if last_activity is None:
                            created = db_config.created_at
                            if created.tzinfo is None:
                                created = created.replace(tzinfo=timezone.utc)
                            last_activity = created

                        use_count = 0  # Future: set from inference tracking tags
                        tolerance = min(
                            use_count * policy.tolerance_per_use_days,
                            policy.max_tolerance_days,
                        )
                        effective_ttl_days = policy.base_ttl_days + tolerance
                        expires_at = last_activity.timestamp() + effective_ttl_days * 86400

                        if now <= expires_at:
                            continue

                        logger.info(
                            f"Expiring model '{db_config.name}' (id={model_id}, "
                            f"last_activity={last_activity.isoformat()}, "
                            f"effective_ttl={effective_ttl_days}d)"
                        )

                        # Delete from MLflow (best-effort)
                        try:
                            client.delete_registered_model(model_id)
                        except Exception as e:
                            logger.warning(f"Failed to delete MLflow model '{model_id}': {e}")

                        # Write audit record
                        audit = ExpiryAuditDB(
                            model_id=model_id,
                            model_name=db_config.name,
                            expired_at=datetime.now(timezone.utc),
                            last_used_at=last_activity,
                            effective_ttl_days=effective_ttl_days,
                            policy_snapshot=policy.model_dump(),
                        )
                        db.add(audit)

                        # Delete from PostgreSQL
                        db.delete(db_config)
                        db.commit()
                        deleted += 1

                    except Exception as e:
                        logger.warning(f"Error evaluating model '{db_config.model_id}': {e}")
                        continue

        except Exception as e:
            logger.error(f"Expiry sweep failed: {e}", exc_info=True)

        logger.info(f"Expiry sweep complete: {deleted} model(s) deleted")
        return deleted


class ExpiryScheduler:
    """Background thread that runs ExpiryService.run_sweep() on a fixed interval."""

    def __init__(self, expiry_service: ExpiryService) -> None:
        self._service = expiry_service
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="expiry-scheduler-thread",
        )
        self._thread.start()
        logger.info("ExpiryScheduler started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("ExpiryScheduler stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._service.run_sweep()
            except Exception as e:
                logger.error(f"Expiry sweep failed: {e}", exc_info=True)
            interval = self._service.get_policy().sweep_interval_seconds
            self._stop_event.wait(timeout=interval)
