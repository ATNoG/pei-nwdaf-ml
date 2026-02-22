import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional

from mlflow.tracking import MlflowClient

from src.schemas.expiry import ExpiryPolicy, ExpiryPolicyUpdate
from src.services.model_service import ModelService

logger = logging.getLogger(__name__)


class ExpiryService:
    """
    Manages the lifecycle of registered ML models in MLflow.

    Models that have not been used for longer than their effective TTL are
    automatically deleted from the registry.

    Effective TTL per model:
        base_ttl_days + min(use_count * tolerance_per_use_days, max_tolerance_days)
    """

    def __init__(self, ml_interface) -> None:
        self._ml_interface = ml_interface
        self._policy = ExpiryPolicy(
            base_ttl_days=int(os.getenv("EXPIRY_BASE_TTL_DAYS", "30")),
            tolerance_per_use_days=int(os.getenv("EXPIRY_TOLERANCE_PER_USE_DAYS", "1")),
            max_tolerance_days=int(os.getenv("EXPIRY_MAX_TOLERANCE_DAYS", "90")),
            sweep_interval_seconds=int(os.getenv("EXPIRY_SWEEP_INTERVAL_SECONDS", "3600")),
        )

    def get_policy(self) -> ExpiryPolicy:
        return self._policy

    def update_policy(self, update: ExpiryPolicyUpdate) -> ExpiryPolicy:
        """Apply a partial update to the current policy."""
        current = self._policy
        self._policy = ExpiryPolicy(
            base_ttl_days=update.base_ttl_days if update.base_ttl_days is not None else current.base_ttl_days,
            tolerance_per_use_days=update.tolerance_per_use_days if update.tolerance_per_use_days is not None else current.tolerance_per_use_days,
            max_tolerance_days=update.max_tolerance_days if update.max_tolerance_days is not None else current.max_tolerance_days,
            sweep_interval_seconds=update.sweep_interval_seconds if update.sweep_interval_seconds is not None else current.sweep_interval_seconds,
        )
        return self._policy

    def replace_policy(self, policy: ExpiryPolicy) -> ExpiryPolicy:
        self._policy = policy
        return self._policy

    def run_sweep(self) -> int:
        """
        Scan all registered MLflow models and delete those whose effective TTL has expired.

        Models tagged with expiry_protected=true are skipped.
        Last-used time falls back to last_trained, then creation_timestamp if tags are absent.

        Returns the number of models deleted.
        """
        if not self._ml_interface.is_mlflow_connected():
            logger.warning("MLflow not connected — skipping expiry sweep")
            return 0

        client = MlflowClient()
        model_service = ModelService(self._ml_interface)
        policy = self._policy
        now = datetime.now(timezone.utc).timestamp()
        deleted = 0

        try:
            registered_models = client.search_registered_models()
        except Exception as e:
            logger.error(f"Failed to list models for expiry sweep: {e}")
            return 0

        for rm in registered_models or []:
            try:
                tags = rm.tags or {}

                if tags.get("expiry_protected") == "true":
                    logger.debug(f"Skipping protected model: {rm.name}")
                    continue

                # resolve last-used timestamp
                last_used_str = tags.get("last_used_at") or tags.get("last_trained")
                if last_used_str:
                    last_used = float(last_used_str)
                else:
                    # creation_timestamp is in milliseconds
                    last_used = (rm.creation_timestamp or 0) / 1000

                use_count = int(tags.get("use_count", "0"))
                tolerance = min(use_count * policy.tolerance_per_use_days, policy.max_tolerance_days)
                effective_ttl_seconds = (policy.base_ttl_days + tolerance) * 86400
                expires_at = last_used + effective_ttl_seconds

                if now > expires_at:
                    logger.info(
                        f"Expiring model '{rm.name}' "
                        f"(last_used={datetime.fromtimestamp(last_used, tz=timezone.utc).isoformat()}, "
                        f"effective_ttl={policy.base_ttl_days + tolerance}d)"
                    )
                    model_service.delete_model_instance(rm.name)
                    deleted += 1

            except Exception as e:
                logger.warning(f"Error evaluating model '{rm.name}' during sweep: {e}")
                continue

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
        # first sweep runs immediately on startup
        while not self._stop_event.is_set():
            try:
                self._service.run_sweep()
            except Exception as e:
                logger.error(f"Expiry sweep failed: {e}", exc_info=True)
            interval = self._service.get_policy().sweep_interval_seconds
            self._stop_event.wait(timeout=interval)
