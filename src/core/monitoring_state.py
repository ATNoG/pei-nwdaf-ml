"""Shared in-memory state for the per-field monitoring state machine.

Populated by the background _monitoring_loop() in main.py.
Read by the performance router to expose current state via API.
"""

from datetime import datetime

# field → "monitoring" | "retraining" | "evaluating"
_field_states: dict[str, str] = {}

# field → list of training job IDs being tracked during retraining
_field_jobs: dict[str, list[str]] = {}

# field → timestamp of last completed monitoring cycle
_last_checked: dict[str, datetime] = {}


def get_field_state(field_name: str) -> str:
    return _field_states.get(field_name, "monitoring")


def set_field_state(field_name: str, state: str) -> None:
    _field_states[field_name] = state


def get_field_jobs(field_name: str) -> list[str]:
    return _field_jobs.get(field_name, [])


def set_field_jobs(field_name: str, job_ids: list[str]) -> None:
    _field_jobs[field_name] = job_ids


def clear_field_jobs(field_name: str) -> None:
    _field_jobs.pop(field_name, None)


def set_last_checked(field_name: str, ts: datetime) -> None:
    _last_checked[field_name] = ts


def get_last_checked(field_name: str) -> datetime | None:
    return _last_checked.get(field_name)
