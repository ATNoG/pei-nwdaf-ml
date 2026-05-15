import time
from typing import Any

_cache: dict[str, tuple[Any, float]] = {}
TTL = 60.0


def get(key: str) -> Any | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    value, expires_at = entry
    if time.monotonic() > expires_at:
        _cache.pop(key, None)
        return None
    return value


def set(key: str, value: Any, ttl: float = TTL) -> None:
    _cache[key] = (value, time.monotonic() + ttl)


def invalidate_field(output_field: str) -> None:
    for k in [k for k in _cache if output_field in k]:
        _cache.pop(k, None)
