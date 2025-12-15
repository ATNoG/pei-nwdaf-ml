from src.config import EXCLUDED_FIELDS
from typing import Any

def extract_features( window: dict[str, Any], analytics_type: str):
    return {
        k: float(v) if v is not None else .0
        for k, v in window.items()
        if k not in EXCLUDED_FIELDS
        and not k.startswith(analytics_type + "_")
        and ( isinstance(v, (int, float)) or v is None)
    }
