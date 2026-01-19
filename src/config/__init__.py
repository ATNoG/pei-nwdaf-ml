"""Configuration module for ML inference component"""
import os

WINDOW_DURATION = int(os.getenv("WINDOW_DURATION", "60"))  # seconds

EXCLUDED_FIELDS = {
    "window_start_time",
    "window_end_time",
    "window_duration_seconds",
    "network",
    "sample_count",
}
