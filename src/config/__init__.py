"""Configuration module for ML inference component"""
import os


WINDOW_DURATION = int(os.getenv("WINDOW_DURATION", "60"))  # seconds
