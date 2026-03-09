"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=True, extra="ignore"
    )

    # API Configuration
    API_HOST: str
    API_PORT: int
    LOG_LEVEL: str

    # MLflow Configuration
    MLFLOW_TRACKING_URI: str
    MLFLOW_S3_ENDPOINT_URL: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_DEFAULT_REGION: str = "us-east-1"

    # Data Storage API
    DATA_STORAGE_API_URL: str
    DATA_STORAGE_EXAMPLE_ENDPOINT: str
    DATA_STORAGE_EXCLUDED_FIELDS: str
    DATA_STORAGE_DATA_ENDPOINT: str
    DATA_STORAGE_CELL_ENDPOINT: str

    # Database Configuration
    DATABASE_URL: str

    # Auto-monitoring
    MONITORING_ENABLED: bool = True
    MONITORING_INTERVAL_SECONDS: int = 300
    MONITORING_TRIGGER_MODE: str = "time"  # "time" | "data" | "both"
    MONITORING_DEGRADATION_FACTOR: float = 1.5

    KAFKA_HOST: str = "kafka"
    KAFKA_PORT: str = "9092"
    KAFKA_INPUT_TOPIC: str = "network.data.processed"
    KAFKA_OUTPUT_TOPIC: str = "network.ml.results"
    KAFKA_ENABLED: bool = True


settings = Settings()
