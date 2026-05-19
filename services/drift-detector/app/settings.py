"""Drift detector settings loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DRIFT_", case_sensitive=False)

    # Prometheus Push Gateway
    pushgateway_url: str = (
        "http://prometheus-pushgateway.monitoring.svc.cluster.local:9091"
    )
    job_name: str = "moiraweave_drift_detector"

    # Qdrant (embedding snapshots)
    qdrant_url: str = "http://qdrant.moiraweave.svc.cluster.local:6333"
    qdrant_collection: str = "embeddings"
    # Number of recent vectors to sample for current window
    qdrant_sample_size: int = 500
    # Number of reference vectors to compare against
    qdrant_reference_size: int = 500

    # Drift thresholds (used to set Prometheus gauge labels)
    drift_threshold: float = 0.5
