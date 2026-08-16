"""Central configuration, loaded from environment variables / .env."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Vision escalation provider
    vision_provider: str = "none"  # claude | local | none
    anthropic_api_key: str | None = None
    local_vlm_endpoint: str | None = None

    # Confidence tiers for auto-accept / escalation / manual review
    confidence_auto_accept: float = 0.90
    confidence_escalate_min: float = 0.70
    paddle_ocr_confidence_threshold: float = 0.85

    # Storage
    database_path: str = "data/local.db"
    data_dir: str = "data"
    uploads_dir: str = "data/uploads"
    processed_dir: str = "data/processed"
    crops_dir: str = "data/crops"
    exports_dir: str = "data/exports"

    # Logging
    log_level: str = "INFO"

    @property
    def database_url(self) -> str:
        db_path = Path(self.database_path)
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path
        return f"sqlite:///{db_path}"

    def resolved_dir(self, name: str) -> Path:
        p = Path(getattr(self, name))
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def uploads_path(self) -> Path:
        return self.resolved_dir("uploads_dir")

    @property
    def processed_path(self) -> Path:
        return self.resolved_dir("processed_dir")

    @property
    def crops_path(self) -> Path:
        return self.resolved_dir("crops_dir")

    @property
    def exports_path(self) -> Path:
        return self.resolved_dir("exports_dir")


settings = Settings()
