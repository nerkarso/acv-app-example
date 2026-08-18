from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    vision_provider: str = "none"  # claude | none
    anthropic_api_key: str | None = None
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_claude_model: str = "claude-haiku-4-5-20251001"

    confidence_auto_accept: float = 0.90
    confidence_escalate_min: float = 0.70
    paddle_ocr_confidence_threshold: float = 0.85

    database_path: str = "data/local.db"
    data_dir: str = "data"
    uploads_dir: str = "data/uploads"
    processed_dir: str = "data/processed"
    crops_dir: str = "data/crops"
    exports_dir: str = "data/exports"

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
