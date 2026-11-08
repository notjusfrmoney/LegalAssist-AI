from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "LegalAssist AI"
    app_env: str = "development"
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-20b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    knowledge_base_dir: Path = Field(default=PROJECT_ROOT / "knowledge_base")
    top_k: int = 5
    similarity_threshold: float = 0.30
    max_history_messages: int = 10
    max_completion_tokens: int = Field(default=700, gt=0)
    groq_temperature: float = Field(default=0.2, ge=0, le=2)
    groq_timeout_seconds: float = Field(default=30.0, gt=0)
    allowed_origins_config: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001",
        validation_alias="ALLOWED_ORIGINS",
    )

    @field_validator("allowed_origins_config", mode="before")
    @classmethod
    def validate_allowed_origins_config(cls, value: object) -> str:
        if value in (None, ""):
            return "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001"
        if not isinstance(value, str):
            raise ValueError("ALLOWED_ORIGINS must be a comma-separated string or JSON list.")

        configured = value.strip()
        if configured.startswith("["):
            try:
                parsed = json.loads(configured)
            except json.JSONDecodeError as error:
                raise ValueError("ALLOWED_ORIGINS contains invalid JSON.") from error
            if not isinstance(parsed, list) or not all(
                isinstance(origin, str) and origin.strip() for origin in parsed
            ):
                raise ValueError("ALLOWED_ORIGINS JSON must be a list of non-empty strings.")
            origins = [origin.strip() for origin in parsed]
        else:
            origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
        if not origins:
            raise ValueError("ALLOWED_ORIGINS must include at least one origin.")
        return ",".join(origins)

    @property
    def allowed_origins(self) -> List[str]:
        return [origin.strip() for origin in self.allowed_origins_config.split(",")]

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
