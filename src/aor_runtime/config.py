from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    workspace_root: Path = Field(default_factory=lambda: Path.cwd())
    prompts_root: Path = Field(default_factory=lambda: Path.cwd() / "prompts")
    run_store_path: Path = Field(default_factory=lambda: Path.cwd() / "artifacts" / "runtime.db")
    sql_database_url: str | None = Field(default_factory=lambda: os.getenv("AOR_SQL_DATABASE_URL") or None)
    sql_databases_json: str | None = Field(default_factory=lambda: os.getenv("AOR_SQL_DATABASES") or None)
    sql_default_database: str | None = Field(default_factory=lambda: os.getenv("AOR_SQL_DEFAULT_DATABASE") or None)
    sql_row_limit: int = Field(default_factory=lambda: int(os.getenv("AOR_SQL_ROW_LIMIT", "500")))
    sql_timeout_seconds: int = Field(default_factory=lambda: int(float(os.getenv("AOR_SQL_TIMEOUT_SECONDS", "10"))))
    llm_base_url: str = Field(default_factory=lambda: os.getenv("AOR_LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    llm_api_key: str = Field(default_factory=lambda: os.getenv("AOR_LLM_API_KEY", "local"))
    default_model: str = Field(default_factory=lambda: os.getenv("AOR_DEFAULT_MODEL", "qwen-coder"))
    default_temperature: float = Field(default_factory=lambda: float(os.getenv("AOR_DEFAULT_TEMPERATURE", "0.1")))
    llm_timeout_seconds: float = Field(default_factory=lambda: float(os.getenv("AOR_LLM_TIMEOUT_SECONDS", "120")))
    allow_destructive_shell: bool = Field(
        default_factory=lambda: os.getenv("AOR_ALLOW_DESTRUCTIVE_SHELL", "0").strip().lower() in {"1", "true", "yes", "on"}
    )
    max_plan_retries: int = Field(default_factory=lambda: int(os.getenv("AOR_MAX_PLAN_RETRIES", "2")))


def get_settings() -> Settings:
    settings = Settings()
    settings.run_store_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
