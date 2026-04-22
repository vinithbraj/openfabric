from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    workspace_root: Path = Field(default_factory=lambda: Path.cwd())
    prompts_root: Path = Field(default_factory=lambda: Path.cwd() / "prompts")
    run_store_path: Path = Field(default_factory=lambda: Path.cwd() / "artifacts" / "runtime.db")
    llm_base_url: str = Field(default_factory=lambda: os.getenv("AOR_LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    llm_api_key: str = Field(default_factory=lambda: os.getenv("AOR_LLM_API_KEY", "local"))
    default_model: str = Field(default_factory=lambda: os.getenv("AOR_DEFAULT_MODEL", "qwen-coder"))
    default_temperature: float = Field(default_factory=lambda: float(os.getenv("AOR_DEFAULT_TEMPERATURE", "0.1")))
    llm_timeout_seconds: float = Field(default_factory=lambda: float(os.getenv("AOR_LLM_TIMEOUT_SECONDS", "120")))
    allow_destructive_shell: bool = Field(
        default_factory=lambda: os.getenv("AOR_ALLOW_DESTRUCTIVE_SHELL", "0").strip().lower() in {"1", "true", "yes", "on"}
    )
    max_agent_iterations: int = Field(default_factory=lambda: int(os.getenv("AOR_MAX_AGENT_ITERATIONS", "4")))
    max_node_retries: int = Field(default_factory=lambda: int(os.getenv("AOR_MAX_NODE_RETRIES", "2")))


def get_settings() -> Settings:
    settings = Settings()
    settings.run_store_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
