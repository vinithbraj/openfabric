from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class Settings(BaseModel):
    node_name: str = Field(default_factory=lambda: os.getenv("GATEWAY_NODE_NAME", "localhost"))
    bind_host: str = Field(default_factory=lambda: os.getenv("GATEWAY_BIND_HOST", "127.0.0.1"))
    bind_port: int = Field(default_factory=lambda: int(os.getenv("GATEWAY_BIND_PORT", "8787")))
    exec_timeout_seconds: float = Field(default_factory=lambda: float(os.getenv("GATEWAY_EXEC_TIMEOUT_SECONDS", "30")))
    trace_commands: bool = Field(
        default_factory=lambda: os.getenv("GATEWAY_TRACE_COMMANDS", "").strip().lower() in {"1", "true", "yes", "on"}
    )
    workdir: Path | None = Field(
        default_factory=lambda: Path(os.getenv("GATEWAY_WORKDIR")).expanduser() if os.getenv("GATEWAY_WORKDIR") else None
    )

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        normalized_node = str(self.node_name or "").strip()
        self.node_name = normalized_node or "localhost"
        if self.bind_port <= 0 or self.bind_port > 65535:
            raise ValueError("GATEWAY_BIND_PORT must be between 1 and 65535.")
        if self.exec_timeout_seconds <= 0:
            raise ValueError("GATEWAY_EXEC_TIMEOUT_SECONDS must be greater than zero.")
        if self.workdir is not None:
            self.workdir = self.workdir.resolve()
            if not self.workdir.exists():
                raise ValueError(f"GATEWAY_WORKDIR does not exist: {self.workdir}")
            if not self.workdir.is_dir():
                raise ValueError(f"GATEWAY_WORKDIR is not a directory: {self.workdir}")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
