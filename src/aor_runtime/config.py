from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class Settings(BaseModel):
    workspace_root: Path = Field(default_factory=lambda: Path.cwd())
    prompts_root: Path = Field(default_factory=lambda: Path.cwd() / "prompts")
    run_store_path: Path = Field(default_factory=lambda: Path.cwd() / "artifacts" / "runtime.db")
    available_nodes_raw: str | None = Field(default_factory=lambda: os.getenv("AOR_AVAILABLE_NODES") or None)
    default_node: str | None = Field(default_factory=lambda: os.getenv("AOR_DEFAULT_NODE") or None)
    gateway_url: str | None = Field(default_factory=lambda: os.getenv("AOR_GATEWAY_URL") or None)
    gateway_endpoints: dict[str, str] = Field(default_factory=dict)
    gateway_timeout_seconds: float = Field(default_factory=lambda: float(os.getenv("AOR_GATEWAY_TIMEOUT_SECONDS", "30")))
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

    @property
    def available_nodes(self) -> list[str]:
        raw_value = str(self.available_nodes_raw or "")
        nodes: list[str] = []
        seen: set[str] = set()
        for chunk in raw_value.split(","):
            node = chunk.strip()
            if not node or node in seen:
                continue
            nodes.append(node)
            seen.add(node)
        for node in self.gateway_endpoints:
            if node in seen:
                continue
            nodes.append(node)
            seen.add(node)
        default_node = str(self.default_node or "").strip()
        if default_node and default_node not in seen:
            nodes.append(default_node)
        if not nodes:
            nodes.append("localhost")
        return nodes

    def resolved_default_node(self) -> str | None:
        normalized_default = str(self.default_node or "").strip()
        if normalized_default:
            return normalized_default
        available = self.available_nodes
        if len(available) == 1:
            return available[0]
        return None

    def resolve_node(self, node: str = "") -> str:
        requested = str(node or "").strip()
        if not requested:
            requested = str(self.resolved_default_node() or "").strip()
        if not requested:
            raise ValueError("No node specified and no default node is configured.")
        if requested not in self.available_nodes:
            allowed = ", ".join(self.available_nodes) or "<none configured>"
            raise ValueError(f"Node is not available: {requested}. Available nodes: {allowed}.")
        return requested

    def resolve_gateway_url(self, node: str = "") -> str:
        resolved_node = self.resolve_node(node)
        gateway_url = str(self.gateway_endpoints.get(resolved_node, "") or self.gateway_url or "").strip()
        if not gateway_url:
            raise ValueError(f"Gateway URL is not configured for node: {resolved_node}.")
        return gateway_url

    @model_validator(mode="after")
    def validate_default_node(self) -> "Settings":
        normalized_default = str(self.default_node or "").strip()
        self.default_node = normalized_default or None
        normalized_endpoints: dict[str, str] = {}
        for raw_node, raw_url in dict(self.gateway_endpoints or {}).items():
            node = str(raw_node or "").strip()
            url = str(raw_url or "").strip()
            if not node:
                raise ValueError("Gateway endpoint names must be non-empty.")
            if not url:
                raise ValueError(f"Gateway URL must be non-empty for node {node!r}.")
            normalized_endpoints[node] = url
        self.gateway_endpoints = normalized_endpoints
        if self.default_node and self.default_node not in self.available_nodes:
            allowed = ", ".join(self.available_nodes) or "<none configured>"
            raise ValueError(f"Default node must be one of the available nodes. Available nodes: {allowed}.")
        return self


def get_settings() -> Settings:
    settings = Settings()
    settings.run_store_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
