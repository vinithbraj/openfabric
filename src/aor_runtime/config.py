from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from aor_runtime.app_config import load_app_config


class Settings(BaseModel):
    workspace_root: Path = Field(default_factory=lambda: Path.cwd())
    prompts_root: Path = Field(default_factory=lambda: Path.cwd() / "prompts")
    run_store_path: Path = Field(default_factory=lambda: Path.cwd() / "artifacts" / "runtime.db")
    app_config_path: Path | None = None
    server_host: str = "127.0.0.1"
    server_port: int = 8011
    available_nodes_raw: str | None = Field(default_factory=lambda: os.getenv("AOR_AVAILABLE_NODES") or None)
    default_node: str | None = Field(default_factory=lambda: os.getenv("AOR_DEFAULT_NODE") or None)
    gateway_url: str | None = Field(default_factory=lambda: os.getenv("AOR_GATEWAY_URL") or None)
    gateway_endpoints: dict[str, str] = Field(default_factory=dict)
    gateway_timeout_seconds: float = Field(default_factory=lambda: float(os.getenv("AOR_GATEWAY_TIMEOUT_SECONDS", "30")))
    sql_database_url: str | None = None
    sql_databases: dict[str, str] = Field(default_factory=dict)
    sql_default_database: str | None = None
    sql_row_limit: int = 500
    sql_timeout_seconds: int = 10
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_api_key: str = "local"
    default_model: str = "stelterlab/Qwen3-Coder-30B-A3B-Instruct-AWQ"
    default_temperature: float = 0.1
    llm_timeout_seconds: float = 120.0
    allow_destructive_shell: bool = False
    max_plan_retries: int = 2
    openai_compat_enabled: bool = True
    openai_compat_model_name: str = "general-purpose-assistant"
    openai_compat_spec_path: str = "examples/general_purpose_assistant.yaml"

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

    def resolve_openai_compat_spec_path(self) -> Path:
        raw_path = str(self.openai_compat_spec_path or "").strip()
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            return candidate
        return (self.workspace_root / candidate).resolve()

    @model_validator(mode="after")
    def validate_default_node(self) -> "Settings":
        if self.server_port <= 0 or self.server_port > 65535:
            raise ValueError("server_port must be between 1 and 65535.")
        normalized_default = str(self.default_node or "").strip()
        self.default_node = normalized_default or None
        normalized_database_url = str(self.sql_database_url or "").strip()
        self.sql_database_url = normalized_database_url or None
        normalized_sql_databases: dict[str, str] = {}
        for raw_name, raw_url in dict(self.sql_databases or {}).items():
            name = str(raw_name or "").strip()
            url = str(raw_url or "").strip()
            if not name:
                raise ValueError("SQL database names must be non-empty.")
            if not url:
                raise ValueError(f"SQL database URL must be non-empty for database {name!r}.")
            normalized_sql_databases[name] = url
        self.sql_databases = normalized_sql_databases
        normalized_sql_default = str(self.sql_default_database or "").strip()
        self.sql_default_database = normalized_sql_default or None
        self.llm_base_url = str(self.llm_base_url or "").strip()
        self.llm_api_key = str(self.llm_api_key or "").strip()
        self.default_model = str(self.default_model or "").strip()
        if not self.llm_base_url:
            raise ValueError("llm_base_url is required.")
        if not self.llm_api_key:
            raise ValueError("llm_api_key is required.")
        if not self.default_model:
            raise ValueError("default_model is required.")
        if self.default_temperature < 0:
            raise ValueError("default_temperature must be zero or greater.")
        if self.llm_timeout_seconds <= 0:
            raise ValueError("llm_timeout_seconds must be greater than zero.")
        if self.sql_row_limit <= 0:
            raise ValueError("sql_row_limit must be greater than zero.")
        if self.sql_timeout_seconds <= 0:
            raise ValueError("sql_timeout_seconds must be greater than zero.")
        if self.max_plan_retries < 0:
            raise ValueError("max_plan_retries must be zero or greater.")
        self.openai_compat_model_name = str(self.openai_compat_model_name or "").strip() or "general-purpose-assistant"
        self.openai_compat_spec_path = str(self.openai_compat_spec_path or "").strip() or "examples/general_purpose_assistant.yaml"
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


@lru_cache(maxsize=8)
def _cached_settings(config_path: str, cwd: str) -> Settings:
    app_config, resolved_config_path = load_app_config(config_path=config_path or None, cwd=cwd or None)
    workspace_root = Path(cwd).resolve()
    settings = Settings(
        workspace_root=workspace_root,
        prompts_root=workspace_root / "prompts",
        run_store_path=workspace_root / "artifacts" / "runtime.db",
        app_config_path=resolved_config_path,
        server_host=app_config.server.host,
        server_port=app_config.server.port,
        llm_base_url=app_config.llm.base_url,
        llm_api_key=app_config.llm.api_key,
        default_model=app_config.llm.default_model,
        default_temperature=app_config.llm.default_temperature,
        llm_timeout_seconds=app_config.llm.timeout_seconds,
        allow_destructive_shell=app_config.runtime.allow_destructive_shell,
        max_plan_retries=app_config.runtime.max_plan_retries,
        sql_database_url=app_config.sql.database_url,
        sql_databases=app_config.sql.databases,
        sql_default_database=app_config.sql.default_database,
        sql_row_limit=app_config.sql.row_limit,
        sql_timeout_seconds=app_config.sql.timeout_seconds,
        openai_compat_enabled=app_config.runtime.openai_compat_enabled,
        openai_compat_model_name=app_config.runtime.openai_compat_model_name,
        openai_compat_spec_path=app_config.runtime.openai_compat_spec_path,
    )
    settings.run_store_path.parent.mkdir(parents=True, exist_ok=True)
    return settings


def get_settings(config_path: str | Path | None = None, cwd: str | Path | None = None) -> Settings:
    resolved_cwd = str(Path(cwd).resolve()) if cwd is not None else str(Path.cwd().resolve())
    resolved_config = str(Path(config_path).expanduser().resolve()) if config_path is not None else ""
    return _cached_settings(resolved_config, resolved_cwd).model_copy(deep=True)
