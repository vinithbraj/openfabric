from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


APP_CONFIG_FILENAME = "config.yaml"
APP_CONFIG_PATH_ENV = "AOR_APP_CONFIG_PATH"


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8011

    @model_validator(mode="after")
    def validate_server(self) -> "ServerConfig":
        self.host = str(self.host or "").strip() or "127.0.0.1"
        if self.port <= 0 or self.port > 65535:
            raise ValueError("server.port must be between 1 and 65535.")
        return self


class LLMConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "local"
    default_model: str = "stelterlab/Qwen3-Coder-30B-A3B-Instruct-AWQ"
    default_temperature: float = 0.1
    timeout_seconds: float = 120.0

    @model_validator(mode="after")
    def validate_llm(self) -> "LLMConfig":
        self.base_url = str(self.base_url or "").strip()
        self.api_key = str(self.api_key or "").strip()
        self.default_model = str(self.default_model or "").strip()
        if not self.base_url:
            raise ValueError("llm.base_url is required.")
        if not self.api_key:
            raise ValueError("llm.api_key is required.")
        if not self.default_model:
            raise ValueError("llm.default_model is required.")
        if self.timeout_seconds <= 0:
            raise ValueError("llm.timeout_seconds must be greater than zero.")
        return self


class RuntimeAppConfig(BaseModel):
    allow_destructive_shell: bool = False
    shell_mode: str = "read_only"
    shell_allow_mutation_with_approval: bool = False
    shell_allowed_roots: str | None = None
    shell_default_cwd: str | None = None
    shell_max_output_chars: int = 20000
    shell_command_timeout_seconds: int = 30
    max_plan_retries: int = 2
    run_store_path: str | None = None
    enable_llm_intent_extraction: bool = False
    enable_sql_llm_generation: bool = False
    presentation_mode: str = "user"
    enable_llm_summary: bool = False
    llm_summary_max_facts: int = 50
    include_internal_telemetry: bool = False
    response_render_mode: str = "user"
    show_executed_commands: bool = True
    show_validation_events: bool = False
    show_planner_events: bool = False
    show_tool_events: bool = False
    show_debug_metadata: bool = False
    enable_presentation_llm_summary: bool = False
    presentation_llm_max_facts: int = 50
    presentation_llm_max_input_chars: int = 4000
    presentation_llm_max_output_chars: int = 1500
    presentation_llm_include_row_samples: bool = False
    presentation_llm_include_paths: bool = False
    enable_insight_layer: bool = True
    enable_llm_insights: bool = False
    insight_max_facts: int = 50
    insight_max_input_chars: int = 4000
    insight_max_output_chars: int = 1500
    openai_compat_enabled: bool = True
    openai_compat_model_name: str = "general-purpose-assistant"
    openai_compat_spec_path: str = "examples/general_purpose_assistant.yaml"

    @model_validator(mode="after")
    def validate_runtime(self) -> "RuntimeAppConfig":
        if self.max_plan_retries < 0:
            raise ValueError("runtime.max_plan_retries must be zero or greater.")
        self.shell_mode = str(self.shell_mode or "read_only").strip().lower() or "read_only"
        if self.shell_mode not in {"disabled", "read_only", "approval_required", "permissive"}:
            raise ValueError("runtime.shell_mode must be one of: disabled, read_only, approval_required, permissive.")
        if self.shell_max_output_chars <= 0:
            raise ValueError("runtime.shell_max_output_chars must be greater than zero.")
        if self.shell_command_timeout_seconds <= 0:
            raise ValueError("runtime.shell_command_timeout_seconds must be greater than zero.")
        if self.llm_summary_max_facts <= 0:
            raise ValueError("runtime.llm_summary_max_facts must be greater than zero.")
        if self.presentation_llm_max_facts <= 0:
            raise ValueError("runtime.presentation_llm_max_facts must be greater than zero.")
        if self.presentation_llm_max_input_chars <= 0:
            raise ValueError("runtime.presentation_llm_max_input_chars must be greater than zero.")
        if self.presentation_llm_max_output_chars <= 0:
            raise ValueError("runtime.presentation_llm_max_output_chars must be greater than zero.")
        if self.insight_max_facts <= 0:
            raise ValueError("runtime.insight_max_facts must be greater than zero.")
        if self.insight_max_input_chars <= 0:
            raise ValueError("runtime.insight_max_input_chars must be greater than zero.")
        if self.insight_max_output_chars <= 0:
            raise ValueError("runtime.insight_max_output_chars must be greater than zero.")
        normalized_run_store_path = str(self.run_store_path or "").strip()
        self.run_store_path = normalized_run_store_path or None
        self.presentation_mode = str(self.presentation_mode or "user").strip().lower() or "user"
        if self.presentation_mode not in {"user", "debug", "raw"}:
            raise ValueError("runtime.presentation_mode must be one of: user, debug, raw.")
        self.response_render_mode = str(self.response_render_mode or self.presentation_mode or "user").strip().lower() or "user"
        if self.response_render_mode not in {"user", "debug", "raw"}:
            raise ValueError("runtime.response_render_mode must be one of: user, debug, raw.")
        self.openai_compat_model_name = str(self.openai_compat_model_name or "").strip() or "general-purpose-assistant"
        self.openai_compat_spec_path = str(self.openai_compat_spec_path or "").strip() or "examples/general_purpose_assistant.yaml"
        return self


class SQLConfig(BaseModel):
    database_url: str | None = None
    databases: dict[str, str] = Field(default_factory=dict)
    default_database: str | None = None
    row_limit: int = 500
    timeout_seconds: int = 10

    @model_validator(mode="after")
    def validate_sql(self) -> "SQLConfig":
        normalized_database_url = str(self.database_url or "").strip()
        self.database_url = normalized_database_url or None

        normalized_databases: dict[str, str] = {}
        for raw_name, raw_url in dict(self.databases or {}).items():
            name = str(raw_name or "").strip()
            url = str(raw_url or "").strip()
            if not name:
                raise ValueError("sql.databases keys must be non-empty.")
            if not url:
                raise ValueError(f"sql.databases[{name!r}] must be non-empty.")
            normalized_databases[name] = url
        self.databases = normalized_databases

        normalized_default_database = str(self.default_database or "").strip()
        self.default_database = normalized_default_database or None
        if self.default_database and self.databases and self.default_database not in self.databases:
            available = ", ".join(sorted(self.databases))
            raise ValueError(f"sql.default_database must be one of: {available}.")
        if self.row_limit <= 0:
            raise ValueError("sql.row_limit must be greater than zero.")
        if self.timeout_seconds <= 0:
            raise ValueError("sql.timeout_seconds must be greater than zero.")
        return self


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    runtime: RuntimeAppConfig = Field(default_factory=RuntimeAppConfig)
    sql: SQLConfig = Field(default_factory=SQLConfig)


def _config_error_message(path: Path | None = None) -> str:
    if path is not None:
        return f"App config not found at {path}. Create {APP_CONFIG_FILENAME} or pass --config."
    return f"App config not found. Create {APP_CONFIG_FILENAME} or pass --config."


def resolve_app_config_path(config_path: str | Path | None = None, cwd: str | Path | None = None) -> Path:
    base_dir = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
    requested = config_path if config_path is not None else os.getenv(APP_CONFIG_PATH_ENV)
    if requested is not None and str(requested).strip():
        resolved = Path(requested).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(_config_error_message(resolved))
        return resolved

    resolved = (base_dir / APP_CONFIG_FILENAME).resolve()
    if not resolved.exists():
        raise FileNotFoundError(_config_error_message())
    return resolved


def load_app_config(config_path: str | Path | None = None, cwd: str | Path | None = None) -> tuple[AppConfig, Path]:
    resolved = resolve_app_config_path(config_path=config_path, cwd=cwd)
    try:
        payload = yaml.safe_load(resolved.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid app config YAML at {resolved}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid app config YAML at {resolved}: expected a top-level mapping.")

    return AppConfig.model_validate(payload), resolved
