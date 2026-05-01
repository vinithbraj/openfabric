"""OpenFABRIC Runtime Module: aor_runtime.app_config

Purpose:
    Load application-level configuration from YAML and environment sources.

Responsibilities:
    Preserve existing server, model, gateway, SQL, and runtime keys so current
    config files continue to load during the reset.

Data flow / Interfaces:
    Consumes config files/environment variables and produces typed app
    configuration used by CLI/API startup.

Boundaries:
    Keeps process configuration separate from prompt payloads; the reset runtime
    does not convert these settings into tool calls.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from aor_runtime.model_identity import DEFAULT_OPENAI_COMPAT_MODEL_NAME, normalize_openai_compat_model_name


APP_CONFIG_FILENAME = "config.yaml"
APP_CONFIG_PATH_ENV = "AOR_APP_CONFIG_PATH"


class ServerConfig(BaseModel):
    """Represent server config within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ServerConfig.

    Data flow / Interfaces:
        Instances are created and consumed by OpenFABRIC runtime support code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.app_config.ServerConfig and related tests.
    """
    host: str = "127.0.0.1"
    port: int = 8011

    @model_validator(mode="after")
    def validate_server(self) -> "ServerConfig":
        """Validate validate server invariants before runtime data crosses this boundary.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the validated value or model instance after enforcing the declared invariant.

        Used by:
            Used by OpenFABRIC runtime support through ServerConfig.validate_server calls and related tests.
        """
        self.host = str(self.host or "").strip() or "127.0.0.1"
        if self.port <= 0 or self.port > 65535:
            raise ValueError("server.port must be between 1 and 65535.")
        return self


class LLMConfig(BaseModel):
    """Represent l l m config within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by LLMConfig.

    Data flow / Interfaces:
        Instances are created and consumed by OpenFABRIC runtime support code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.app_config.LLMConfig and related tests.
    """
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "local"
    default_model: str = "stelterlab/Qwen3-Coder-30B-A3B-Instruct-AWQ"
    default_temperature: float = 0.1
    timeout_seconds: float = 120.0

    @model_validator(mode="after")
    def validate_llm(self) -> "LLMConfig":
        """Validate validate llm invariants before runtime data crosses this boundary.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the validated value or model instance after enforcing the declared invariant.

        Used by:
            Used by OpenFABRIC runtime support through LLMConfig.validate_llm calls and related tests.
        """
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
    """Represent runtime app config within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by RuntimeAppConfig.

    Data flow / Interfaces:
        Instances are created and consumed by OpenFABRIC runtime support code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.app_config.RuntimeAppConfig and related tests.
    """
    allow_destructive_shell: bool = False
    shell_mode: str = "read_only"
    shell_allow_mutation_with_approval: bool = False
    shell_allowed_roots: str | None = None
    shell_default_cwd: str | None = None
    shell_max_output_chars: int = 20000
    shell_command_timeout_seconds: int = 30
    shutdown_grace_seconds: float = 5.0
    worker_join_timeout_seconds: float = 2.0
    tool_process_kill_grace_seconds: float = 1.0
    runtime_timezone: str = ""
    max_plan_retries: int = 2
    action_planner_enabled: bool = True
    legacy_execution_planner_enabled: bool = False
    auto_artifacts_enabled: bool = True
    auto_artifact_row_threshold: int = 50
    auto_artifact_dir: str = "outputs"
    auto_artifact_format: str = "csv"
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
    openwebui_trace_mode: str = ""
    show_response_stats: bool = True
    show_prompt_suggestions: bool = False
    show_debug_metadata: bool = False
    enable_presentation_llm_summary: bool = False
    presentation_llm_max_facts: int = 50
    presentation_llm_max_input_chars: int = 4000
    presentation_llm_max_output_chars: int = 1500
    presentation_llm_include_row_samples: bool = False
    presentation_llm_include_paths: bool = False
    intelligent_output_mode: str = "off"
    intelligent_output_max_fields: int = 8
    semantic_frame_mode: str = "enforce"
    semantic_frame_max_depth: int = 10
    semantic_frame_max_children: int = 8
    llm_stage_max_depth: int = 10
    presentation_intent_max_depth: int = 10
    enable_insight_layer: bool = True
    enable_llm_insights: bool = False
    insight_max_facts: int = 50
    insight_max_input_chars: int = 4000
    insight_max_output_chars: int = 1500
    openai_compat_enabled: bool = True
    openai_compat_model_name: str = DEFAULT_OPENAI_COMPAT_MODEL_NAME
    openai_compat_spec_path: str = "examples/general_purpose_assistant.yaml"

    @model_validator(mode="after")
    def validate_runtime(self) -> "RuntimeAppConfig":
        """Validate validate runtime invariants before runtime data crosses this boundary.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the validated value or model instance after enforcing the declared invariant.

        Used by:
            Used by OpenFABRIC runtime support through RuntimeAppConfig.validate_runtime calls and related tests.
        """
        if self.max_plan_retries < 0:
            raise ValueError("runtime.max_plan_retries must be zero or greater.")
        self.shell_mode = str(self.shell_mode or "read_only").strip().lower() or "read_only"
        if self.shell_mode not in {"disabled", "read_only", "approval_required", "permissive"}:
            raise ValueError("runtime.shell_mode must be one of: disabled, read_only, approval_required, permissive.")
        if self.shell_max_output_chars <= 0:
            raise ValueError("runtime.shell_max_output_chars must be greater than zero.")
        if self.shell_command_timeout_seconds <= 0:
            raise ValueError("runtime.shell_command_timeout_seconds must be greater than zero.")
        if self.shutdown_grace_seconds <= 0:
            raise ValueError("runtime.shutdown_grace_seconds must be greater than zero.")
        if self.worker_join_timeout_seconds <= 0:
            raise ValueError("runtime.worker_join_timeout_seconds must be greater than zero.")
        if self.tool_process_kill_grace_seconds <= 0:
            raise ValueError("runtime.tool_process_kill_grace_seconds must be greater than zero.")
        self.runtime_timezone = str(self.runtime_timezone or "").strip()
        if self.llm_summary_max_facts <= 0:
            raise ValueError("runtime.llm_summary_max_facts must be greater than zero.")
        if self.presentation_llm_max_facts <= 0:
            raise ValueError("runtime.presentation_llm_max_facts must be greater than zero.")
        if self.presentation_llm_max_input_chars <= 0:
            raise ValueError("runtime.presentation_llm_max_input_chars must be greater than zero.")
        if self.presentation_llm_max_output_chars <= 0:
            raise ValueError("runtime.presentation_llm_max_output_chars must be greater than zero.")
        self.intelligent_output_mode = str(self.intelligent_output_mode or "off").strip().lower() or "off"
        if self.intelligent_output_mode not in {"off", "compare", "replace"}:
            raise ValueError("runtime.intelligent_output_mode must be one of: off, compare, replace.")
        if self.intelligent_output_max_fields <= 0:
            raise ValueError("runtime.intelligent_output_max_fields must be greater than zero.")
        self.semantic_frame_mode = str(self.semantic_frame_mode or "enforce").strip().lower() or "enforce"
        if self.semantic_frame_mode not in {"off", "shadow", "enforce"}:
            raise ValueError("runtime.semantic_frame_mode must be one of: off, shadow, enforce.")
        if self.semantic_frame_max_depth <= 0:
            raise ValueError("runtime.semantic_frame_max_depth must be greater than zero.")
        if self.semantic_frame_max_children <= 0:
            raise ValueError("runtime.semantic_frame_max_children must be greater than zero.")
        if self.llm_stage_max_depth <= 0:
            raise ValueError("runtime.llm_stage_max_depth must be greater than zero.")
        if self.presentation_intent_max_depth <= 0:
            raise ValueError("runtime.presentation_intent_max_depth must be greater than zero.")
        if self.insight_max_facts <= 0:
            raise ValueError("runtime.insight_max_facts must be greater than zero.")
        if self.insight_max_input_chars <= 0:
            raise ValueError("runtime.insight_max_input_chars must be greater than zero.")
        if self.insight_max_output_chars <= 0:
            raise ValueError("runtime.insight_max_output_chars must be greater than zero.")
        if self.auto_artifact_row_threshold < 0:
            raise ValueError("runtime.auto_artifact_row_threshold must be zero or greater.")
        self.auto_artifact_dir = str(self.auto_artifact_dir or "outputs").strip() or "outputs"
        self.auto_artifact_format = str(self.auto_artifact_format or "csv").strip().lower() or "csv"
        if self.auto_artifact_format not in {"csv"}:
            raise ValueError("runtime.auto_artifact_format must be csv.")
        normalized_run_store_path = str(self.run_store_path or "").strip()
        self.run_store_path = normalized_run_store_path or None
        self.presentation_mode = str(self.presentation_mode or "user").strip().lower() or "user"
        if self.presentation_mode not in {"user", "debug", "raw"}:
            raise ValueError("runtime.presentation_mode must be one of: user, debug, raw.")
        self.response_render_mode = str(self.response_render_mode or self.presentation_mode or "user").strip().lower() or "user"
        if self.response_render_mode not in {"user", "debug", "raw"}:
            raise ValueError("runtime.response_render_mode must be one of: user, debug, raw.")
        self.openwebui_trace_mode = str(self.openwebui_trace_mode or "").strip().lower()
        if self.openwebui_trace_mode and self.openwebui_trace_mode not in {"off", "summary", "diagnostic"}:
            raise ValueError("runtime.openwebui_trace_mode must be one of: off, summary, diagnostic.")
        self.openai_compat_model_name = normalize_openai_compat_model_name(self.openai_compat_model_name)
        self.openai_compat_spec_path = str(self.openai_compat_spec_path or "").strip() or "examples/general_purpose_assistant.yaml"
        return self


class SQLConfig(BaseModel):
    """Represent s q l config within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SQLConfig.

    Data flow / Interfaces:
        Instances are created and consumed by OpenFABRIC runtime support code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.app_config.SQLConfig and related tests.
    """
    database_url: str | None = None
    databases: dict[str, str] = Field(default_factory=dict)
    default_database: str | None = None
    row_limit: int = 0
    timeout_seconds: int = 10

    @model_validator(mode="after")
    def validate_sql(self) -> "SQLConfig":
        """Validate validate sql invariants before runtime data crosses this boundary.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the validated value or model instance after enforcing the declared invariant.

        Used by:
            Used by OpenFABRIC runtime support through SQLConfig.validate_sql calls and related tests.
        """
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
        if self.row_limit < 0:
            raise ValueError("sql.row_limit must be zero or greater.")
        if self.timeout_seconds <= 0:
            raise ValueError("sql.timeout_seconds must be greater than zero.")
        return self


class AppConfig(BaseModel):
    """Represent app config within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by AppConfig.

    Data flow / Interfaces:
        Instances are created and consumed by OpenFABRIC runtime support code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.app_config.AppConfig and related tests.
    """
    server: ServerConfig = Field(default_factory=ServerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    runtime: RuntimeAppConfig = Field(default_factory=RuntimeAppConfig)
    sql: SQLConfig = Field(default_factory=SQLConfig)


def _config_error_message(path: Path | None = None) -> str:
    """Handle the internal config error message helper path for this module.

    Inputs:
        Receives path for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by OpenFABRIC runtime support code paths that import or call aor_runtime.app_config._config_error_message.
    """
    if path is not None:
        return f"App config not found at {path}. Create {APP_CONFIG_FILENAME} or pass --config."
    return f"App config not found. Create {APP_CONFIG_FILENAME} or pass --config."


def resolve_app_config_path(config_path: str | Path | None = None, cwd: str | Path | None = None) -> Path:
    """Resolve app config path for the surrounding runtime workflow.

    Inputs:
        Receives config_path, cwd for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by OpenFABRIC runtime support code paths that import or call aor_runtime.app_config.resolve_app_config_path.
    """
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
    """Load app config for the surrounding runtime workflow.

    Inputs:
        Receives config_path, cwd for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by OpenFABRIC runtime support code paths that import or call aor_runtime.app_config.load_app_config.
    """
    resolved = resolve_app_config_path(config_path=config_path, cwd=cwd)
    try:
        payload = yaml.safe_load(resolved.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid app config YAML at {resolved}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid app config YAML at {resolved}: expected a top-level mapping.")

    return AppConfig.model_validate(payload), resolved
