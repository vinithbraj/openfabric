"""OpenFABRIC Runtime Module: aor_runtime.config

Purpose:
    Define runtime Settings and environment-driven configuration defaults.

Responsibilities:
    Centralize workspace roots, model settings, SQL connections, artifact policy, lifecycle timeouts, and render flags.

Data flow / Interfaces:
    Consumes app config/environment variables and provides typed settings to engine, tools, and API layers.

Boundaries:
    Configuration values must be validated before they affect filesystem, worker lifecycle, or tool execution behavior.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from aor_runtime.app_config import load_app_config
from aor_runtime.model_identity import DEFAULT_OPENAI_COMPAT_MODEL_NAME, normalize_openai_compat_model_name


def _env_bool(name: str, default: bool = False) -> bool:
    """Handle the internal env bool helper path for this module.

    Inputs:
        Receives name, default for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by OpenFABRIC runtime support code paths that import or call aor_runtime.config._env_bool.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    """Represent settings within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by Settings.

    Data flow / Interfaces:
        Instances are created and consumed by OpenFABRIC runtime support code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.config.Settings and related tests.
    """
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
    sql_row_limit: int = 0
    sql_timeout_seconds: int = 10
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_api_key: str = "local"
    default_model: str = "stelterlab/Qwen3-Coder-30B-A3B-Instruct-AWQ"
    default_temperature: float = 0.1
    llm_timeout_seconds: float = 120.0
    allow_destructive_shell: bool = False
    shell_mode: str = Field(default_factory=lambda: os.getenv("AOR_SHELL_MODE", "read_only"))
    shell_allow_mutation_with_approval: bool = Field(default_factory=lambda: _env_bool("AOR_SHELL_ALLOW_MUTATION_WITH_APPROVAL"))
    shell_allowed_roots_raw: str | None = Field(default_factory=lambda: os.getenv("AOR_SHELL_ALLOWED_ROOTS") or None)
    shell_default_cwd: str | None = Field(default_factory=lambda: os.getenv("AOR_SHELL_DEFAULT_CWD") or None)
    shell_max_output_chars: int = Field(default_factory=lambda: int(os.getenv("AOR_SHELL_MAX_OUTPUT_CHARS", "20000")))
    shell_command_timeout_seconds: int = Field(default_factory=lambda: int(os.getenv("AOR_SHELL_COMMAND_TIMEOUT_SECONDS", "30")))
    shutdown_grace_seconds: float = Field(default_factory=lambda: float(os.getenv("AOR_SHUTDOWN_GRACE_SECONDS", "5")))
    worker_join_timeout_seconds: float = Field(default_factory=lambda: float(os.getenv("AOR_WORKER_JOIN_TIMEOUT_SECONDS", "2")))
    tool_process_kill_grace_seconds: float = Field(default_factory=lambda: float(os.getenv("AOR_TOOL_PROCESS_KILL_GRACE_SECONDS", "1")))
    runtime_timezone: str = Field(default_factory=lambda: os.getenv("AOR_RUNTIME_TIMEZONE", "").strip())
    enable_llm_intent_extraction: bool = Field(
        default_factory=lambda: os.getenv("AOR_ENABLE_LLM_INTENT_EXTRACTION", "").strip().lower() in {"1", "true", "yes", "on"}
    )
    enable_sql_llm_generation: bool = Field(
        default_factory=lambda: os.getenv("AOR_ENABLE_SQL_LLM_GENERATION", "").strip().lower() in {"1", "true", "yes", "on"}
    )
    presentation_mode: str = Field(default_factory=lambda: os.getenv("AOR_PRESENTATION_MODE", "user"))
    enable_llm_summary: bool = Field(
        default_factory=lambda: os.getenv("AOR_ENABLE_LLM_SUMMARY", "").strip().lower() in {"1", "true", "yes", "on"}
    )
    llm_summary_max_facts: int = Field(default_factory=lambda: int(os.getenv("AOR_LLM_SUMMARY_MAX_FACTS", "50")))
    include_internal_telemetry: bool = Field(
        default_factory=lambda: os.getenv("AOR_INCLUDE_INTERNAL_TELEMETRY", "").strip().lower() in {"1", "true", "yes", "on"}
    )
    response_render_mode: str = Field(
        default_factory=lambda: os.getenv("AOR_RESPONSE_RENDER_MODE") or os.getenv("AOR_PRESENTATION_MODE", "user")
    )
    show_executed_commands: bool = Field(default_factory=lambda: _env_bool("AOR_SHOW_EXECUTED_COMMANDS", True))
    show_validation_events: bool = Field(default_factory=lambda: _env_bool("AOR_SHOW_VALIDATION_EVENTS"))
    show_planner_events: bool = Field(default_factory=lambda: _env_bool("AOR_SHOW_PLANNER_EVENTS"))
    show_tool_events: bool = Field(default_factory=lambda: _env_bool("AOR_SHOW_TOOL_EVENTS"))
    openwebui_trace_mode: str = Field(default_factory=lambda: os.getenv("AOR_OPENWEBUI_TRACE_MODE", "").strip().lower())
    show_response_stats: bool = Field(default_factory=lambda: _env_bool("AOR_SHOW_RESPONSE_STATS", True))
    show_prompt_suggestions: bool = Field(default_factory=lambda: _env_bool("AOR_SHOW_PROMPT_SUGGESTIONS"))
    show_debug_metadata: bool = Field(default_factory=lambda: _env_bool("AOR_SHOW_DEBUG_METADATA"))
    enable_presentation_llm_summary: bool = Field(default_factory=lambda: _env_bool("AOR_ENABLE_PRESENTATION_LLM_SUMMARY"))
    presentation_llm_max_facts: int = Field(default_factory=lambda: int(os.getenv("AOR_PRESENTATION_LLM_MAX_FACTS", "50")))
    presentation_llm_max_input_chars: int = Field(default_factory=lambda: int(os.getenv("AOR_PRESENTATION_LLM_MAX_INPUT_CHARS", "4000")))
    presentation_llm_max_output_chars: int = Field(default_factory=lambda: int(os.getenv("AOR_PRESENTATION_LLM_MAX_OUTPUT_CHARS", "1500")))
    presentation_llm_include_row_samples: bool = Field(default_factory=lambda: _env_bool("AOR_PRESENTATION_LLM_INCLUDE_ROW_SAMPLES"))
    presentation_llm_include_paths: bool = Field(default_factory=lambda: _env_bool("AOR_PRESENTATION_LLM_INCLUDE_PATHS"))
    intelligent_output_mode: str = Field(default_factory=lambda: os.getenv("AOR_INTELLIGENT_OUTPUT_MODE", "off").strip().lower())
    intelligent_output_max_fields: int = Field(default_factory=lambda: int(os.getenv("AOR_INTELLIGENT_OUTPUT_MAX_FIELDS", "8")))
    semantic_frame_mode: str = Field(default_factory=lambda: os.getenv("AOR_SEMANTIC_FRAME_MODE", "enforce").strip().lower())
    semantic_frame_max_depth: int = Field(default_factory=lambda: int(os.getenv("AOR_SEMANTIC_FRAME_MAX_DEPTH", "10")))
    semantic_frame_max_children: int = Field(default_factory=lambda: int(os.getenv("AOR_SEMANTIC_FRAME_MAX_CHILDREN", "8")))
    llm_stage_max_depth: int = Field(default_factory=lambda: int(os.getenv("AOR_LLM_STAGE_MAX_DEPTH", "10")))
    presentation_intent_max_depth: int = Field(default_factory=lambda: int(os.getenv("AOR_PRESENTATION_INTENT_MAX_DEPTH", "10")))
    enable_insight_layer: bool = Field(default_factory=lambda: _env_bool("AOR_ENABLE_INSIGHT_LAYER", True))
    enable_llm_insights: bool = Field(default_factory=lambda: _env_bool("AOR_ENABLE_LLM_INSIGHTS"))
    insight_max_facts: int = Field(default_factory=lambda: int(os.getenv("AOR_INSIGHT_MAX_FACTS", "50")))
    insight_max_input_chars: int = Field(default_factory=lambda: int(os.getenv("AOR_INSIGHT_MAX_INPUT_CHARS", "4000")))
    insight_max_output_chars: int = Field(default_factory=lambda: int(os.getenv("AOR_INSIGHT_MAX_OUTPUT_CHARS", "1500")))
    action_planner_enabled: bool = Field(default_factory=lambda: _env_bool("AOR_ACTION_PLANNER_ENABLED", True))
    legacy_execution_planner_enabled: bool = Field(default_factory=lambda: _env_bool("AOR_LEGACY_EXECUTION_PLANNER_ENABLED"))
    auto_artifacts_enabled: bool = Field(default_factory=lambda: _env_bool("AOR_AUTO_ARTIFACTS_ENABLED", True))
    auto_artifact_row_threshold: int = Field(default_factory=lambda: int(os.getenv("AOR_AUTO_ARTIFACT_ROW_THRESHOLD", "50")))
    auto_artifact_dir: str = Field(default_factory=lambda: os.getenv("AOR_AUTO_ARTIFACT_DIR", "outputs"))
    auto_artifact_format: str = Field(default_factory=lambda: os.getenv("AOR_AUTO_ARTIFACT_FORMAT", "csv"))
    max_plan_retries: int = 2
    openai_compat_enabled: bool = True
    openai_compat_model_name: str = DEFAULT_OPENAI_COMPAT_MODEL_NAME
    openai_compat_spec_path: str = "examples/general_purpose_assistant.yaml"

    @property
    def available_nodes(self) -> list[str]:
        """Available nodes for Settings instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed property value for callers that need this runtime fact.

        Used by:
            Used by OpenFABRIC runtime support through Settings.available_nodes calls and related tests.
        """
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
        """Resolved default node for Settings instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by OpenFABRIC runtime support through Settings.resolved_default_node calls and related tests.
        """
        normalized_default = str(self.default_node or "").strip()
        if normalized_default:
            return normalized_default
        available = self.available_nodes
        if len(available) == 1:
            return available[0]
        return None

    def resolve_node(self, node: str = "") -> str:
        """Resolve node for Settings instances.

        Inputs:
            Receives node for this Settings method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by OpenFABRIC runtime support through Settings.resolve_node calls and related tests.
        """
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
        """Resolve gateway url for Settings instances.

        Inputs:
            Receives node for this Settings method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by OpenFABRIC runtime support through Settings.resolve_gateway_url calls and related tests.
        """
        resolved_node = self.resolve_node(node)
        gateway_url = str(self.gateway_endpoints.get(resolved_node, "") or self.gateway_url or "").strip()
        if not gateway_url:
            raise ValueError(f"Gateway URL is not configured for node: {resolved_node}.")
        return gateway_url

    def resolve_openai_compat_spec_path(self) -> Path:
        """Resolve openai compat spec path for Settings instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by OpenFABRIC runtime support through Settings.resolve_openai_compat_spec_path calls and related tests.
        """
        raw_path = str(self.openai_compat_spec_path or "").strip()
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            return candidate
        return (self.workspace_root / candidate).resolve()

    @model_validator(mode="after")
    def validate_default_node(self) -> "Settings":
        """Validate validate default node invariants before runtime data crosses this boundary.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the validated value or model instance after enforcing the declared invariant.

        Used by:
            Used by OpenFABRIC runtime support through Settings.validate_default_node calls and related tests.
        """
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
        if self.sql_row_limit < 0:
            raise ValueError("sql_row_limit must be zero or greater.")
        if self.sql_timeout_seconds <= 0:
            raise ValueError("sql_timeout_seconds must be greater than zero.")
        self.shell_mode = str(self.shell_mode or "read_only").strip().lower() or "read_only"
        if self.shell_mode not in {"disabled", "read_only", "approval_required", "permissive"}:
            raise ValueError("shell_mode must be one of: disabled, read_only, approval_required, permissive.")
        if self.shell_max_output_chars <= 0:
            raise ValueError("shell_max_output_chars must be greater than zero.")
        if self.shell_command_timeout_seconds <= 0:
            raise ValueError("shell_command_timeout_seconds must be greater than zero.")
        if self.shutdown_grace_seconds <= 0:
            raise ValueError("shutdown_grace_seconds must be greater than zero.")
        if self.worker_join_timeout_seconds <= 0:
            raise ValueError("worker_join_timeout_seconds must be greater than zero.")
        if self.tool_process_kill_grace_seconds <= 0:
            raise ValueError("tool_process_kill_grace_seconds must be greater than zero.")
        self.runtime_timezone = str(self.runtime_timezone or "").strip()
        if self.max_plan_retries < 0:
            raise ValueError("max_plan_retries must be zero or greater.")
        self.presentation_mode = str(self.presentation_mode or "user").strip().lower() or "user"
        if self.presentation_mode not in {"user", "debug", "raw"}:
            raise ValueError("presentation_mode must be one of: user, debug, raw.")
        self.response_render_mode = str(self.response_render_mode or self.presentation_mode or "user").strip().lower() or "user"
        if self.response_render_mode not in {"user", "debug", "raw"}:
            raise ValueError("response_render_mode must be one of: user, debug, raw.")
        if self.llm_summary_max_facts <= 0:
            raise ValueError("llm_summary_max_facts must be greater than zero.")
        if self.presentation_llm_max_facts <= 0:
            raise ValueError("presentation_llm_max_facts must be greater than zero.")
        if self.presentation_llm_max_input_chars <= 0:
            raise ValueError("presentation_llm_max_input_chars must be greater than zero.")
        if self.presentation_llm_max_output_chars <= 0:
            raise ValueError("presentation_llm_max_output_chars must be greater than zero.")
        self.intelligent_output_mode = str(self.intelligent_output_mode or "off").strip().lower() or "off"
        if self.intelligent_output_mode not in {"off", "compare", "replace"}:
            raise ValueError("intelligent_output_mode must be one of: off, compare, replace.")
        if self.intelligent_output_max_fields <= 0:
            raise ValueError("intelligent_output_max_fields must be greater than zero.")
        self.semantic_frame_mode = str(self.semantic_frame_mode or "enforce").strip().lower() or "enforce"
        if self.semantic_frame_mode not in {"off", "shadow", "enforce"}:
            raise ValueError("semantic_frame_mode must be one of: off, shadow, enforce.")
        if self.semantic_frame_max_depth <= 0:
            raise ValueError("semantic_frame_max_depth must be greater than zero.")
        if self.semantic_frame_max_children <= 0:
            raise ValueError("semantic_frame_max_children must be greater than zero.")
        if self.llm_stage_max_depth <= 0:
            raise ValueError("llm_stage_max_depth must be greater than zero.")
        if self.presentation_intent_max_depth <= 0:
            raise ValueError("presentation_intent_max_depth must be greater than zero.")
        if self.insight_max_facts <= 0:
            raise ValueError("insight_max_facts must be greater than zero.")
        if self.insight_max_input_chars <= 0:
            raise ValueError("insight_max_input_chars must be greater than zero.")
        if self.insight_max_output_chars <= 0:
            raise ValueError("insight_max_output_chars must be greater than zero.")
        if self.auto_artifact_row_threshold < 0:
            raise ValueError("auto_artifact_row_threshold must be zero or greater.")
        self.auto_artifact_dir = str(self.auto_artifact_dir or "outputs").strip() or "outputs"
        self.auto_artifact_format = str(self.auto_artifact_format or "csv").strip().lower() or "csv"
        if self.auto_artifact_format not in {"csv"}:
            raise ValueError("auto_artifact_format must be csv.")
        self.openai_compat_model_name = normalize_openai_compat_model_name(self.openai_compat_model_name)
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
    """Handle the internal cached settings helper path for this module.

    Inputs:
        Receives config_path, cwd for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by OpenFABRIC runtime support code paths that import or call aor_runtime.config._cached_settings.
    """
    app_config, resolved_config_path = load_app_config(config_path=config_path or None, cwd=cwd or None)
    workspace_root = Path(cwd).resolve()
    run_store_path = workspace_root / "artifacts" / "runtime.db"
    if app_config.runtime.run_store_path:
        configured_run_store_path = Path(app_config.runtime.run_store_path).expanduser()
        run_store_path = (
            configured_run_store_path
            if configured_run_store_path.is_absolute()
            else workspace_root / configured_run_store_path
        )
    settings = Settings(
        workspace_root=workspace_root,
        prompts_root=workspace_root / "prompts",
        run_store_path=run_store_path,
        app_config_path=resolved_config_path,
        server_host=app_config.server.host,
        server_port=app_config.server.port,
        llm_base_url=app_config.llm.base_url,
        llm_api_key=app_config.llm.api_key,
        default_model=app_config.llm.default_model,
        default_temperature=app_config.llm.default_temperature,
        llm_timeout_seconds=app_config.llm.timeout_seconds,
        allow_destructive_shell=app_config.runtime.allow_destructive_shell,
        shell_mode=os.getenv("AOR_SHELL_MODE", "").strip().lower() or app_config.runtime.shell_mode,
        shell_allow_mutation_with_approval=_env_bool(
            "AOR_SHELL_ALLOW_MUTATION_WITH_APPROVAL", app_config.runtime.shell_allow_mutation_with_approval
        ),
        shell_allowed_roots_raw=os.getenv("AOR_SHELL_ALLOWED_ROOTS") or app_config.runtime.shell_allowed_roots,
        shell_default_cwd=os.getenv("AOR_SHELL_DEFAULT_CWD") or app_config.runtime.shell_default_cwd,
        shell_max_output_chars=int(os.getenv("AOR_SHELL_MAX_OUTPUT_CHARS", str(app_config.runtime.shell_max_output_chars))),
        shell_command_timeout_seconds=int(
            os.getenv("AOR_SHELL_COMMAND_TIMEOUT_SECONDS", str(app_config.runtime.shell_command_timeout_seconds))
        ),
        shutdown_grace_seconds=float(os.getenv("AOR_SHUTDOWN_GRACE_SECONDS", str(app_config.runtime.shutdown_grace_seconds))),
        worker_join_timeout_seconds=float(
            os.getenv("AOR_WORKER_JOIN_TIMEOUT_SECONDS", str(app_config.runtime.worker_join_timeout_seconds))
        ),
        tool_process_kill_grace_seconds=float(
            os.getenv("AOR_TOOL_PROCESS_KILL_GRACE_SECONDS", str(app_config.runtime.tool_process_kill_grace_seconds))
        ),
        runtime_timezone=os.getenv("AOR_RUNTIME_TIMEZONE", "").strip() or app_config.runtime.runtime_timezone,
        enable_llm_intent_extraction=app_config.runtime.enable_llm_intent_extraction,
        enable_sql_llm_generation=(
            os.getenv("AOR_ENABLE_SQL_LLM_GENERATION", "").strip().lower() in {"1", "true", "yes", "on"}
            or app_config.runtime.enable_sql_llm_generation
        ),
        presentation_mode=os.getenv("AOR_PRESENTATION_MODE", "").strip().lower() or app_config.runtime.presentation_mode,
        enable_llm_summary=(
            os.getenv("AOR_ENABLE_LLM_SUMMARY", "").strip().lower() in {"1", "true", "yes", "on"}
            or app_config.runtime.enable_llm_summary
        ),
        llm_summary_max_facts=int(os.getenv("AOR_LLM_SUMMARY_MAX_FACTS", str(app_config.runtime.llm_summary_max_facts))),
        include_internal_telemetry=(
            os.getenv("AOR_INCLUDE_INTERNAL_TELEMETRY", "").strip().lower() in {"1", "true", "yes", "on"}
            or app_config.runtime.include_internal_telemetry
        ),
        response_render_mode=(
            os.getenv("AOR_RESPONSE_RENDER_MODE", "").strip().lower()
            or os.getenv("AOR_PRESENTATION_MODE", "").strip().lower()
            or app_config.runtime.response_render_mode
            or app_config.runtime.presentation_mode
        ),
        show_executed_commands=_env_bool("AOR_SHOW_EXECUTED_COMMANDS", app_config.runtime.show_executed_commands),
        show_validation_events=_env_bool("AOR_SHOW_VALIDATION_EVENTS", app_config.runtime.show_validation_events),
        show_planner_events=_env_bool("AOR_SHOW_PLANNER_EVENTS", app_config.runtime.show_planner_events),
        show_tool_events=_env_bool("AOR_SHOW_TOOL_EVENTS", app_config.runtime.show_tool_events),
        openwebui_trace_mode=(
            os.getenv("AOR_OPENWEBUI_TRACE_MODE", "").strip().lower()
            or app_config.runtime.openwebui_trace_mode
        ),
        show_response_stats=_env_bool("AOR_SHOW_RESPONSE_STATS", app_config.runtime.show_response_stats),
        show_prompt_suggestions=_env_bool("AOR_SHOW_PROMPT_SUGGESTIONS", app_config.runtime.show_prompt_suggestions),
        show_debug_metadata=_env_bool("AOR_SHOW_DEBUG_METADATA", app_config.runtime.show_debug_metadata),
        enable_presentation_llm_summary=_env_bool(
            "AOR_ENABLE_PRESENTATION_LLM_SUMMARY",
            app_config.runtime.enable_presentation_llm_summary or app_config.runtime.enable_llm_summary,
        ),
        presentation_llm_max_facts=int(os.getenv("AOR_PRESENTATION_LLM_MAX_FACTS", str(app_config.runtime.presentation_llm_max_facts))),
        presentation_llm_max_input_chars=int(
            os.getenv("AOR_PRESENTATION_LLM_MAX_INPUT_CHARS", str(app_config.runtime.presentation_llm_max_input_chars))
        ),
        presentation_llm_max_output_chars=int(
            os.getenv("AOR_PRESENTATION_LLM_MAX_OUTPUT_CHARS", str(app_config.runtime.presentation_llm_max_output_chars))
        ),
        presentation_llm_include_row_samples=_env_bool(
            "AOR_PRESENTATION_LLM_INCLUDE_ROW_SAMPLES", app_config.runtime.presentation_llm_include_row_samples
        ),
        presentation_llm_include_paths=_env_bool("AOR_PRESENTATION_LLM_INCLUDE_PATHS", app_config.runtime.presentation_llm_include_paths),
        intelligent_output_mode=(
            os.getenv("AOR_INTELLIGENT_OUTPUT_MODE", "").strip().lower()
            or app_config.runtime.intelligent_output_mode
        ),
        intelligent_output_max_fields=int(
            os.getenv("AOR_INTELLIGENT_OUTPUT_MAX_FIELDS", str(app_config.runtime.intelligent_output_max_fields))
        ),
        semantic_frame_mode=(
            os.getenv("AOR_SEMANTIC_FRAME_MODE", "").strip().lower()
            or app_config.runtime.semantic_frame_mode
        ),
        semantic_frame_max_depth=int(
            os.getenv("AOR_SEMANTIC_FRAME_MAX_DEPTH", str(app_config.runtime.semantic_frame_max_depth))
        ),
        semantic_frame_max_children=int(
            os.getenv("AOR_SEMANTIC_FRAME_MAX_CHILDREN", str(app_config.runtime.semantic_frame_max_children))
        ),
        llm_stage_max_depth=int(os.getenv("AOR_LLM_STAGE_MAX_DEPTH", str(app_config.runtime.llm_stage_max_depth))),
        presentation_intent_max_depth=int(
            os.getenv("AOR_PRESENTATION_INTENT_MAX_DEPTH", str(app_config.runtime.presentation_intent_max_depth))
        ),
        enable_insight_layer=_env_bool("AOR_ENABLE_INSIGHT_LAYER", app_config.runtime.enable_insight_layer),
        enable_llm_insights=_env_bool("AOR_ENABLE_LLM_INSIGHTS", app_config.runtime.enable_llm_insights),
        insight_max_facts=int(os.getenv("AOR_INSIGHT_MAX_FACTS", str(app_config.runtime.insight_max_facts))),
        insight_max_input_chars=int(os.getenv("AOR_INSIGHT_MAX_INPUT_CHARS", str(app_config.runtime.insight_max_input_chars))),
        insight_max_output_chars=int(os.getenv("AOR_INSIGHT_MAX_OUTPUT_CHARS", str(app_config.runtime.insight_max_output_chars))),
        action_planner_enabled=_env_bool("AOR_ACTION_PLANNER_ENABLED", app_config.runtime.action_planner_enabled),
        legacy_execution_planner_enabled=_env_bool(
            "AOR_LEGACY_EXECUTION_PLANNER_ENABLED", app_config.runtime.legacy_execution_planner_enabled
        ),
        auto_artifacts_enabled=_env_bool("AOR_AUTO_ARTIFACTS_ENABLED", app_config.runtime.auto_artifacts_enabled),
        auto_artifact_row_threshold=int(
            os.getenv("AOR_AUTO_ARTIFACT_ROW_THRESHOLD", str(app_config.runtime.auto_artifact_row_threshold))
        ),
        auto_artifact_dir=os.getenv("AOR_AUTO_ARTIFACT_DIR", app_config.runtime.auto_artifact_dir),
        auto_artifact_format=os.getenv("AOR_AUTO_ARTIFACT_FORMAT", app_config.runtime.auto_artifact_format),
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
    """Get settings for the surrounding runtime workflow.

    Inputs:
        Receives config_path, cwd for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by OpenFABRIC runtime support code paths that import or call aor_runtime.config.get_settings.
    """
    resolved_cwd = str(Path(cwd).resolve()) if cwd is not None else str(Path.cwd().resolve())
    resolved_config = str(Path(config_path).expanduser().resolve()) if config_path is not None else ""
    return _cached_settings(resolved_config, resolved_cwd).model_copy(deep=True)
