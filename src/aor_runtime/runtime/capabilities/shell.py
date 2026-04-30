"""OpenFABRIC Runtime Module: aor_runtime.runtime.capabilities.shell

Purpose:
    Provide compatibility capability-pack helpers and fixtures for domain-specific tests and utilities.

Responsibilities:
    Classify or compile typed intents when called directly by tests or compatibility surfaces.

Data flow / Interfaces:
    Consumes compile contexts, allowed tools, and typed intents; returns execution-plan fragments or eval metadata.

Boundaries:
    These modules are not the active top-level natural-language planner; user prompts route through LLMActionPlanner.
"""

from __future__ import annotations

import re
import shlex
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from aor_runtime.core.contracts import ExecutionPlan
from aor_runtime.runtime.capabilities.base import CapabilityPack, ClassificationContext, CompileContext, CompiledIntentPlan
from aor_runtime.runtime.file_query import normalize_file_query
from aor_runtime.runtime.intent_classifier import classify_single_intent
from aor_runtime.runtime.intent_compiler import compile_intent_to_plan
from aor_runtime.runtime.intents import FetchExtractIntent, FileAggregateIntent, IntentResult, ShellCommandIntent
from aor_runtime.runtime.llm_intent_extractor import LLMIntentExtractor
from aor_runtime.runtime.output_contract import build_output_contract
from aor_runtime.runtime.shell_safety import FORBIDDEN_COMMANDS, MUTATING_COMMANDS, SAFE_READ_ONLY_COMMANDS, classify_shell_command


SHELL_FRAMING_RE = re.compile(r"\b(?:using\s+shell|with\s+shell|shell:|bash|terminal|command\s+line)\b", re.IGNORECASE)
SYSTEM_INSPECTION_RE = re.compile(
    r"\b(?:uptime|host\s*name|hostname|whoami|current\s+user|pwd|working\s+directory|disk\s+usage|directory\s+size|folder\s+size|memory\s+usage|memory\s+summary|cpu\s+summary|top\s+processes|process(?:es)?|port\s+\d+|using\s+port|listening\s+ports|open\s+ports|network(?:\s+summary)?|mounts?|service\s+status|recent\s+logs?|system\s+load)\b",
    re.IGNORECASE,
)
SHELL_MUTATION_RE = re.compile(
    r"\b(?:rm\b|delete\b|kill\b|shutdown\b|reboot\b|systemctl\b|service\s+restart\b|chmod\b|chown\b|scancel\b|sbatch\b|drain\b|resume\b)\b",
    re.IGNORECASE,
)
SHELL_FETCH_RE = re.compile(r"\b(?:fetch|curl|title|head)\b", re.IGNORECASE)
SHELL_CONTENT_DIAGNOSTIC_RE = re.compile(
    r"\b(?:matching\s+lines?|lines?\s+containing|files?\s+containing|contents?\s+(?:include|contain)|mention(?:ing)?)\b",
    re.IGNORECASE,
)
SHELL_FILE_AGGREGATE_RE = re.compile(
    r"\b(?:total\s+file\s+size|total\s+size|sum(?:\s+the)?\s+size|how\s+much\s+space|disk\s+space\s+used\s+by|size\s+of\s+all|total\s+bytes|count\s+and\s+total\s+size)\b",
    re.IGNORECASE,
)
SHELL_SQL_RE = re.compile(r"\b(?:select\b|query\b|database\b|table\b|\bfrom\s+[A-Za-z_][A-Za-z0-9_]*\b)\b", re.IGNORECASE)
SHELL_SLURM_RE = re.compile(r"\b(?:slurm|squeue|sacct|sinfo|scontrol|slurmdbd|partition|queue|scheduler)\b", re.IGNORECASE)
PATH_CANDIDATE_RE = re.compile(r"(~?/[^,\s]+|\.\.?/[^,\s]+|[A-Za-z]:[\\/][^,\s]+)")
URL_RE = re.compile(r"\b(?:https?://|file://)\S+|\b[a-z0-9.-]+\.[a-z]{2,}(?:/\S*)?\b", re.IGNORECASE)
NODE_FOR_RE = re.compile(r"\bon\s+node\s+([A-Za-z0-9._-]+)\b", re.IGNORECASE)
TRAILING_NODE_RE = re.compile(r"\bon\s+([A-Za-z0-9._-]+)\s*$", re.IGNORECASE)
SHELL_META_RE = re.compile(r"[;|&$`><\n]")
SAFE_NODE_RE = re.compile(r"^[A-Za-z0-9._-]+$")
EXPLICIT_COMMAND_RE = re.compile(
    r"^\s*(?:run|execute)\s+(?:this\s+command\s*:?\s*)?(?P<command>.+?)\s*$|^\s*(?:shell|bash|terminal)\s*:\s*(?P<prefixed>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
PORT_RE = re.compile(r"\bport\s+(?P<port>\d{1,5})\b|:\s*(?P<colon_port>\d{1,5})\b", re.IGNORECASE)
SERVICE_STATUS_RE = re.compile(r"\b(?:service\s+status|status\s+of\s+service|systemctl\s+status)\s+(?:for\s+)?(?P<service>[A-Za-z0-9_.@-]+)\b", re.IGNORECASE)
PROCESS_SEARCH_RE = re.compile(r"\b(?:process(?:es)?\s+(?:named|called|matching)|show\s+running)\s+(?P<name>[A-Za-z0-9_.@:+-]+)\s+process(?:es)?\b", re.IGNORECASE)
PATH_VALUE_RE = re.compile(r"(~?/[^,\s]+|\.\.?/[^,\s]+|[A-Za-z]:[\\/][^,\s]+|\.)")

ALLOWED_SHELL_KINDS = {
    "cpu_summary",
    "current_user",
    "current_working_directory",
    "uptime",
    "host_identity",
    "disk_usage",
    "directory_size",
    "memory_summary",
    "process_list",
    "process_summary",
    "process_search",
    "port_usage",
    "network_summary",
    "listening_ports",
    "mounts",
    "service_status",
    "recent_logs",
    "large_files",
}


class ShellExplicitCommandIntent(BaseModel):
    """Represent shell explicit command intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ShellExplicitCommandIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.shell.ShellExplicitCommandIntent and related tests.
    """
    model_config = ConfigDict(extra="forbid")

    command: str
    cwd: str | None = None
    node: str | None = None
    output_mode: Literal["text", "json"] = "text"
    requires_approval: bool = False


class ShellInspectionIntent(BaseModel):
    """Represent shell inspection intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ShellInspectionIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.shell.ShellInspectionIntent and related tests.
    """
    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "directory_listing",
        "uptime",
        "host_identity",
        "disk_usage",
        "directory_size",
        "memory_summary",
        "cpu_summary",
        "current_user",
        "current_working_directory",
        "process_list",
        "process_summary",
        "process_search",
        "port_usage",
        "network_summary",
        "listening_ports",
        "mounts",
        "environment_summary",
        "service_status",
        "recent_logs",
        "file_find",
        "large_files",
    ]
    path: str | None = None
    pattern: str | None = None
    process_name: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    service: str | None = None
    node: str | None = None
    limit: int | None = Field(default=20, ge=1, le=200)
    cwd: str | None = None
    output_mode: Literal["text", "csv", "json"] = "text"


class FileContentDiagnosticIntent(BaseModel):
    """Represent file content diagnostic intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by FileContentDiagnosticIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.shell.FileContentDiagnosticIntent and related tests.
    """
    model_config = ConfigDict(extra="forbid")

    path: str
    needle: str
    pattern: str = "*"
    recursive: bool = True
    result_kind: Literal["files", "lines"] = "files"
    output_mode: Literal["text", "csv", "json"] = "text"


class ShellUnsupportedMutationIntent(BaseModel):
    """Represent shell unsupported mutation intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ShellUnsupportedMutationIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.shell.ShellUnsupportedMutationIntent and related tests.
    """
    model_config = ConfigDict(extra="forbid")

    operation: str
    reason: str


class ShellMutationIntent(BaseModel):
    """Represent shell mutation intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ShellMutationIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.shell.ShellMutationIntent and related tests.
    """
    model_config = ConfigDict(extra="forbid")

    operation: Literal[
        "delete",
        "move",
        "copy",
        "chmod",
        "chown",
        "kill_process",
        "restart_service",
        "stop_service",
        "start_service",
        "install_package",
        "modify_system",
        "network_change",
        "unknown_mutation",
    ]
    target: str | None = None
    reason: str
    requires_approval: bool = True


class ShellUnsupportedIntent(BaseModel):
    """Represent shell unsupported intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ShellUnsupportedIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.shell.ShellUnsupportedIntent and related tests.
    """
    model_config = ConfigDict(extra="forbid")

    reason: str
    safer_alternatives: list[str] = Field(default_factory=list)


class ShellCapabilityPack(CapabilityPack):
    """Represent shell capability pack within the OpenFABRIC runtime. It extends CapabilityPack.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ShellCapabilityPack.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.shell.ShellCapabilityPack and related tests.
    """
    name = "shell"
    supports_llm_intent_extraction = True
    intent_types = (
        ShellExplicitCommandIntent,
        ShellCommandIntent,
        ShellInspectionIntent,
        FileContentDiagnosticIntent,
        ShellMutationIntent,
        ShellUnsupportedIntent,
        ShellUnsupportedMutationIntent,
    )

    def classify(self, goal: str, context: ClassificationContext) -> IntentResult:
        """Classify for ShellCapabilityPack instances.

        Inputs:
            Receives goal, context for this ShellCapabilityPack method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ShellCapabilityPack.classify calls and related tests.
        """
        shell_result = _classify_shell_goal(goal, context)
        if shell_result.matched:
            return shell_result
        result = classify_single_intent(goal, schema_payload=context.schema_payload)
        if result.matched and isinstance(result.intent, self.intent_types):
            return result
        return IntentResult(matched=False, reason=f"{self.name}_no_match")

    def is_llm_intent_domain(self, goal: str, context: ClassificationContext) -> bool:
        """Is llm intent domain for ShellCapabilityPack instances.

        Inputs:
            Receives goal, context for this ShellCapabilityPack method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ShellCapabilityPack.is_llm_intent_domain calls and related tests.
        """
        if not context.settings.enable_llm_intent_extraction:
            return False
        prompt = str(goal or "").strip()
        if not prompt:
            return False
        if SHELL_SLURM_RE.search(prompt):
            return False
        if SHELL_MUTATION_RE.search(prompt):
            return True

        has_shell_framing = SHELL_FRAMING_RE.search(prompt) is not None
        is_system_inspection = SYSTEM_INSPECTION_RE.search(prompt) is not None

        if not has_shell_framing and not is_system_inspection:
            return False
        if not has_shell_framing and _looks_like_plain_filesystem_prompt(prompt):
            return False
        if not has_shell_framing and SHELL_FETCH_RE.search(prompt):
            return False
        if SHELL_SQL_RE.search(prompt):
            return False

        if has_shell_framing:
            return any(
                checker(prompt)
                for checker in (
                    lambda text: SYSTEM_INSPECTION_RE.search(text) is not None,
                    _looks_like_file_aggregate_prompt,
                    _looks_like_content_diagnostic_prompt,
                    _looks_like_fetch_prompt,
                )
            )
        return True

    def try_llm_extract(self, goal: str, context: ClassificationContext, extractor: LLMIntentExtractor) -> IntentResult:
        """Try llm extract for ShellCapabilityPack instances.

        Inputs:
            Receives goal, context, extractor for this ShellCapabilityPack method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ShellCapabilityPack.try_llm_extract calls and related tests.
        """
        if not context.settings.enable_llm_intent_extraction:
            return IntentResult(matched=False, reason="shell_llm_intent_disabled")
        prompt = str(goal or "").strip()
        if not prompt:
            return IntentResult(matched=False, reason="shell_llm_empty_goal")

        routed_prompt, node = _extract_prompt_node(prompt, context.settings.available_nodes)
        mutation_match = SHELL_MUTATION_RE.search(routed_prompt)
        if mutation_match is not None:
            return IntentResult(
                matched=True,
                intent=ShellUnsupportedMutationIntent(
                    operation=mutation_match.group(0),
                    reason="This runtime supports read-only shell/system inspection only.",
                ),
                metadata={
                    "planning_mode": "llm_intent_extractor",
                    "capability": self.name,
                    "llm_calls": 0,
                    "llm_intent_calls": 0,
                    "raw_planner_llm_calls": 0,
                    "llm_intent_reason": "Rejected mutating shell/system request before LLM extraction.",
                },
            )

        extracted = extractor.extract_intent(
            routed_prompt,
            self.name,
            [
                ShellInspectionIntent,
                ShellMutationIntent,
                ShellUnsupportedIntent,
                FileContentDiagnosticIntent,
                FileAggregateIntent,
                FetchExtractIntent,
            ],
            context={
                "system_prompt": _shell_llm_intent_system_prompt(),
                "confidence_threshold": 0.70,
                "temperature": 0.0,
            },
        )
        if not extracted.matched or extracted.intent is None:
            return IntentResult(matched=False, reason=extracted.reason or "shell_llm_no_match")

        try:
            safe_intent = _finalize_llm_shell_intent(
                extracted.intent,
                routed_prompt,
                node=node,
                available_nodes=context.settings.available_nodes,
            )
        except Exception as exc:  # noqa: BLE001
            return IntentResult(matched=False, reason=str(exc))

        return IntentResult(
            matched=True,
            intent=safe_intent,
            metadata={
                "planning_mode": "llm_intent_extractor",
                "capability": _llm_capability_owner(safe_intent),
                "llm_calls": 1,
                "llm_intent_calls": 1,
                "raw_planner_llm_calls": 0,
                "llm_intent_type": type(safe_intent).__name__,
                "llm_intent_confidence": extracted.confidence,
                "llm_intent_reason": extracted.reason,
            },
        )

    def compile(self, intent: Any, context: CompileContext) -> CompiledIntentPlan | None:
        """Compile for ShellCapabilityPack instances.

        Inputs:
            Receives intent, context for this ShellCapabilityPack method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ShellCapabilityPack.compile calls and related tests.
        """
        if isinstance(intent, ShellExplicitCommandIntent):
            _require_tools(context.allowed_tools, "shell.exec")
            policy = classify_shell_command(
                intent.command,
                mode=context.settings.shell_mode,
                allow_mutation_with_approval=context.settings.shell_allow_mutation_with_approval
                or context.settings.allow_destructive_shell,
            )
            if not policy.allowed:
                return _compile_shell_refusal(
                    reason=policy.reason,
                    risk=policy.risk,
                    command=policy.redacted_command or intent.command,
                    approval_required=policy.requires_approval,
                )
            shell_args: dict[str, Any] = {"command": intent.command}
            if intent.node:
                shell_args["node"] = intent.node
            return CompiledIntentPlan(
                plan=ExecutionPlan.model_validate(
                    {
                        "steps": [
                            {"id": 1, "action": "shell.exec", "args": shell_args, "output": "shell_output"},
                            {
                                "id": 2,
                                "action": "runtime.return",
                                "input": ["shell_output"],
                                "args": {
                                    "value": {"$ref": "shell_output", "path": "stdout"},
                                    "mode": intent.output_mode,
                                    "output_contract": build_output_contract(mode=intent.output_mode),
                                },
                            },
                        ]
                    }
                ),
                metadata={"capability_pack": self.name, "intent_type": type(intent).__name__},
            )

        if isinstance(intent, ShellCommandIntent):
            return CompiledIntentPlan(
                plan=compile_intent_to_plan(intent, context.allowed_tools, context.settings),
                metadata={"capability_pack": self.name, "intent_type": type(intent).__name__},
            )

        if isinstance(intent, (ShellUnsupportedMutationIntent, ShellMutationIntent, ShellUnsupportedIntent)):
            if isinstance(intent, ShellMutationIntent):
                reason = intent.reason
                operation = intent.operation
            elif isinstance(intent, ShellUnsupportedIntent):
                reason = intent.reason
                operation = "unsupported shell request"
            else:
                reason = intent.reason
                operation = intent.operation
            return CompiledIntentPlan(
                plan=ExecutionPlan.model_validate(
                    {
                        "steps": [
                            {
                                "id": 1,
                                "action": "runtime.return",
                                "args": {
                                    "value": _shell_refusal_markdown(reason=reason, operation=operation),
                                    "mode": "text",
                                    "output_contract": build_output_contract(mode="text"),
                                },
                            }
                        ]
                    }
                ),
                metadata={"capability_pack": self.name, "intent_type": type(intent).__name__},
            )

        if isinstance(intent, ShellInspectionIntent):
            _require_tools(context.allowed_tools, "shell.exec")
            command = _shell_inspection_command(intent)
            shell_args: dict[str, Any] = {"command": command}
            if intent.node:
                shell_args["node"] = intent.node
            contract = (
                build_output_contract(mode="json", json_shape="value")
                if intent.output_mode == "json"
                else build_output_contract(mode=intent.output_mode)
            )
            return CompiledIntentPlan(
                plan=ExecutionPlan.model_validate(
                    {
                        "steps": [
                            {"id": 1, "action": "shell.exec", "args": shell_args, "output": "shell_output"},
                            {
                                "id": 2,
                                "action": "runtime.return",
                                "input": ["shell_output"],
                                "args": {
                                    "value": {"$ref": "shell_output", "path": "stdout"},
                                    "mode": intent.output_mode,
                                    "output_contract": contract,
                                },
                            },
                        ]
                    }
                ),
                metadata={"capability_pack": self.name, "intent_type": type(intent).__name__},
            )

        if isinstance(intent, FileContentDiagnosticIntent):
            _require_tools(context.allowed_tools, "fs.search_content")
            search_step = {
                "id": 1,
                "action": "fs.search_content",
                "args": {
                    "path": intent.path,
                    "needle": intent.needle,
                    "pattern": intent.pattern or "*",
                    "recursive": intent.recursive,
                    "file_only": True,
                    "case_insensitive": False,
                    "path_style": "relative",
                },
                "output": "search_results",
            }
            if intent.result_kind == "files":
                contract = build_output_contract(
                    mode=intent.output_mode,
                    path_style="relative",
                    json_shape="matches" if intent.output_mode == "json" else None,
                )
                value = {"$ref": "search_results", "path": "matches"}
                steps = [
                    search_step,
                    {
                        "id": 2,
                        "action": "runtime.return",
                        "input": ["search_results"],
                        "args": {
                            "value": value,
                            "mode": intent.output_mode,
                            "output_contract": contract,
                        },
                    },
                ]
            else:
                _require_tools(context.allowed_tools, "python.exec")
                steps = [
                    search_step,
                    {
                        "id": 2,
                        "action": "python.exec",
                        "input": ["search_results"],
                        "args": {
                            "inputs": {"entries": {"$ref": "search_results", "path": "entries"}},
                            "code": (
                                "rows = []\n"
                                "for entry in inputs['entries']:\n"
                                "    path = entry.get('relative_path') or entry.get('path') or entry.get('name') or ''\n"
                                "    for line in entry.get('matched_lines', []):\n"
                                "        rows.append({'path': path, 'line_number': line.get('line_number'), 'text': line.get('text', '')})\n"
                                "result = rows"
                            ),
                        },
                        "output": "search_rows",
                    },
                    {
                        "id": 3,
                        "action": "runtime.return",
                        "input": ["search_rows"],
                        "args": {
                            "value": {"$ref": "search_rows", "path": "result"},
                            "mode": intent.output_mode,
                            "output_contract": build_output_contract(mode=intent.output_mode, json_shape="rows"),
                        },
                    },
                ]
            return CompiledIntentPlan(
                plan=ExecutionPlan.model_validate({"steps": steps}),
                metadata={"capability_pack": self.name, "intent_type": type(intent).__name__},
            )

        return None


def _shell_llm_intent_system_prompt() -> str:
    """Handle the internal shell llm intent system prompt helper path for this module.

    Inputs:
        Uses module or instance state; no caller-supplied data parameters are required.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._shell_llm_intent_system_prompt.
    """
    return """You are an intent extractor only.
You convert shell-style user requests into one typed, read-only intent.
You must output JSON only.
You may only choose one of the allowed intent types.
You must not create shell commands.
You must not create tool calls.
You must not create execution plans.
You must not use python.
You must not include argv arrays, command strings, or gateway commands.
Only these read-only request families are supported:
- system inspection summaries
- filesystem aggregation summaries
- filesystem content diagnostics
- fetch title/head extraction
Mutation/admin requests are unsupported.
If the request is unsafe, destructive, or outside the allowed catalog, emit a mutation/unsupported intent or return matched=false.
For host identity, use the hostname-oriented inspection intent.
Use the exact JSON keys matched, intent_type, confidence, arguments, reason."""


def _require_tools(allowed_tools: list[str], *required: str) -> None:
    """Handle the internal require tools helper path for this module.

    Inputs:
        Receives allowed_tools for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._require_tools.
    """
    missing = [tool for tool in required if tool not in allowed_tools]
    if missing:
        raise ValueError(f"Deterministic intent requires unavailable tools: {', '.join(missing)}")


def _looks_like_plain_filesystem_prompt(prompt: str) -> bool:
    """Handle the internal looks like plain filesystem prompt helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._looks_like_plain_filesystem_prompt.
    """
    lower = prompt.lower()
    if "shell" in lower or "terminal" in lower or "bash" in lower:
        return False
    return bool(PATH_CANDIDATE_RE.search(prompt)) and (
        _looks_like_file_aggregate_prompt(prompt) or _looks_like_content_diagnostic_prompt(prompt)
    )


def _looks_like_file_aggregate_prompt(prompt: str) -> bool:
    """Handle the internal looks like file aggregate prompt helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._looks_like_file_aggregate_prompt.
    """
    return SHELL_FILE_AGGREGATE_RE.search(prompt) is not None


def _looks_like_content_diagnostic_prompt(prompt: str) -> bool:
    """Handle the internal looks like content diagnostic prompt helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._looks_like_content_diagnostic_prompt.
    """
    return SHELL_CONTENT_DIAGNOSTIC_RE.search(prompt) is not None and PATH_CANDIDATE_RE.search(prompt) is not None


def _looks_like_fetch_prompt(prompt: str) -> bool:
    """Handle the internal looks like fetch prompt helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._looks_like_fetch_prompt.
    """
    return URL_RE.search(prompt) is not None and SHELL_FETCH_RE.search(prompt) is not None


def _extract_prompt_node(prompt: str, available_nodes: list[str]) -> tuple[str, str | None]:
    """Handle the internal extract prompt node helper path for this module.

    Inputs:
        Receives prompt, available_nodes for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._extract_prompt_node.
    """
    prompt_text = str(prompt or "").strip()
    if not prompt_text:
        return "", None
    match = NODE_FOR_RE.search(prompt_text)
    if match is not None:
        node = match.group(1)
        if node in available_nodes:
            stripped = NODE_FOR_RE.sub("", prompt_text).strip()
            return re.sub(r"\s{2,}", " ", stripped), node
    trailing = TRAILING_NODE_RE.search(prompt_text)
    if trailing is not None:
        node = trailing.group(1)
        if node in available_nodes:
            stripped = TRAILING_NODE_RE.sub("", prompt_text).strip()
            return re.sub(r"\s{2,}", " ", stripped), node
    return prompt_text, None


def _finalize_llm_shell_intent(intent: Any, prompt: str, *, node: str | None, available_nodes: list[str]) -> Any:
    """Handle the internal finalize llm shell intent helper path for this module.

    Inputs:
        Receives intent, prompt, node, available_nodes for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._finalize_llm_shell_intent.
    """
    if isinstance(intent, (ShellExplicitCommandIntent, ShellCommandIntent)):
        raise ValueError("LLM shell intent extraction may not emit raw command intents.")
    if isinstance(intent, (ShellMutationIntent, ShellUnsupportedIntent, ShellUnsupportedMutationIntent)):
        return intent

    if isinstance(intent, ShellInspectionIntent):
        payload = intent.model_dump()
        if not payload.get("node") and node is not None:
            payload["node"] = node
        _validate_shell_string_fields(payload, allow_path=False)
        node_value = str(payload.get("node") or "").strip()
        if node_value:
            if not SAFE_NODE_RE.match(node_value):
                raise ValueError("Shell inspection node contains unsafe characters.")
            if node_value not in available_nodes:
                raise ValueError(f"Shell inspection node {node_value!r} is not allowed.")
        if payload.get("kind") not in ALLOWED_SHELL_KINDS:
            raise ValueError("Unsupported shell inspection kind.")
        return ShellInspectionIntent.model_validate(payload)

    if isinstance(intent, FileContentDiagnosticIntent):
        payload = intent.model_dump()
        if node is not None:
            raise ValueError("Remote node routing is not supported for shell-derived content diagnostics.")
        prompt_path = _extract_prompt_path(prompt)
        payload["path"] = str(payload.get("path") or prompt_path or "").strip()
        if not payload["path"]:
            raise ValueError("Shell-derived content diagnostics require an explicit path.")
        file_query = normalize_file_query(prompt, explicit_path=payload["path"])
        if str(payload.get("pattern") or "*").strip() == "*" and file_query.pattern != "*":
            payload["pattern"] = file_query.pattern
        payload["recursive"] = bool(file_query.recursive)
        if _prompt_wants_matching_lines(prompt):
            payload["result_kind"] = "lines"
        _validate_shell_string_fields(payload, allow_path=True)
        if not str(payload.get("needle") or "").strip():
            raise ValueError("Shell-derived content diagnostics require a non-empty needle.")
        return FileContentDiagnosticIntent.model_validate(payload)

    if isinstance(intent, FileAggregateIntent):
        payload = intent.model_dump()
        if node is not None:
            raise ValueError("Remote node routing is not supported for shell-derived filesystem aggregation.")
        prompt_path = _extract_prompt_path(prompt)
        payload["path"] = str(payload.get("path") or prompt_path or "").strip()
        if not payload["path"]:
            raise ValueError("Shell-derived filesystem aggregation requires an explicit path.")
        file_query = normalize_file_query(prompt, explicit_path=payload["path"])
        if str(payload.get("pattern") or "*").strip() == "*" and file_query.pattern != "*":
            payload["pattern"] = file_query.pattern
        payload["recursive"] = bool(file_query.recursive)
        if re.search(r"\bcount\s+and\s+total\s+size\b", prompt, re.IGNORECASE):
            payload["aggregate"] = "count_and_total_size"
        _validate_shell_string_fields(payload, allow_path=True)
        return FileAggregateIntent.model_validate(payload)

    if isinstance(intent, FetchExtractIntent):
        if node is not None:
            raise ValueError("Remote node routing is not supported for shell-derived fetch extraction.")
        payload = intent.model_dump()
        payload["url"] = str(payload.get("url") or _extract_prompt_url(prompt) or "").strip()
        if not payload["url"]:
            raise ValueError("Shell-derived fetch extraction requires an explicit URL.")
        _validate_shell_string_fields(payload, allow_path=False, allow_url=True)
        return FetchExtractIntent.model_validate(payload)

    raise ValueError(f"Unsupported shell LLM intent: {type(intent).__name__}")


def _llm_capability_owner(intent: Any) -> str:
    """Handle the internal llm capability owner helper path for this module.

    Inputs:
        Receives intent for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._llm_capability_owner.
    """
    if isinstance(intent, FileAggregateIntent):
        return "filesystem"
    if isinstance(intent, FetchExtractIntent):
        return "fetch"
    return "shell"


def _extract_prompt_path(prompt: str) -> str | None:
    """Handle the internal extract prompt path helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._extract_prompt_path.
    """
    match = PATH_CANDIDATE_RE.search(prompt)
    if match is None:
        return None
    return match.group(0).rstrip("?!,:;")


def _extract_prompt_url(prompt: str) -> str | None:
    """Handle the internal extract prompt url helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._extract_prompt_url.
    """
    match = URL_RE.search(prompt)
    if match is None:
        return None
    url = str(match.group(0) or "").rstrip("?!,:;")
    if re.match(r"^(?:https?://|file://)", url, re.IGNORECASE):
        return url
    return f"https://{url}"


def _prompt_wants_matching_lines(prompt: str) -> bool:
    """Handle the internal prompt wants matching lines helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._prompt_wants_matching_lines.
    """
    return bool(re.search(r"\b(?:matching\s+lines?|show\s+lines?|lines?\s+containing)\b", prompt, re.IGNORECASE))


def _validate_shell_string_fields(payload: dict[str, Any], *, allow_path: bool, allow_url: bool = False) -> None:
    """Handle the internal validate shell string fields helper path for this module.

    Inputs:
        Receives payload, allow_path, allow_url for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._validate_shell_string_fields.
    """
    for key, value in payload.items():
        if value is None or not isinstance(value, str):
            continue
        text = value.strip()
        if not text:
            continue
        if key == "path" and allow_path:
            if SHELL_META_RE.search(text):
                raise ValueError("Shell-derived path contains unsafe shell metacharacters.")
            continue
        if key == "url" and allow_url:
            if SHELL_META_RE.search(text):
                raise ValueError("Shell-derived URL contains unsafe shell metacharacters.")
            continue
        if SHELL_META_RE.search(text):
            raise ValueError(f"Shell-derived {key} contains unsafe shell metacharacters.")


def _shell_inspection_command(intent: ShellInspectionIntent) -> str:
    """Handle the internal shell inspection command helper path for this module.

    Inputs:
        Receives intent for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._shell_inspection_command.
    """
    path = _quote_path(intent.path or ".")
    if intent.kind == "uptime":
        return "uptime"
    if intent.kind == "host_identity":
        return "hostname"
    if intent.kind == "disk_usage":
        return "df -h"
    if intent.kind == "directory_size":
        return f"du -sh {path}"
    if intent.kind == "memory_summary":
        return "free -h"
    if intent.kind == "cpu_summary":
        return "lscpu"
    if intent.kind == "current_user":
        return "id -un"
    if intent.kind == "current_working_directory":
        return "pwd"
    if intent.kind in {"process_summary", "process_list"}:
        limit = int(intent.limit or 20)
        return f"ps -eo pid,user,comm,pcpu,pmem --sort=-pcpu | head -n {limit + 1}"
    if intent.kind == "process_search":
        if not intent.process_name:
            raise ValueError("process_search requires process_name.")
        pattern = shlex.quote(intent.process_name)
        limit = int(intent.limit or 20)
        return f"ps -eo pid,user,comm,args,pcpu,pmem | grep -i -- {pattern} | head -n {limit}"
    if intent.kind == "port_usage":
        if intent.port is None:
            raise ValueError("port_usage requires port.")
        return f"lsof -i :{int(intent.port)}"
    if intent.kind == "network_summary":
        return "ip addr"
    if intent.kind == "listening_ports":
        return "ss -tuln"
    if intent.kind == "mounts":
        return "findmnt"
    if intent.kind == "service_status":
        if not intent.service:
            raise ValueError("service_status requires service.")
        return f"systemctl status {shlex.quote(intent.service)} --no-pager"
    if intent.kind == "recent_logs":
        limit = int(intent.limit or 50)
        return f"journalctl -n {limit} --no-pager"
    if intent.kind == "large_files":
        limit = int(intent.limit or 20)
        return f"find {path} -type f -printf '%s %p\\n' | sort -nr | head -n {limit}"
    if intent.kind == "directory_listing":
        return f"ls -la {path}"
    raise ValueError(f"Unsupported shell inspection kind: {intent.kind}")


def _classify_shell_goal(goal: str, context: ClassificationContext) -> IntentResult:
    """Handle the internal classify shell goal helper path for this module.

    Inputs:
        Receives goal, context for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._classify_shell_goal.
    """
    prompt = str(goal or "").strip()
    if not prompt:
        return IntentResult(matched=False, reason="shell_empty_goal")

    mutation = _classify_shell_mutation_prompt(prompt)
    if mutation.matched:
        return mutation

    explicit = _extract_explicit_command(prompt)
    if explicit:
        command, node = _extract_prompt_node(explicit, context.settings.available_nodes)
        if not command:
            return IntentResult(matched=False, reason="shell_explicit_empty_command")
        policy = classify_shell_command(
            command,
            mode=context.settings.shell_mode,
            allow_mutation_with_approval=context.settings.shell_allow_mutation_with_approval
            or context.settings.allow_destructive_shell,
        )
        if not policy.allowed:
            return IntentResult(
                matched=True,
                intent=ShellMutationIntent(
                    operation=_operation_from_policy(policy.detected_operations),
                    target=policy.redacted_command or command,
                    reason=policy.reason,
                    requires_approval=policy.requires_approval,
                ),
                metadata={"shell_risk": policy.risk, "shell_policy_reason": policy.reason},
            )
        return IntentResult(
            matched=True,
            intent=ShellExplicitCommandIntent(command=command, node=node, output_mode="json" if _wants_json(prompt) else "text"),
            metadata={"shell_risk": policy.risk, "shell_policy_reason": policy.reason},
        )

    inspection = _classify_shell_inspection_prompt(prompt)
    if inspection is not None:
        return IntentResult(matched=True, intent=inspection, metadata={"capability": "shell", "planning_mode": "deterministic_intent"})

    return IntentResult(matched=False, reason="shell_no_deterministic_match")


def _extract_explicit_command(prompt: str) -> str:
    """Handle the internal extract explicit command helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._extract_explicit_command.
    """
    match = EXPLICIT_COMMAND_RE.match(prompt)
    if match is None:
        return ""
    command = str(match.group("command") or match.group("prefixed") or "").strip()
    if re.match(r"^(?:a\s+)?shell\s+command\s+that\b", command, re.IGNORECASE):
        return ""
    if re.match(r"^(?:print|prints|output)\b", command, re.IGNORECASE) and re.search(
        r"\b(?:newline|separate\s+lines|return)\b", command, re.IGNORECASE
    ):
        return ""
    if not _looks_like_explicit_shell_command(command):
        return ""
    return command


def _classify_shell_inspection_prompt(prompt: str) -> ShellInspectionIntent | None:
    """Handle the internal classify shell inspection prompt helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._classify_shell_inspection_prompt.
    """
    text = prompt.lower()
    path = _extract_prompt_path(prompt) or ("." if _current_scope_mentioned(prompt) else None)

    service_match = SERVICE_STATUS_RE.search(prompt)
    if service_match is not None:
        return ShellInspectionIntent(kind="service_status", service=service_match.group("service"))
    if re.search(r"\b(?:check|show)\s+disk\s+usage\b|\bdf\b", text):
        return ShellInspectionIntent(kind="disk_usage")
    if re.search(r"\b(?:show|check)\s+(?:memory|ram)(?:\s+usage|\s+summary)?\b|\bfree\s+memory\b", text):
        return ShellInspectionIntent(kind="memory_summary")
    if re.search(r"\b(?:show|check)\s+cpu(?:\s+summary|\s+info)?\b|\blscpu\b", text):
        return ShellInspectionIntent(kind="cpu_summary")
    if re.search(r"\buptime\b", text):
        return ShellInspectionIntent(kind="uptime")
    if re.search(r"\b(?:hostname|host\s*name|host identity)\b", text):
        return ShellInspectionIntent(kind="host_identity")
    if re.search(r"\b(?:whoami|current user|which user)\b", text):
        return ShellInspectionIntent(kind="current_user")
    if re.search(r"\b(?:pwd|working directory|current directory)\b", text) and "file" not in text and "entries" not in text:
        return ShellInspectionIntent(kind="current_working_directory")
    if re.search(r"\b(?:directory|folder)\s+size\b|\bdu\s+-sh\b", text):
        return ShellInspectionIntent(kind="directory_size", path=path or ".")
    port = _extract_port(prompt)
    if port is not None and re.search(r"\b(?:using|listening|open|process|port|lsof)\b", text):
        return ShellInspectionIntent(kind="port_usage", port=port)
    if re.search(r"\b(?:listening\s+ports|open\s+ports|ss\s+-tuln)\b", text):
        return ShellInspectionIntent(kind="listening_ports")
    if re.search(r"\b(?:top\s+processes|process\s+list|show\s+processes|running\s+processes)\b", text):
        return ShellInspectionIntent(kind="process_list", limit=_extract_limit(prompt) or 20)
    process_match = PROCESS_SEARCH_RE.search(prompt)
    if process_match is not None:
        return ShellInspectionIntent(kind="process_search", process_name=process_match.group("name"), limit=_extract_limit(prompt) or 20)
    if re.search(r"\b(?:network\s+summary|ip\s+addr|network\s+interfaces)\b", text):
        return ShellInspectionIntent(kind="network_summary")
    if re.search(r"\b(?:mounts?|mounted\s+filesystems?|findmnt)\b", text):
        return ShellInspectionIntent(kind="mounts")
    if re.search(r"\b(?:recent\s+logs?|journalctl)\b", text):
        return ShellInspectionIntent(kind="recent_logs", limit=_extract_limit(prompt) or 50)
    if re.search(r"\b(?:large|largest|biggest)\s+files?\b", text):
        return ShellInspectionIntent(kind="large_files", path=path or ".", limit=_extract_limit(prompt) or 20)
    return None


def _classify_shell_mutation_prompt(prompt: str) -> IntentResult:
    """Handle the internal classify shell mutation prompt helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._classify_shell_mutation_prompt.
    """
    text = prompt.lower()
    explicit = _extract_explicit_command(prompt)
    if explicit:
        return IntentResult(matched=False, reason="shell_explicit_classified_later")
    patterns: list[tuple[str, str]] = [
        ("delete", r"\b(?:delete|remove|rm)\b"),
        ("kill_process", r"\b(?:kill|pkill)\b"),
        ("chmod", r"\bchmod\b"),
        ("chown", r"\bchown\b"),
        ("restart_service", r"\b(?:restart|reload)\s+(?:service|systemctl|daemon)\b|\bsystemctl\s+(?:restart|reload)\b"),
        ("stop_service", r"\b(?:stop)\s+(?:service|systemctl|daemon)\b|\bsystemctl\s+stop\b"),
        ("start_service", r"\b(?:start)\s+(?:service|systemctl|daemon)\b|\bsystemctl\s+start\b"),
        ("install_package", r"\b(?:apt|apt-get|yum|pip)\s+install\b|\binstall\s+package\b"),
        ("modify_system", r"\b(?:sudo|mkfs|dd|shutdown|reboot|poweroff)\b"),
    ]
    for operation, pattern in patterns:
        match = re.search(pattern, text)
        if match is None:
            continue
        return IntentResult(
            matched=True,
            intent=ShellMutationIntent(
                operation=operation,  # type: ignore[arg-type]
                target=_mutation_target(prompt),
                reason="This appears to be a mutating or high-risk shell/system request. Only read-only inspection runs automatically.",
            ),
            metadata={"shell_mutation": operation},
        )
    return IntentResult(matched=False, reason="shell_mutation_no_match")


def _compile_shell_refusal(*, reason: str, risk: str, command: str, approval_required: bool) -> CompiledIntentPlan:
    """Handle the internal compile shell refusal helper path for this module.

    Inputs:
        Receives reason, risk, command, approval_required for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._compile_shell_refusal.
    """
    return CompiledIntentPlan(
        plan=ExecutionPlan.model_validate(
            {
                "steps": [
                    {
                        "id": 1,
                        "action": "runtime.return",
                        "args": {
                            "value": _shell_refusal_markdown(
                                reason=reason,
                                operation=command,
                                risk=risk,
                                approval_required=approval_required,
                            ),
                            "mode": "text",
                            "output_contract": build_output_contract(mode="text"),
                        },
                    }
                ]
            }
        ),
        metadata={"capability_pack": "shell", "intent_type": "ShellMutationIntent"},
    )


def _shell_refusal_markdown(
    *,
    reason: str,
    operation: str,
    risk: str | None = None,
    approval_required: bool | None = None,
) -> str:
    """Handle the internal shell refusal markdown helper path for this module.

    Inputs:
        Receives reason, operation, risk, approval_required for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._shell_refusal_markdown.
    """
    status = "requires approval" if approval_required else "was not executed"
    risk_line = f"\n\nRisk: {risk}." if risk else ""
    return (
        "Request Not Executed\n\n"
        f"The shell request {status}: {operation}.\n\n"
        f"{reason}.{risk_line}\n\n"
        "Safer read-only alternatives:\n"
        "- List matching files before changing them.\n"
        "- Show disk usage or directory size.\n"
        "- Inspect processes or service status without modifying them."
    )


def _quote_path(path: str) -> str:
    """Handle the internal quote path helper path for this module.

    Inputs:
        Receives path for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._quote_path.
    """
    return shlex.quote(str(path or ".").strip() or ".")


def _current_scope_mentioned(prompt: str) -> bool:
    """Handle the internal current scope mentioned helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._current_scope_mentioned.
    """
    return re.search(r"\b(?:this|current)\s+(?:folder|directory)\b|\bhere\b|(?:^|\s)\.(?:\s|$|[?!.,;:])", prompt, re.IGNORECASE) is not None


def _extract_port(prompt: str) -> int | None:
    """Handle the internal extract port helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._extract_port.
    """
    match = PORT_RE.search(prompt)
    if match is None:
        return None
    raw = match.group("port") or match.group("colon_port")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if 1 <= value <= 65535:
        return value
    return None


def _extract_limit(prompt: str) -> int | None:
    """Handle the internal extract limit helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._extract_limit.
    """
    match = re.search(r"\b(?:top|first|last|limit)\s+(\d{1,3})\b|\b-n\s+(\d{1,3})\b", prompt, re.IGNORECASE)
    if match is None:
        return None
    value = int(match.group(1) or match.group(2))
    return max(1, min(200, value))


def _wants_json(prompt: str) -> bool:
    """Handle the internal wants json helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._wants_json.
    """
    return re.search(r"\bjson\b", prompt, re.IGNORECASE) is not None


def _mutation_target(prompt: str) -> str | None:
    """Handle the internal mutation target helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._mutation_target.
    """
    text = re.sub(r"^\s*(?:please\s+)?(?:delete|remove|kill|chmod|chown|restart|stop|start)\b", "", prompt, flags=re.IGNORECASE).strip()
    return text or None


def _operation_from_policy(detected: list[str]) -> str:
    """Handle the internal operation from policy helper path for this module.

    Inputs:
        Receives detected for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._operation_from_policy.
    """
    ops = set(detected)
    if "rm" in ops:
        return "delete"
    if ops & {"kill", "pkill"}:
        return "kill_process"
    if "chmod" in ops:
        return "chmod"
    if "chown" in ops:
        return "chown"
    if "systemctl" in ops or "service" in ops:
        return "modify_system"
    return "unknown_mutation"


def _looks_like_explicit_shell_command(command: str) -> bool:
    """Handle the internal looks like explicit shell command helper path for this module.

    Inputs:
        Receives command for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.shell._looks_like_explicit_shell_command.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens:
        return False
    first = tokens[0].rsplit("/", 1)[-1].lower()
    known = SAFE_READ_ONLY_COMMANDS | MUTATING_COMMANDS | FORBIDDEN_COMMANDS
    if first in known:
        return True
    if tokens[0].startswith(("./", "../", "/")):
        return True
    return False
