from __future__ import annotations

import re
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


SHELL_FRAMING_RE = re.compile(r"\b(?:using\s+shell|with\s+shell|shell:|bash|terminal|command\s+line)\b", re.IGNORECASE)
SYSTEM_INSPECTION_RE = re.compile(
    r"\b(?:uptime|host\s*name|hostname|whoami|disk\s+usage|memory\s+usage|memory\s+summary|top\s+processes|process(?:es)?|network(?:\s+summary)?|listening\s+ports|open\s+ports|system\s+load)\b",
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

ALLOWED_SHELL_KINDS = {
    "uptime",
    "host_identity",
    "disk_usage",
    "memory_summary",
    "process_summary",
    "network_summary",
    "listening_ports",
}


class ShellInspectionIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "uptime",
        "host_identity",
        "disk_usage",
        "memory_summary",
        "process_summary",
        "network_summary",
        "listening_ports",
    ]
    node: str | None = None
    limit: int | None = Field(default=20, ge=1, le=200)
    output_mode: Literal["text", "csv", "json"] = "text"


class FileContentDiagnosticIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    needle: str
    pattern: str = "*"
    recursive: bool = True
    result_kind: Literal["files", "lines"] = "files"
    output_mode: Literal["text", "csv", "json"] = "text"


class ShellUnsupportedMutationIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: str
    reason: str


class ShellCapabilityPack(CapabilityPack):
    name = "shell"
    supports_llm_intent_extraction = True
    intent_types = (
        ShellCommandIntent,
        ShellInspectionIntent,
        FileContentDiagnosticIntent,
        ShellUnsupportedMutationIntent,
    )

    def classify(self, goal: str, context: ClassificationContext) -> IntentResult:
        result = classify_single_intent(goal, schema_payload=context.schema_payload)
        if result.matched and isinstance(result.intent, self.intent_types):
            return result
        return IntentResult(matched=False, reason=f"{self.name}_no_match")

    def is_llm_intent_domain(self, goal: str, context: ClassificationContext) -> bool:
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
        if isinstance(intent, ShellCommandIntent):
            return CompiledIntentPlan(
                plan=compile_intent_to_plan(intent, context.allowed_tools, context.settings),
                metadata={"capability_pack": self.name, "intent_type": type(intent).__name__},
            )

        if isinstance(intent, ShellUnsupportedMutationIntent):
            return CompiledIntentPlan(
                plan=ExecutionPlan.model_validate(
                    {
                        "steps": [
                            {
                                "id": 1,
                                "action": "runtime.return",
                                "args": {
                                    "value": f"{intent.reason} Unsupported request: {intent.operation}.",
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
If the request is unsafe or outside the allowed catalog, return matched=false.
For host identity, use the hostname-oriented inspection intent.
Use the exact JSON keys matched, intent_type, confidence, arguments, reason."""


def _require_tools(allowed_tools: list[str], *required: str) -> None:
    missing = [tool for tool in required if tool not in allowed_tools]
    if missing:
        raise ValueError(f"Deterministic intent requires unavailable tools: {', '.join(missing)}")


def _looks_like_plain_filesystem_prompt(prompt: str) -> bool:
    lower = prompt.lower()
    if "shell" in lower or "terminal" in lower or "bash" in lower:
        return False
    return bool(PATH_CANDIDATE_RE.search(prompt)) and (
        _looks_like_file_aggregate_prompt(prompt) or _looks_like_content_diagnostic_prompt(prompt)
    )


def _looks_like_file_aggregate_prompt(prompt: str) -> bool:
    return SHELL_FILE_AGGREGATE_RE.search(prompt) is not None


def _looks_like_content_diagnostic_prompt(prompt: str) -> bool:
    return SHELL_CONTENT_DIAGNOSTIC_RE.search(prompt) is not None and PATH_CANDIDATE_RE.search(prompt) is not None


def _looks_like_fetch_prompt(prompt: str) -> bool:
    return URL_RE.search(prompt) is not None and SHELL_FETCH_RE.search(prompt) is not None


def _extract_prompt_node(prompt: str, available_nodes: list[str]) -> tuple[str, str | None]:
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
    if isinstance(intent, FileAggregateIntent):
        return "filesystem"
    if isinstance(intent, FetchExtractIntent):
        return "fetch"
    return "shell"


def _extract_prompt_path(prompt: str) -> str | None:
    match = PATH_CANDIDATE_RE.search(prompt)
    if match is None:
        return None
    return match.group(0).rstrip("?!,:;")


def _extract_prompt_url(prompt: str) -> str | None:
    match = URL_RE.search(prompt)
    if match is None:
        return None
    url = str(match.group(0) or "").rstrip("?!,:;")
    if re.match(r"^(?:https?://|file://)", url, re.IGNORECASE):
        return url
    return f"https://{url}"


def _prompt_wants_matching_lines(prompt: str) -> bool:
    return bool(re.search(r"\b(?:matching\s+lines?|show\s+lines?|lines?\s+containing)\b", prompt, re.IGNORECASE))


def _validate_shell_string_fields(payload: dict[str, Any], *, allow_path: bool, allow_url: bool = False) -> None:
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
    if intent.kind == "uptime":
        return "uptime"
    if intent.kind == "host_identity":
        return "hostname"
    if intent.kind == "disk_usage":
        return "df -h"
    if intent.kind == "memory_summary":
        return "free -h"
    if intent.kind == "process_summary":
        limit = int(intent.limit or 20)
        return f"ps -eo pid,user,comm,pcpu,pmem --sort=-pcpu | head -n {limit + 1}"
    if intent.kind == "network_summary":
        return "ip addr"
    if intent.kind == "listening_ports":
        return "ss -tuln"
    raise ValueError(f"Unsupported shell inspection kind: {intent.kind}")
