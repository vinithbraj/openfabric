from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from aor_runtime.runtime.tool_surfaces import friendly_label_for_tool


TraceMode = str


_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth(?:orization)?|bearer|cookie|password|passwd|private[_-]?key|secret|token)\b"
    r"\s*[:=]\s*"
    r"([^\s,;]+)"
)
_BEARER_TOKEN_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_LONG_LITERAL_RE = re.compile(r"(['\"])([^'\"]{80,})(['\"])")
_WHITESPACE_RE = re.compile(r"\s+")


def resolve_openwebui_trace_mode(settings: Any | None) -> TraceMode:
    raw_mode = str(getattr(settings, "openwebui_trace_mode", "") or "").strip().lower()
    if raw_mode in {"off", "summary", "diagnostic"}:
        return raw_mode

    legacy_enabled = any(
        bool(getattr(settings, name, False))
        for name in ("show_planner_events", "show_tool_events", "show_validation_events")
    )
    return "summary" if legacy_enabled else "off"


@dataclass
class OpenWebUITraceRenderer:
    mode: TraceMode = "off"
    max_detail_chars: int = 240
    _planner_started_count: int = 0
    _last_plan_signature: str = ""
    _started_steps: set[tuple[int, str]] = field(default_factory=set)
    _completed_steps: set[tuple[int, str]] = field(default_factory=set)

    @classmethod
    def from_settings(cls, settings: Any | None) -> "OpenWebUITraceRenderer":
        return cls(mode=resolve_openwebui_trace_mode(settings))

    def render(self, event: dict[str, Any]) -> str | None:
        if self.mode == "off":
            return None

        event_type = str(event.get("event_type") or "")
        payload = dict(event.get("payload") or {})

        if event_type == "planner.started":
            return self._render_planner_started(payload)
        if event_type == "planner.completed":
            return self._render_planner_completed(payload)
        if event_type == "executor.step.started":
            return self._render_step_started(payload)
        if event_type == "executor.step.completed":
            return self._render_step_completed(payload)
        if event_type == "validator.started":
            return _trace_progress(["Checking result..."])
        if event_type == "validator.completed":
            return self._render_validation_completed(payload)
        if event_type == "validator.result_shape":
            return self._render_result_shape(payload)
        if event_type == "executor.step.awaiting_confirmation":
            return _trace_progress(["Waiting for approval before running a higher-risk step."])
        if event_type.endswith(".failed"):
            return self._render_failed(event_type, payload)
        return None

    @property
    def diagnostic(self) -> bool:
        return self.mode == "diagnostic"

    def _render_planner_started(self, payload: dict[str, Any]) -> str:
        self._planner_started_count += 1
        attempt = _to_int(payload.get("attempt")) or self._planner_started_count
        if attempt <= 1:
            return _trace_progress(["Thinking..."])
        return _trace_progress([f"Repair attempt {attempt - 1}: replanning..."])

    def _render_planner_completed(self, payload: dict[str, Any]) -> str:
        steps = _plan_steps(payload)
        signature = "|".join(steps)
        changed = signature and signature != self._last_plan_signature
        if signature:
            self._last_plan_signature = signature

        lines = ["`Planning complete.`"]
        if steps and changed:
            lines.extend(["", "**Plan Overview**"])
            lines.append(" -> ".join(f"`{_inline_code(step)}`" for step in steps[:8]))
            if len(steps) > 8:
                lines.append(f"`{len(steps) - 8} more steps hidden.`")

        if self.diagnostic:
            repairs = [sanitize_detail(item, max_chars=self.max_detail_chars) for item in _as_list(payload.get("repair_trace"))]
            if repairs:
                lines.extend([""])
                lines.extend(f"`Repair: {_inline_code(repair)}`" for repair in repairs[:5] if repair)
        return _trace_markdown(lines)

    def _render_step_started(self, payload: dict[str, Any]) -> str | None:
        step = dict(payload.get("step") or {})
        step_index = _to_int(payload.get("step_index")) or _to_int(step.get("id")) or 0
        tool = str(step.get("action") or "step").strip()
        marker = (step_index, tool)
        if marker in self._started_steps:
            return None
        self._started_steps.add(marker)

        if tool == "runtime.return":
            return _trace_progress(["Preparing final response..."])

        args = dict(step.get("args") or {})
        label = sanitize_detail(friendly_label_for_tool(tool, args), max_chars=120)
        detail = _step_detail(tool, args, payload, max_chars=self.max_detail_chars)
        if detail:
            return _trace_progress([f"Running {label}: {detail}"])
        return _trace_progress([f"Running {label}..."])

    def _render_step_completed(self, payload: dict[str, Any]) -> str | None:
        step = dict(payload.get("step") or {})
        step_index = _to_int(step.get("id")) or 0
        tool = str(step.get("action") or "").strip()
        marker = (step_index, tool)
        if marker in self._completed_steps:
            return None
        self._completed_steps.add(marker)

        if not bool(payload.get("success")):
            return _trace_progress(
                [f"Step failed: {sanitize_detail(payload.get('error') or 'tool execution failed', max_chars=self.max_detail_chars)}"],
            )
        if tool == "runtime.return":
            return None

        result = dict(payload.get("result") or {})
        summary = _result_summary(tool, result, max_chars=self.max_detail_chars)
        return _trace_progress([summary]) if summary else None

    def _render_validation_completed(self, payload: dict[str, Any]) -> str | None:
        result = dict(payload.get("result") or {})
        if bool(result.get("success")):
            return _trace_progress(["Checks passed."]) if self.diagnostic else None
        reason = sanitize_detail(result.get("reason") or "validation failed", max_chars=self.max_detail_chars)
        return _trace_progress([f"Validation failed: {reason}. Repairing if possible..."])

    def _render_result_shape(self, payload: dict[str, Any]) -> str | None:
        if bool(payload.get("success")):
            return _trace_progress(["Result shape verified."]) if self.diagnostic else None
        reason = sanitize_detail(payload.get("reason") or "result shape did not match the request", max_chars=self.max_detail_chars)
        return _trace_progress([f"Result check failed: {reason}. Repairing if possible..."])

    def _render_failed(self, event_type: str, payload: dict[str, Any]) -> str:
        phase = event_type.split(".", 1)[0].replace("_", " ").title()
        error = sanitize_detail(payload.get("error") or "Task failed.", max_chars=self.max_detail_chars)
        return _trace_progress([f"{phase} failed: {error}"])


def _trace_progress(body: list[str]) -> str:
    lines = [f"`{_inline_code(item)}`" for item in body if str(item or "")]
    return _trace_markdown(lines)


def _trace_markdown(lines: str | list[str], *, code: bool = False) -> str:
    raw_lines = [lines] if isinstance(lines, str) else list(lines)
    rendered: list[str] = []
    for line in raw_lines:
        text = str(line)
        if code and text:
            text = f"`{_inline_code(text)}`"
        rendered.append(">" if not text else f"> {text}")
    return "\n".join(rendered).rstrip() + "\n\n"


def _inline_code(value: Any) -> str:
    return str(value or "").replace("`", "'")


def sanitize_detail(value: Any, *, max_chars: int = 240) -> str:
    text = str(value or "")
    text = text.replace("\x00", "")
    text = _BEARER_TOKEN_RE.sub("Bearer <redacted>", text)
    text = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=<redacted>", text)
    text = _LONG_LITERAL_RE.sub(r"\1<redacted>\3", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "..."
    return text


def _plan_steps(payload: dict[str, Any]) -> list[str]:
    high_level = [sanitize_detail(item, max_chars=140) for item in _as_list(payload.get("high_level_plan"))]
    if high_level:
        return [item for item in high_level if item]

    execution_plan = dict(payload.get("execution_plan") or {})
    raw_steps = _as_list(execution_plan.get("steps") or payload.get("steps"))
    steps: list[str] = []
    for item in raw_steps:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("action") or "step").strip()
        args = dict(item.get("args") or {})
        steps.append(_friendly_step(tool, args))
    return [step for step in steps if step]


def _friendly_step(tool: str, args: dict[str, Any]) -> str:
    return sanitize_detail(friendly_label_for_tool(tool, args), max_chars=120)


def _step_detail(tool: str, args: dict[str, Any], payload: dict[str, Any], *, max_chars: int) -> str:
    if tool in {"sql.query", "sql.validate"}:
        database = str(args.get("database") or "").strip()
        sql = sanitize_sql(args.get("query") or payload.get("command") or "", max_chars=max_chars)
        return f"{database} / {sql}" if database and sql else database or sql
    if tool == "shell.exec":
        return sanitize_detail(args.get("command") or payload.get("command") or "", max_chars=max_chars)
    if tool == "fs.write":
        return sanitize_detail(args.get("path") or "", max_chars=max_chars)
    if tool in {"fs.read", "fs.list", "fs.find", "fs.glob", "fs.size", "fs.aggregate"}:
        bits = [sanitize_detail(args.get("path") or "", max_chars=120)]
        if args.get("pattern"):
            bits.append(f"pattern={sanitize_detail(args.get('pattern'), max_chars=80)}")
        return " ".join(bit for bit in bits if bit)
    if tool == "text.format":
        return sanitize_detail(args.get("format") or "txt", max_chars=40)
    return ""


def sanitize_sql(value: Any, *, max_chars: int = 240) -> str:
    text = sanitize_detail(value, max_chars=max_chars * 2)
    text = re.sub(r"(?s)'[^']{48,}'", "'<redacted>'", text)
    text = re.sub(r'(?s)"[^"]{80,}"', '"<redacted>"', text)
    return sanitize_detail(text, max_chars=max_chars)


def _result_summary(tool: str, result: dict[str, Any], *, max_chars: int) -> str:
    if tool == "sql.validate":
        valid = result.get("valid")
        if isinstance(valid, bool):
            return f"SQL validation: {'valid' if valid else 'invalid'}"
    if tool == "sql.query":
        row_count = result.get("row_count")
        if row_count is None and isinstance(result.get("rows"), list):
            row_count = len(result["rows"])
        if row_count is not None:
            return f"Rows returned: {row_count}"
    if tool == "text.format":
        output_format = str(result.get("format") or "").strip()
        content = str(result.get("content") or "")
        if content:
            suffix = f" as {output_format}" if output_format else ""
            return f"Formatted output{suffix}: {len(content)} characters"
    if tool == "fs.write":
        path = sanitize_detail(result.get("path") or "", max_chars=max_chars)
        bytes_written = result.get("bytes_written")
        if path and bytes_written is not None:
            return f"File written: {path} ({bytes_written} bytes)"
        if path:
            return f"File written: {path}"
    if tool == "shell.exec":
        exit_code = result.get("exit_code", result.get("returncode"))
        if exit_code is not None:
            return f"Command finished with exit code {exit_code}"
    if tool.startswith("slurm."):
        for key in ("row_count", "count", "node_count", "job_count"):
            if result.get(key) is not None:
                return f"SLURM result: {key.replace('_', ' ')} {result[key]}"
        for key in ("nodes", "jobs", "partitions"):
            if isinstance(result.get(key), list):
                return f"SLURM result: {len(result[key])} {key}"
    if tool.startswith("fs."):
        for key in ("entries", "matches", "files"):
            if isinstance(result.get(key), list):
                return f"Filesystem result: {len(result[key])} {key}"
        if result.get("path"):
            return f"Filesystem result: {sanitize_detail(result.get('path'), max_chars=max_chars)}"
    return ""


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
