"""OpenFABRIC Runtime Module: aor_runtime.runtime.response_stats

Purpose:
    Build compact runtime statistics and DAG summaries for final responses.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

import re
from typing import Any

from aor_runtime.runtime.markdown import table as md_table


STATS_HEADING_RE = re.compile(r"(^|\n)##\s+Stats\b", re.IGNORECASE)
INTERNAL_TOOLS = {"runtime.return"}
FORMATTER_TOOLS = {"text.format"}


def append_response_stats(
    content: str,
    *,
    state: dict[str, Any],
    metrics: dict[str, Any],
    status: str,
    enabled: bool = True,
) -> str:
    """Append response stats for the surrounding runtime workflow.

    Inputs:
        Receives content, state, metrics, status, enabled for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.response_stats.append_response_stats.
    """
    body = str(content or "")
    if not enabled or not body.strip() or STATS_HEADING_RE.search(body):
        return body

    steps = _plan_steps(state)
    if not steps:
        steps = _history_steps(state)

    markdown_parts = [body.rstrip(), "", "---", "", "## Stats", "", *_stats_table(steps, metrics, status)]
    dag_lines = _dag_lines(steps)
    if dag_lines:
        markdown_parts.extend(["", "## DAG Steps", "", *dag_lines])
    return "\n".join(markdown_parts).strip()


def _stats_table(steps: list[dict[str, Any]], metrics: dict[str, Any], status: str) -> list[str]:
    """Handle the internal stats table helper path for this module.

    Inputs:
        Receives steps, metrics, status for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.response_stats._stats_table.
    """
    tools = _unique([step["action"] for step in steps if step.get("action")])
    primary_tools = [tool for tool in tools if tool not in INTERNAL_TOOLS | FORMATTER_TOOLS]
    displayed_tools = primary_tools or [tool for tool in tools if tool not in INTERNAL_TOOLS] or tools
    llm_calls = _coerce_int(metrics.get("llm_calls"))
    prompt_tokens = _coerce_int(metrics.get("llm_prompt_tokens") or metrics.get("prompt_tokens") or metrics.get("tokens_in"))
    completion_tokens = _coerce_int(
        metrics.get("llm_completion_tokens") or metrics.get("completion_tokens") or metrics.get("tokens_out")
    )
    rows = [
        ("Backend", _backend_label(tools)),
        ("Tools", ", ".join(displayed_tools) if displayed_tools else "Runtime"),
        ("Status", _title(status)),
        ("Time Taken", _format_duration_ms(metrics.get("latency_ms"))),
        ("LLM Passes", str(llm_calls)),
        ("Tokens In", _token_value(prompt_tokens, llm_calls)),
        ("Tokens Out", _token_value(completion_tokens, llm_calls)),
        ("Steps", str(len(steps))),
    ]
    return md_table(["Field", "Value"], [[_code_cell(key), _code_cell(value)] for key, value in rows])


def _dag_lines(steps: list[dict[str, Any]]) -> list[str]:
    """Handle the internal dag lines helper path for this module.

    Inputs:
        Receives steps for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.response_stats._dag_lines.
    """
    lines: list[str] = []
    for index, step in enumerate(steps, start=1):
        action = str(step.get("action") or "unknown")
        output = str(step.get("output") or "").strip()
        suffix = f" -> `{output}`" if output else ""
        lines.append(f"{index}. `{action}`{suffix}")
    return lines


def _plan_steps(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Handle the internal plan steps helper path for this module.

    Inputs:
        Receives state for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.response_stats._plan_steps.
    """
    plan = state.get("plan")
    if not isinstance(plan, dict):
        return []
    raw_steps = plan.get("steps")
    if not isinstance(raw_steps, list):
        return []
    steps: list[dict[str, Any]] = []
    for raw in raw_steps:
        if not isinstance(raw, dict):
            continue
        action = str(raw.get("action") or "").strip()
        if not action:
            continue
        steps.append(
            {
                "id": raw.get("id"),
                "action": action,
                "output": raw.get("output"),
            }
        )
    return steps


def _history_steps(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Handle the internal history steps helper path for this module.

    Inputs:
        Receives state for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.response_stats._history_steps.
    """
    raw_history = state.get("attempt_history") or state.get("history") or []
    if not isinstance(raw_history, list):
        return []
    steps: list[dict[str, Any]] = []
    for item in raw_history:
        if not isinstance(item, dict):
            continue
        step = item.get("step")
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "").strip()
        if not action:
            continue
        steps.append(
            {
                "id": step.get("id"),
                "action": action,
                "output": step.get("output"),
            }
        )
    return steps


def _backend_label(tools: list[str]) -> str:
    """Handle the internal backend label helper path for this module.

    Inputs:
        Receives tools for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.response_stats._backend_label.
    """
    labels: list[str] = []
    if any(tool.startswith("sql.") for tool in tools):
        labels.append("SQL database")
    if any(tool.startswith("slurm.") for tool in tools):
        labels.append("SLURM gateway")
    if any(tool.startswith("fs.") for tool in tools):
        labels.append("Filesystem")
    if any(tool == "shell.exec" for tool in tools):
        labels.append("Shell")
    if any(tool.startswith("fetch.") for tool in tools):
        labels.append("Fetch")
    if any(tool in FORMATTER_TOOLS for tool in tools) and not labels:
        labels.append("Local formatter")
    if not labels:
        labels.append("Runtime")
    return ", ".join(_unique(labels))


def _format_duration_ms(value: Any) -> str:
    """Handle the internal format duration ms helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.response_stats._format_duration_ms.
    """
    try:
        milliseconds = float(value or 0)
    except Exception:  # noqa: BLE001
        milliseconds = 0.0
    if milliseconds <= 0:
        return "Unavailable"
    seconds = milliseconds / 1000.0
    if seconds < 1:
        return f"{milliseconds:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes = int(seconds // 60)
    remaining = seconds - (minutes * 60)
    return f"{minutes}m {remaining:.1f}s"


def _token_value(value: int, llm_calls: int) -> str:
    """Handle the internal token value helper path for this module.

    Inputs:
        Receives value, llm_calls for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.response_stats._token_value.
    """
    if value > 0:
        return f"{value:,}"
    return "Unavailable" if llm_calls > 0 else "0"


def _title(value: Any) -> str:
    """Handle the internal title helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.response_stats._title.
    """
    text = str(value or "").strip().replace("_", " ")
    return text.title() if text else "Unknown"


def _code_cell(value: Any) -> str:
    """Handle the internal code cell helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.response_stats._code_cell.
    """
    text = str(value if value is not None else "").replace("`", "'").strip()
    return f"`{text or '-'}`"


def _unique(values: list[str]) -> list[str]:
    """Handle the internal unique helper path for this module.

    Inputs:
        Receives values for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.response_stats._unique.
    """
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique_values.append(text)
    return unique_values


def _coerce_int(value: Any) -> int:
    """Handle the internal coerce int helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.response_stats._coerce_int.
    """
    try:
        return max(0, int(value or 0))
    except Exception:  # noqa: BLE001
        return 0
