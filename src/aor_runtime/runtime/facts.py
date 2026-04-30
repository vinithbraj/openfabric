"""OpenFABRIC Runtime Module: aor_runtime.runtime.facts

Purpose:
    Collect sanitized runtime facts that can be used by presenters and optional LLM summaries.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from aor_runtime.runtime.presentation import strip_internal_telemetry
from aor_runtime.runtime.slurm_result_normalizer import NormalizedSlurmResult, SlurmResultKind, normalize_slurm_result


FORBIDDEN_KEY_PARTS = {
    "password",
    "token",
    "secret",
    "credential",
    "api_key",
    "raw_output",
    "stdout",
    "stderr",
    "semantic_frame",
    "coverage",
    "telemetry",
    "environment",
    "env",
    "payload",
}

FORBIDDEN_EXACT_KEYS = {
    "rows",
    "raw_rows",
    "raw_jobs",
    "raw_nodes",
    "raw_partitions",
    "matches",
    "entries",
}


@dataclass
class FactBuildContext:
    """Represent fact build context within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by FactBuildContext.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.facts.FactBuildContext and related tests.
    """
    max_facts: int = 50
    max_string_length: int = 4000
    include_row_samples: bool = False
    include_paths: bool = False
    source_action: str | None = None
    source_args: dict[str, Any] | None = None
    output_mode: str = "text"


def build_sanitized_facts(
    result: Any,
    actions: list[Any],
    domain: str | None = None,
    context: Any | None = None,
) -> dict[str, Any]:
    """Build sanitized facts for the surrounding runtime workflow.

    Inputs:
        Receives result, actions, domain, context for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.facts.build_sanitized_facts.
    """
    ctx = _context(context)
    clean = strip_internal_telemetry(result)
    tools = _action_tools(actions)
    source_action = str(domain or ctx.source_action or (tools[-1] if tools else "") or "")
    inferred_domain = _infer_domain(clean, source_action, domain)

    if inferred_domain == "slurm":
        facts = _build_slurm_facts(clean, actions, ctx)
    elif inferred_domain == "sql":
        facts = _build_sql_facts(clean, actions, ctx)
    elif inferred_domain == "filesystem":
        facts = _build_filesystem_facts(clean, actions, ctx)
    else:
        facts = _build_generic_facts(clean, tools)

    facts.setdefault("domain", inferred_domain)
    facts["tools"] = tools
    compact = _cap_facts(_drop_empty(facts), max_items=ctx.max_facts, max_string_length=ctx.max_string_length)
    if not isinstance(compact, dict):
        return {"domain": inferred_domain, "redacted": True, "warnings": ["Facts were redacted because they were too large."]}
    compact.setdefault("redacted", True)
    compact.setdefault("warnings", [])
    return compact


def validate_facts_for_llm(
    facts: dict[str, Any],
    *,
    max_items: int = 50,
    max_string_length: int = 4000,
    max_depth: int = 6,
) -> bool:
    """Validate facts for llm for the surrounding runtime workflow.

    Inputs:
        Receives facts, max_items, max_string_length, max_depth for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.facts.validate_facts_for_llm.
    """
    try:
        _validate_node(facts, max_items=max_items, max_string_length=max_string_length, max_depth=max_depth, depth=0)
    except ValueError:
        return False
    return True


def _build_slurm_facts(result: Any, actions: list[Any], context: FactBuildContext) -> dict[str, Any]:
    """Handle the internal build slurm facts helper path for this module.

    Inputs:
        Receives result, actions, context for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.facts._build_slurm_facts.
    """
    normalized = normalize_slurm_result(result, _SourceContext(context.source_action, context.source_args or {}))
    facts: dict[str, Any] = {
        "domain": "slurm",
        "result_kind": normalized.kind.value,
        "title": normalized.title,
        "warnings": list(normalized.warnings),
    }
    _merge_normalized_slurm(facts, normalized)
    return facts


def _merge_normalized_slurm(facts: dict[str, Any], normalized: NormalizedSlurmResult) -> None:
    """Handle the internal merge normalized slurm helper path for this module.

    Inputs:
        Receives facts, normalized for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.facts._merge_normalized_slurm.
    """
    if normalized.kind == SlurmResultKind.COMPOUND:
        for child in normalized.grouped.get("children", []):
            if isinstance(child, NormalizedSlurmResult):
                _merge_normalized_slurm(facts, child)
        _merge_summary_facts(facts, normalized.summary)
        return

    summary = dict(normalized.summary or {})
    _merge_summary_facts(facts, summary)

    if normalized.kind in {SlurmResultKind.QUEUE_JOBS, SlurmResultKind.QUEUE_COUNTS, SlurmResultKind.QUEUE_GROUPED_COUNTS}:
        queue = dict(facts.get("queue") or {})
        if normalized.total_count is not None:
            state = str(summary.get("state") or "").upper()
            if state == "RUNNING":
                queue["running_jobs"] = normalized.total_count
            elif state == "PENDING":
                queue["pending_jobs"] = normalized.total_count
            else:
                queue["total_jobs"] = normalized.total_count
        queue.setdefault("truncated", normalized.truncated)
        facts["queue"] = _drop_empty(queue)

    if normalized.kind in {SlurmResultKind.ACCOUNTING_JOBS, SlurmResultKind.ACCOUNTING_COUNTS}:
        accounting = dict(facts.get("accounting") or {})
        filters = dict(summary.get("filters") or {})
        if str(filters.get("state") or "").upper() == "FAILED" and normalized.total_count is not None:
            accounting["failed_jobs_recent"] = normalized.total_count
        if filters.get("min_elapsed_seconds") and normalized.total_count is not None:
            accounting["duration_match_count"] = normalized.total_count
        accounting["truncated"] = normalized.truncated
        facts["accounting"] = _drop_empty(accounting)

    if normalized.kind == SlurmResultKind.ACCOUNTING_AGGREGATE:
        accounting = dict(facts.get("accounting") or {})
        accounting["runtime_metric"] = summary.get("metric")
        accounting["job_count"] = summary.get("job_count")
        accounting["average_elapsed_human"] = summary.get("average_elapsed_human")
        accounting["min_elapsed_human"] = summary.get("min_elapsed_human")
        accounting["max_elapsed_human"] = summary.get("max_elapsed_human")
        accounting["partition"] = summary.get("partition")
        accounting["state"] = summary.get("state")
        accounting["time_window"] = summary.get("time_window_label")
        accounting["truncated"] = normalized.truncated
        facts["accounting"] = _drop_empty(accounting)


def _merge_summary_facts(facts: dict[str, Any], summary: dict[str, Any]) -> None:
    """Handle the internal merge summary facts helper path for this module.

    Inputs:
        Receives facts, summary for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.facts._merge_summary_facts.
    """
    queue = dict(facts.get("queue") or {})
    for source, target in (
        ("queue_count", "total_jobs"),
        ("running_jobs", "running_jobs"),
        ("pending_jobs", "pending_jobs"),
    ):
        if summary.get(source) is not None:
            queue[target] = summary.get(source)
    if queue:
        facts["queue"] = _drop_empty(queue)

    nodes = dict(facts.get("nodes") or {})
    for source, target in (
        ("node_count", "total_nodes"),
        ("idle_nodes", "idle_nodes"),
        ("allocated_nodes", "allocated_nodes"),
        ("mixed_nodes", "mixed_nodes"),
        ("drained_nodes", "drained_nodes"),
        ("down_nodes", "down_nodes"),
        ("problematic_nodes", "problematic_nodes"),
        ("problematic_partition_rows", "affected_partition_rows"),
        ("unique_count", "total_nodes"),
        ("problematic_unique_count", "problematic_nodes"),
    ):
        if summary.get(source) is not None:
            nodes[target] = summary.get(source)
    if summary.get("node_states") is not None:
        nodes["states"] = summary.get("node_states")
    if nodes:
        facts["nodes"] = _drop_empty(nodes)

    gpu = dict(facts.get("gpu") or {})
    if summary.get("gpu_available") is not None:
        gpu["available"] = summary.get("gpu_available")
    if summary.get("total_gpus") is not None:
        gpu["total_gpus"] = summary.get("total_gpus")
    nested_gpu = dict(summary.get("gpu") or {})
    if nested_gpu:
        gpu["available"] = nested_gpu.get("available", gpu.get("available"))
        gpu["gpu_capable_nodes"] = nested_gpu.get("nodes_with_gpu", gpu.get("gpu_capable_nodes"))
        gpu["total_gpus"] = nested_gpu.get("total_gpus", gpu.get("total_gpus"))
    if gpu:
        facts["gpu"] = _drop_empty(gpu)

    accounting = dict(facts.get("accounting") or {})
    slurmdbd = dict(summary.get("slurmdbd") or {})
    if slurmdbd:
        accounting["slurmdbd_status"] = slurmdbd.get("status")
        accounting["available"] = slurmdbd.get("available")
    if summary.get("slurmdbd_status") is not None:
        accounting["slurmdbd_status"] = summary.get("slurmdbd_status")
    if accounting:
        facts["accounting"] = _drop_empty(accounting)

    if summary.get("partition_count") is not None:
        facts["partitions"] = {"partition_count": summary.get("partition_count")}


def _build_sql_facts(result: Any, actions: list[Any], context: FactBuildContext) -> dict[str, Any]:
    """Handle the internal build sql facts helper path for this module.

    Inputs:
        Receives result, actions, context for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.facts._build_sql_facts.
    """
    action = _last_action(actions, "sql.query")
    action_result = _action_attr(action, "result")
    payload = result if isinstance(result, dict) else {}
    if (not payload or "rows" not in payload) and isinstance(action_result, dict):
        payload = action_result
    rows = list(payload.get("rows") or []) if isinstance(payload, dict) else []
    scalar = _single_row_scalar(rows)
    args = dict(_action_attr(action, "args_summary") or {})
    facts: dict[str, Any] = {
        "domain": "sql",
        "database": payload.get("database") or _action_attr(action, "database"),
        "query_type": "count" if context.output_mode == "count" or scalar is not None else "rows",
        "row_count": payload.get("row_count", len(rows)),
        "result_count": scalar if isinstance(scalar, (int, float)) and not isinstance(scalar, bool) else None,
        "tables_used": _safe_list(payload.get("sql_tables_used") or payload.get("tables_used")),
        "columns_used": _safe_list(payload.get("sql_columns_used") or payload.get("columns_used")),
        "constraints_applied": _safe_list(payload.get("sql_constraints_covered") or payload.get("constraints_applied")),
        "projections_used": _safe_list(payload.get("sql_projections_resolved") or payload.get("projections_used")),
        "coverage_passed": payload.get("sql_constraint_coverage_passed") or payload.get("coverage_passed"),
        "truncated": payload.get("truncated"),
    }
    if scalar is not None:
        facts["result"] = {"count" if facts["query_type"] == "count" else "value": scalar}
    if args.get("database") and not facts.get("database"):
        facts["database"] = args.get("database")
    return facts


def _build_filesystem_facts(result: Any, actions: list[Any], context: FactBuildContext) -> dict[str, Any]:
    """Handle the internal build filesystem facts helper path for this module.

    Inputs:
        Receives result, actions, context for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.facts._build_filesystem_facts.
    """
    payload = result if isinstance(result, dict) else {}
    action = _last_visible_action(actions)
    args = dict(_action_attr(action, "args_summary") or {})
    path_value = payload.get("path") or args.get("path")
    facts = {
        "domain": "filesystem",
        "operation": _action_attr(action, "tool") or "filesystem",
        "path_scope": path_value if context.include_paths else ("requested path" if path_value else None),
        "pattern": payload.get("pattern") or args.get("pattern"),
        "file_count": payload.get("file_count") or payload.get("count"),
        "total_size_bytes": payload.get("total_size_bytes"),
        "total_size_human": payload.get("display_size") or payload.get("summary_text"),
        "recursive": payload.get("recursive") if payload.get("recursive") is not None else args.get("recursive"),
        "truncated": payload.get("truncated"),
    }
    return facts


def _build_generic_facts(result: Any, tools: list[str]) -> dict[str, Any]:
    """Handle the internal build generic facts helper path for this module.

    Inputs:
        Receives result, tools for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.facts._build_generic_facts.
    """
    clean = _sanitize_node(result, max_items=10, max_string_length=500, depth=0)
    facts: dict[str, Any] = {"domain": "generic", "type": type(result).__name__}
    if isinstance(clean, dict):
        facts["keys"] = list(clean.keys())[:10]
    elif isinstance(clean, list):
        facts["item_count"] = len(clean)
    elif isinstance(clean, (str, int, float, bool)) or clean is None:
        facts["value"] = clean
    facts["tools"] = tools
    return facts


def _infer_domain(result: Any, source_action: str, explicit: str | None) -> str:
    """Handle the internal infer domain helper path for this module.

    Inputs:
        Receives result, source_action, explicit for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.facts._infer_domain.
    """
    if explicit:
        return explicit
    if source_action.startswith("slurm."):
        return "slurm"
    if source_action == "sql.query":
        return "sql"
    if source_action.startswith("fs."):
        return "filesystem"
    if isinstance(result, dict):
        if "database" in result and "rows" in result:
            return "sql"
        if "results" in result or any(key in result for key in ("jobs", "nodes", "partitions", "metric_group")):
            return "slurm"
        if any(key in result for key in ("file_count", "total_size_bytes", "matches", "entries")):
            return "filesystem"
    return "generic"


def _context(context: Any | None) -> FactBuildContext:
    """Handle the internal context helper path for this module.

    Inputs:
        Receives context for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.facts._context.
    """
    return FactBuildContext(
        max_facts=int(getattr(context, "max_facts", getattr(context, "max_items", 50)) or 50),
        max_string_length=int(getattr(context, "max_input_chars", 4000) or 4000),
        include_row_samples=bool(getattr(context, "include_row_samples", False)),
        include_paths=bool(getattr(context, "include_paths", False)),
        source_action=getattr(context, "source_action", None),
        source_args=dict(getattr(context, "source_args", {}) or {}),
        output_mode=str(getattr(context, "output_mode", "text") or "text"),
    )


class _SourceContext:
    """Represent source context within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by _SourceContext.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.facts._SourceContext and related tests.
    """
    def __init__(self, source_action: str | None, source_args: dict[str, Any]) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives source_action, source_args for this _SourceContext method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by planning, execution, validation, and presentation through _SourceContext.__init__ calls and related tests.
        """
        self.source_action = source_action
        self.source_args = source_args


def _action_tools(actions: list[Any]) -> list[str]:
    """Handle the internal action tools helper path for this module.

    Inputs:
        Receives actions for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.facts._action_tools.
    """
    tools: list[str] = []
    for action in actions:
        tool = str(_action_attr(action, "tool") or "")
        if tool and tool != "runtime.return" and tool not in tools:
            tools.append(tool)
    return tools


def _last_visible_action(actions: list[Any]) -> Any | None:
    """Handle the internal last visible action helper path for this module.

    Inputs:
        Receives actions for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.facts._last_visible_action.
    """
    for action in reversed(actions):
        if str(_action_attr(action, "tool") or "") != "runtime.return":
            return action
    return None


def _last_action(actions: list[Any], tool: str) -> Any | None:
    """Handle the internal last action helper path for this module.

    Inputs:
        Receives actions, tool for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.facts._last_action.
    """
    for action in reversed(actions):
        if str(_action_attr(action, "tool") or "") == tool:
            return action
    return None


def _action_attr(action: Any, key: str) -> Any:
    """Handle the internal action attr helper path for this module.

    Inputs:
        Receives action, key for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.facts._action_attr.
    """
    if action is None:
        return None
    if isinstance(action, dict):
        return action.get(key)
    return getattr(action, key, None)


def _single_row_scalar(rows: list[Any]) -> Any | None:
    """Handle the internal single row scalar helper path for this module.

    Inputs:
        Receives rows for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.facts._single_row_scalar.
    """
    if len(rows) != 1 or not isinstance(rows[0], dict) or len(rows[0]) != 1:
        return None
    return next(iter(rows[0].values()))


def _safe_list(value: Any) -> list[Any]:
    """Handle the internal safe list helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.facts._safe_list.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, (str, int, float, bool))]
    if isinstance(value, (str, int, float, bool)):
        return [value]
    return []


def _drop_empty(value: dict[str, Any]) -> dict[str, Any]:
    """Handle the internal drop empty helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.facts._drop_empty.
    """
    return {key: item for key, item in value.items() if item not in (None, {}, [])}


def _cap_facts(value: Any, *, max_items: int, max_string_length: int) -> Any:
    """Handle the internal cap facts helper path for this module.

    Inputs:
        Receives value, max_items, max_string_length for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.facts._cap_facts.
    """
    clean = _sanitize_node(value, max_items=max_items, max_string_length=max_string_length, depth=0)
    encoded = json.dumps(clean, ensure_ascii=False, sort_keys=True, default=str)
    if len(encoded) > max_string_length:
        return {"domain": value.get("domain", "generic") if isinstance(value, dict) else "generic", "redacted": True, "warnings": ["Facts exceeded size limit."]}
    return clean


def _sanitize_node(value: Any, *, max_items: int, max_string_length: int, depth: int) -> Any:
    """Handle the internal sanitize node helper path for this module.

    Inputs:
        Receives value, max_items, max_string_length, depth for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.facts._sanitize_node.
    """
    if depth > 6:
        return "<redacted>"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:max_items]:
            key_text = str(key)
            lowered = key_text.lower()
            if lowered in FORBIDDEN_EXACT_KEYS or any(part in lowered for part in FORBIDDEN_KEY_PARTS):
                continue
            result[key_text] = _sanitize_node(item, max_items=max_items, max_string_length=max_string_length, depth=depth + 1)
        return _drop_empty(result)
    if isinstance(value, list):
        return [_sanitize_node(item, max_items=max_items, max_string_length=max_string_length, depth=depth + 1) for item in value[:max_items]]
    if isinstance(value, str):
        if len(value) > max_string_length:
            return value[:max_string_length] + "...<truncated>"
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:max_string_length]


def _validate_node(value: Any, *, max_items: int, max_string_length: int, max_depth: int, depth: int) -> None:
    """Handle the internal validate node helper path for this module.

    Inputs:
        Receives value, max_items, max_string_length, max_depth, depth for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.facts._validate_node.
    """
    if depth > max_depth:
        raise ValueError("facts exceed max depth")
    if isinstance(value, dict):
        if len(value) > max_items:
            raise ValueError("facts contain too many keys")
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_EXACT_KEYS or any(part in lowered for part in FORBIDDEN_KEY_PARTS):
                raise ValueError(f"unsafe fact key: {key}")
            _validate_node(item, max_items=max_items, max_string_length=max_string_length, max_depth=max_depth, depth=depth + 1)
        return
    if isinstance(value, list):
        if len(value) > max_items:
            raise ValueError("facts contain too many list items")
        for item in value:
            _validate_node(item, max_items=max_items, max_string_length=max_string_length, max_depth=max_depth, depth=depth + 1)
        return
    if isinstance(value, str) and len(value) > max_string_length:
        raise ValueError("fact string too large")
