"""OpenFABRIC Runtime Module: aor_runtime.runtime.slurm_result_normalizer

Purpose:
    Normalize SLURM tool payloads into presentation-ready result types.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aor_runtime.tools.slurm import dedupe_slurm_nodes_by_name, is_problematic_node_state, summarize_node_states


class SlurmResultKind(str, Enum):
    """Represent slurm result kind within the OpenFABRIC runtime. It extends str, Enum.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SlurmResultKind.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.slurm_result_normalizer.SlurmResultKind and related tests.
    """
    CLUSTER_STATUS = "cluster_status"
    QUEUE_JOBS = "queue_jobs"
    QUEUE_COUNTS = "queue_counts"
    QUEUE_GROUPED_COUNTS = "queue_grouped_counts"
    ACCOUNTING_JOBS = "accounting_jobs"
    ACCOUNTING_COUNTS = "accounting_counts"
    ACCOUNTING_AGGREGATE = "accounting_aggregate"
    NODE_ROWS = "node_rows"
    NODE_SUMMARY = "node_summary"
    PROBLEMATIC_NODES = "problematic_nodes"
    PARTITION_SUMMARY = "partition_summary"
    GPU_SUMMARY = "gpu_summary"
    SLURMDBD_HEALTH = "slurmdbd_health"
    COMPOUND = "compound"
    UNSUPPORTED_MUTATION = "unsupported_mutation"
    UNKNOWN = "unknown"


@dataclass
class NormalizedSlurmResult:
    """Represent normalized slurm result within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by NormalizedSlurmResult.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.slurm_result_normalizer.NormalizedSlurmResult and related tests.
    """
    kind: SlurmResultKind
    title: str
    summary: dict[str, Any] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    grouped: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    total_count: int | None = None
    returned_count: int | None = None
    truncated: bool = False
    commands_used: list[str] = field(default_factory=list)
    raw_debug: dict[str, Any] | None = None


def normalize_slurm_result(result: Any, context: Any | None = None) -> NormalizedSlurmResult:
    """Normalize slurm result for the surrounding runtime workflow.

    Inputs:
        Receives result, context for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer.normalize_slurm_result.
    """
    source_action = str(getattr(context, "source_action", "") or "")
    source_args = dict(getattr(context, "source_args", {}) or {})
    payload = _coerce_payload(result, source_action)

    if isinstance(payload, str) and _looks_like_mutation_refusal(payload):
        return NormalizedSlurmResult(
            kind=SlurmResultKind.UNSUPPORTED_MUTATION,
            title="Request Not Allowed",
            summary={"message": payload},
            warnings=["SLURM mutation/admin operations are not supported."],
            raw_debug={"value": payload},
        )

    if not isinstance(payload, dict):
        return NormalizedSlurmResult(
            kind=SlurmResultKind.UNKNOWN,
            title="SLURM Result",
            summary={"value": payload},
            raw_debug={"value": payload},
        )

    if isinstance(payload.get("results"), dict):
        return _normalize_compound(payload, context)

    metric_group = str(payload.get("metric_group") or "")
    metric_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else payload
    if metric_group == "cluster_summary":
        return _normalize_cluster_status(metric_payload, payload)
    if metric_group in {"queue_summary", "scheduler_health"}:
        return _normalize_queue_summary(metric_payload, payload)
    if metric_group == "node_summary":
        return _normalize_node_summary(metric_payload, payload)
    if metric_group == "problematic_nodes":
        nested = dict(metric_payload)
        nested.setdefault("filters", {"state_group": "problematic"})
        return _normalize_nodes(nested, payload)
    if metric_group == "partition_summary":
        return _normalize_partitions(metric_payload, payload)
    if metric_group == "gpu_summary":
        return _normalize_gpu(metric_payload, payload)
    if metric_group in {"slurmdbd_health", "accounting_health"}:
        return _normalize_slurmdbd(metric_payload, payload)

    if _looks_like_multi_metric_accounting_aggregate(payload):
        return _normalize_multi_metric_accounting_aggregate(payload)
    if source_action == "slurm.accounting_aggregate" or payload.get("result_kind") == "accounting_aggregate":
        return _normalize_accounting_aggregate(payload)
    if "jobs" in payload and source_action == "slurm.accounting":
        return _normalize_accounting_jobs(payload)
    if "jobs" in payload:
        return _normalize_queue_jobs(payload, source_args)
    if "nodes" in payload:
        return _normalize_nodes(payload, payload)
    if "partitions" in payload:
        return _normalize_partitions(payload, payload)
    if {"available", "status"} <= set(payload):
        return _normalize_slurmdbd(payload, payload)
    if {"available", "nodes_with_gpu"} & set(payload):
        return _normalize_gpu(payload, payload)
    if source_action.startswith("slurm."):
        return NormalizedSlurmResult(
            kind=SlurmResultKind.UNKNOWN,
            title="SLURM Result",
            summary=_small_summary(payload),
            raw_debug=payload,
        )
    return NormalizedSlurmResult(kind=SlurmResultKind.UNKNOWN, title="SLURM Result", summary={}, raw_debug=payload)


def _normalize_compound(payload: dict[str, Any], context: Any | None) -> NormalizedSlurmResult:
    """Handle the internal normalize compound helper path for this module.

    Inputs:
        Receives payload, context for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._normalize_compound.
    """
    children: list[NormalizedSlurmResult] = []
    summary: dict[str, Any] = {}
    warnings: list[str] = []
    for key, value in dict(payload.get("results") or {}).items():
        child_context = _ChildContext(source_action=_source_for_compound_key(str(key)), source_args={})
        child = normalize_slurm_result(value, child_context)
        children.append(child)
        warnings.extend(child.warnings)
        _merge_child_summary(summary, child)
    title = "SLURM Cluster Status" if any(child.kind == SlurmResultKind.CLUSTER_STATUS for child in children) else "SLURM Summary"
    return NormalizedSlurmResult(
        kind=SlurmResultKind.COMPOUND,
        title=title,
        summary=summary,
        grouped={"children": children},
        warnings=_dedupe(warnings),
        raw_debug=payload,
    )


def _normalize_cluster_status(payload: dict[str, Any], raw: dict[str, Any]) -> NormalizedSlurmResult:
    """Handle the internal normalize cluster status helper path for this module.

    Inputs:
        Receives payload, raw for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._normalize_cluster_status.
    """
    warnings = []
    problematic = _to_int(payload.get("problematic_nodes"))
    if problematic and problematic > 0:
        warnings.append("Some nodes are unavailable or drained. Check node details before scheduling large jobs.")
    summary = {
        "queue_count": payload.get("queue_count"),
        "running_jobs": payload.get("running_jobs"),
        "pending_jobs": payload.get("pending_jobs"),
        "node_count": payload.get("node_count"),
        "partition_node_rows": payload.get("partition_node_rows"),
        "idle_nodes": payload.get("idle_nodes"),
        "allocated_nodes": payload.get("allocated_nodes"),
        "mixed_nodes": payload.get("mixed_nodes"),
        "down_nodes": payload.get("down_nodes"),
        "drained_nodes": payload.get("drained_nodes"),
        "problematic_nodes": payload.get("problematic_nodes"),
        "problematic_partition_rows": payload.get("problematic_partition_rows"),
        "gpu_available": payload.get("gpu_available"),
        "total_gpus": payload.get("total_gpus"),
    }
    return NormalizedSlurmResult(
        kind=SlurmResultKind.CLUSTER_STATUS,
        title="SLURM Cluster Status",
        summary=_drop_none(summary),
        warnings=warnings,
        raw_debug=raw,
    )


def _normalize_queue_summary(payload: dict[str, Any], raw: dict[str, Any]) -> NormalizedSlurmResult:
    """Handle the internal normalize queue summary helper path for this module.

    Inputs:
        Receives payload, raw for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._normalize_queue_summary.
    """
    grouped = {
        "by_state": payload.get("by_state") or {},
        "by_user": payload.get("by_user") or {},
        "by_partition": payload.get("by_partition") or {},
    }
    summary = {
        "queue_count": payload.get("job_count"),
        "running_jobs": payload.get("running_jobs"),
        "pending_jobs": payload.get("pending_jobs"),
    }
    return NormalizedSlurmResult(
        kind=SlurmResultKind.QUEUE_COUNTS,
        title="SLURM Queue Summary",
        summary=_drop_none(summary),
        grouped=grouped,
        total_count=_to_int(payload.get("job_count")),
        raw_debug=raw,
    )


def _normalize_queue_jobs(payload: dict[str, Any], source_args: dict[str, Any]) -> NormalizedSlurmResult:
    """Handle the internal normalize queue jobs helper path for this module.

    Inputs:
        Receives payload, source_args for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._normalize_queue_jobs.
    """
    jobs = _dict_rows(payload.get("jobs"))
    filters = dict(payload.get("filters") or {})
    filters.update({key: value for key, value in source_args.items() if value not in (None, "") and key not in filters})
    state = str(filters.get("state") or "").upper()
    group_by = str(payload.get("group_by") or filters.get("group_by") or "").strip()
    total = _to_int(payload.get("total_count", payload.get("count", len(jobs))))
    returned = _to_int(payload.get("returned_count", len(jobs))) or len(jobs)
    truncated = bool(payload.get("truncated") or (total is not None and returned < total))
    if group_by:
        grouped = dict(payload.get("grouped") or _count_by(jobs, group_by))
        return NormalizedSlurmResult(
            kind=SlurmResultKind.QUEUE_GROUPED_COUNTS,
            title=_queue_grouped_title(state, group_by),
            summary={"state": state or None, "group_by": group_by},
            grouped=grouped,
            total_count=total,
            returned_count=returned,
            truncated=truncated,
            warnings=_truncation_warnings("jobs", total, returned, truncated),
            raw_debug=payload,
        )
    return NormalizedSlurmResult(
        kind=SlurmResultKind.QUEUE_JOBS,
        title=_queue_title(state),
        summary={"state": state or None, "filters": filters},
        rows=jobs,
        total_count=total,
        returned_count=returned,
        truncated=truncated,
        warnings=_truncation_warnings("jobs", total, returned, truncated),
        raw_debug=payload,
    )


def _normalize_accounting_jobs(payload: dict[str, Any]) -> NormalizedSlurmResult:
    """Handle the internal normalize accounting jobs helper path for this module.

    Inputs:
        Receives payload for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._normalize_accounting_jobs.
    """
    jobs = _dict_rows(payload.get("jobs"))
    filters = dict(payload.get("filters") or {})
    total = _to_int(payload.get("total_count", payload.get("count", len(jobs))))
    returned = _to_int(payload.get("returned_count", len(jobs))) or len(jobs)
    truncated = bool(payload.get("truncated") or (total is not None and returned < total))
    group_by = str(payload.get("group_by") or "").strip()
    title = _accounting_title(filters)
    if group_by:
        return NormalizedSlurmResult(
            kind=SlurmResultKind.ACCOUNTING_COUNTS,
            title=title,
            summary={"filters": filters, "group_by": group_by},
            grouped=dict(payload.get("grouped") or _count_by(jobs, group_by)),
            total_count=total,
            returned_count=returned,
            truncated=truncated,
            warnings=_truncation_warnings("accounting jobs", total, returned, truncated),
            raw_debug=payload,
        )
    warnings = _truncation_warnings("accounting jobs", total, returned, truncated)
    if total == 0:
        warnings.append("No matching accounting jobs were found.")
    return NormalizedSlurmResult(
        kind=SlurmResultKind.ACCOUNTING_JOBS,
        title=title,
        summary={"filters": filters},
        rows=jobs,
        total_count=total,
        returned_count=returned,
        truncated=truncated,
        warnings=warnings,
        raw_debug=payload,
    )


def _normalize_accounting_aggregate(payload: dict[str, Any]) -> NormalizedSlurmResult:
    """Handle the internal normalize accounting aggregate helper path for this module.

    Inputs:
        Receives payload for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._normalize_accounting_aggregate.
    """
    warnings = [str(warning) for warning in payload.get("warnings") or [] if str(warning)]
    metric = str(payload.get("metric") or "average_elapsed")
    partition = payload.get("partition")
    title = "SLURM Job Runtime"
    if metric == "count_longer_than":
        title = "SLURM Jobs Longer Than Requested Duration"
    elif metric == "max_elapsed":
        title = "Maximum SLURM Job Runtime"
    elif metric == "min_elapsed":
        title = "Minimum SLURM Job Runtime"
    elif metric == "sum_elapsed":
        title = "Total SLURM Job Runtime"
    summary = {
        "metric": metric,
        "source": payload.get("source"),
        "partition": partition,
        "user": payload.get("user"),
        "state": payload.get("state"),
        "include_all_states": payload.get("include_all_states"),
        "excluded_states": payload.get("excluded_states"),
        "default_state_applied": payload.get("default_state_applied"),
        "start": payload.get("start"),
        "end": payload.get("end"),
        "time_window_label": payload.get("time_window_label"),
        "group_by": payload.get("group_by"),
        "threshold_seconds": payload.get("threshold_seconds"),
        "threshold_human": payload.get("threshold_human"),
        "job_count": payload.get("job_count"),
        "average_elapsed_seconds": payload.get("average_elapsed_seconds"),
        "average_elapsed_human": payload.get("average_elapsed_human"),
        "min_elapsed_seconds": payload.get("min_elapsed_seconds"),
        "min_elapsed_human": payload.get("min_elapsed_human"),
        "max_elapsed_seconds": payload.get("max_elapsed_seconds"),
        "max_elapsed_human": payload.get("max_elapsed_human"),
        "sum_elapsed_seconds": payload.get("sum_elapsed_seconds"),
        "sum_elapsed_human": payload.get("sum_elapsed_human"),
        "value_seconds": payload.get("value_seconds"),
        "value_human": payload.get("value_human"),
        "truncated": payload.get("truncated"),
    }
    groups = list(payload.get("groups") or [])
    return NormalizedSlurmResult(
        kind=SlurmResultKind.ACCOUNTING_AGGREGATE,
        title=title,
        summary=_drop_none(summary),
        rows=_dict_rows(groups),
        total_count=_to_int(payload.get("job_count")),
        returned_count=_to_int(payload.get("returned_count")),
        truncated=bool(payload.get("truncated")),
        warnings=warnings,
        raw_debug=payload,
    )


def _looks_like_multi_metric_accounting_aggregate(payload: dict[str, Any]) -> bool:
    """Handle the internal looks like multi metric accounting aggregate helper path for this module.

    Inputs:
        Receives payload for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._looks_like_multi_metric_accounting_aggregate.
    """
    if not payload or "result_kind" in payload:
        return False
    metric_children = [
        value
        for value in payload.values()
        if isinstance(value, dict)
        and (
            value.get("result_kind") == "accounting_aggregate"
            or {"average_elapsed_seconds", "min_elapsed_seconds", "max_elapsed_seconds"} & set(value)
        )
    ]
    return len(metric_children) >= 2 and len(metric_children) == len(payload)


def _normalize_multi_metric_accounting_aggregate(payload: dict[str, Any]) -> NormalizedSlurmResult:
    """Handle the internal normalize multi metric accounting aggregate helper path for this module.

    Inputs:
        Receives payload for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._normalize_multi_metric_accounting_aggregate.
    """
    metric_order = {
        "min": 0,
        "min_elapsed": 0,
        "minimum": 0,
        "max": 1,
        "max_elapsed": 1,
        "maximum": 1,
        "avg": 2,
        "average": 2,
        "average_elapsed": 2,
        "sum": 3,
        "sum_elapsed": 3,
        "total": 3,
        "count": 4,
    }
    rows: list[dict[str, Any]] = []
    children = [(str(key), dict(value)) for key, value in payload.items() if isinstance(value, dict)]
    children.sort(key=lambda item: (metric_order.get(str(item[0]).lower(), 99), str(item[0]).lower()))
    first = children[0][1] if children else {}
    warnings: list[str] = []
    for key, child in children:
        warnings.extend(str(warning) for warning in child.get("warnings") or [] if str(warning))
        metric = str(child.get("metric") or key)
        rows.append(
            _drop_none(
                {
                    "metric": _metric_row_label(key, metric),
                    "value": child.get("value_human") or _metric_value_for_child(metric, child),
                    "jobs": child.get("job_count") or child.get("total_count") or child.get("returned_count"),
                    "average": child.get("average_elapsed_human"),
                    "minimum": child.get("min_elapsed_human"),
                    "maximum": child.get("max_elapsed_human"),
                    "total": child.get("sum_elapsed_human"),
                    "partition": child.get("partition"),
                    "state": child.get("state"),
                }
            )
        )
    summary = _drop_none(
        {
            "partition": first.get("partition"),
            "user": first.get("user"),
            "state": first.get("state"),
            "job_count": first.get("job_count") or first.get("total_count") or first.get("returned_count"),
            "source": first.get("source"),
            "start": first.get("start"),
            "end": first.get("end"),
            "time_window_label": first.get("time_window_label"),
            "filters": first.get("filters"),
        }
    )
    return NormalizedSlurmResult(
        kind=SlurmResultKind.ACCOUNTING_AGGREGATE,
        title="SLURM Job Runtime Summary",
        summary=summary,
        grouped={"metrics": rows},
        total_count=_to_int(summary.get("job_count")),
        returned_count=_to_int(summary.get("job_count")),
        truncated=any(bool(child.get("truncated")) for _, child in children),
        warnings=_dedupe(warnings),
        raw_debug=payload,
    )


def _metric_row_label(key: str, metric: str) -> str:
    """Handle the internal metric row label helper path for this module.

    Inputs:
        Receives key, metric for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._metric_row_label.
    """
    normalized = str(metric or key).lower()
    if normalized == "average_elapsed" or key.lower() in {"avg", "average"}:
        return "Average runtime"
    if normalized == "min_elapsed" or key.lower() in {"min", "minimum"}:
        return "Minimum runtime"
    if normalized == "max_elapsed" or key.lower() in {"max", "maximum"}:
        return "Maximum runtime"
    if normalized == "sum_elapsed" or key.lower() in {"sum", "total"}:
        return "Total runtime"
    if normalized == "count":
        return "Matching jobs"
    return str(key or metric).replace("_", " ").title()


def _metric_value_for_child(metric: str, child: dict[str, Any]) -> Any:
    """Handle the internal metric value for child helper path for this module.

    Inputs:
        Receives metric, child for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._metric_value_for_child.
    """
    normalized = str(metric or "").lower()
    if normalized == "average_elapsed":
        return child.get("average_elapsed_human")
    if normalized == "min_elapsed":
        return child.get("min_elapsed_human")
    if normalized == "max_elapsed":
        return child.get("max_elapsed_human")
    if normalized == "sum_elapsed":
        return child.get("sum_elapsed_human")
    if normalized == "count":
        return child.get("count") or child.get("job_count")
    return child.get("value_human") or child.get("value_seconds")


def _normalize_nodes(payload: dict[str, Any], raw: dict[str, Any]) -> NormalizedSlurmResult:
    """Handle the internal normalize nodes helper path for this module.

    Inputs:
        Receives payload, raw for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._normalize_nodes.
    """
    rows = _dict_rows(payload.get("nodes"))
    filters = dict(payload.get("filters") or {})
    state_group = str(filters.get("state_group") or "").lower()
    unique_rows = dedupe_slurm_nodes_by_name(rows)
    problematic_rows = [row for row in rows if is_problematic_node_state(str(row.get("state", "")))]
    problematic_unique = dedupe_slurm_nodes_by_name(problematic_rows)
    is_problematic = state_group == "problematic" or bool(problematic_rows) and len(problematic_rows) == len(rows)
    warnings = []
    if rows and len(unique_rows) < len(rows):
        warnings.append(f"Node rows include {len(rows)} partition rows for {len(unique_rows)} unique nodes.")
    if problematic_unique:
        warnings.append(f"Found {len(problematic_unique)} unique problematic nodes affecting {len(problematic_rows)} partition rows.")
    summary = {
        "unique_count": len(unique_rows),
        "partition_row_count": len(rows),
        "problematic_unique_count": len(problematic_unique),
        "problematic_partition_row_count": len(problematic_rows),
        "by_state": summarize_node_states(rows),
    }
    return NormalizedSlurmResult(
        kind=SlurmResultKind.PROBLEMATIC_NODES if is_problematic else SlurmResultKind.NODE_ROWS,
        title="Problematic Nodes" if is_problematic else "SLURM Nodes",
        summary=summary,
        rows=problematic_unique if is_problematic else unique_rows,
        total_count=len(unique_rows),
        returned_count=len(unique_rows),
        warnings=warnings,
        raw_debug=raw,
    )


def _normalize_node_summary(payload: dict[str, Any], raw: dict[str, Any]) -> NormalizedSlurmResult:
    """Handle the internal normalize node summary helper path for this module.

    Inputs:
        Receives payload, raw for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._normalize_node_summary.
    """
    return NormalizedSlurmResult(
        kind=SlurmResultKind.NODE_SUMMARY,
        title="SLURM Node Summary",
        summary={
            "node_count": payload.get("node_count"),
            "partition_node_rows": payload.get("partition_node_rows"),
            "by_state": payload.get("by_state") or {},
            "by_partition": payload.get("by_partition") or {},
        },
        raw_debug=raw,
    )


def _normalize_partitions(payload: dict[str, Any], raw: dict[str, Any]) -> NormalizedSlurmResult:
    """Handle the internal normalize partitions helper path for this module.

    Inputs:
        Receives payload, raw for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._normalize_partitions.
    """
    rows = _dict_rows(payload.get("partitions"))
    grouped: dict[str, Any] = {}
    for row in rows:
        partition = str(row.get("partition") or "unknown")
        item = grouped.setdefault(partition, {"rows": 0, "states": Counter(), "nodes": 0, "cpu_summaries": [], "gres": Counter()})
        item["rows"] += 1
        item["states"][str(row.get("state") or row.get("availability") or "unknown")] += 1
        try:
            item["nodes"] += int(row.get("nodes") or 0)
        except Exception:
            pass
        if row.get("cpus"):
            item["cpu_summaries"].append(str(row.get("cpus")))
        if row.get("gres"):
            item["gres"][str(row.get("gres"))] += 1
    normalized_grouped = {
        key: {
            "rows": value["rows"],
            "states": dict(value["states"]),
            "nodes": value["nodes"],
            "cpu_summaries": _dedupe(value["cpu_summaries"]),
            "gres": dict(value["gres"]),
        }
        for key, value in grouped.items()
    }
    return NormalizedSlurmResult(
        kind=SlurmResultKind.PARTITION_SUMMARY,
        title="SLURM Partition Summary",
        rows=rows,
        grouped=normalized_grouped,
        total_count=len(rows),
        returned_count=len(rows),
        raw_debug=raw,
    )


def _normalize_gpu(payload: dict[str, Any], raw: dict[str, Any]) -> NormalizedSlurmResult:
    """Handle the internal normalize gpu helper path for this module.

    Inputs:
        Receives payload, raw for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._normalize_gpu.
    """
    return NormalizedSlurmResult(
        kind=SlurmResultKind.GPU_SUMMARY,
        title="SLURM GPU Summary",
        summary={
            "available": payload.get("available"),
            "nodes_with_gpu": payload.get("nodes_with_gpu"),
            "total_gpus": payload.get("total_gpus"),
            "gpu_memory_gres": payload.get("gpu_memory_gres"),
            "by_gres": payload.get("by_gres"),
        },
        raw_debug=raw,
    )


def _normalize_slurmdbd(payload: dict[str, Any], raw: dict[str, Any]) -> NormalizedSlurmResult:
    """Handle the internal normalize slurmdbd helper path for this module.

    Inputs:
        Receives payload, raw for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._normalize_slurmdbd.
    """
    available = payload.get("available")
    status = str(payload.get("status") or "unknown")
    warnings = [] if bool(available) else ["SLURM accounting appears unavailable."]
    return NormalizedSlurmResult(
        kind=SlurmResultKind.SLURMDBD_HEALTH,
        title="SLURM Accounting Health",
        summary={"available": available, "status": status, "message": payload.get("message")},
        warnings=warnings,
        raw_debug=raw,
    )


def _coerce_payload(result: Any, source_action: str) -> Any:
    """Handle the internal coerce payload helper path for this module.

    Inputs:
        Receives result, source_action for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._coerce_payload.
    """
    if isinstance(result, dict):
        if source_action == "slurm.queue" and "value" in result and isinstance(result["value"], list):
            return {"jobs": result["value"]}
        if source_action == "slurm.nodes" and "value" in result and isinstance(result["value"], list):
            return {"nodes": result["value"]}
        if source_action == "slurm.partitions" and "value" in result and isinstance(result["value"], list):
            return {"partitions": result["value"]}
        return result
    if isinstance(result, list):
        if source_action == "slurm.queue":
            return {"jobs": result}
        if source_action == "slurm.nodes":
            return {"nodes": result}
        if source_action == "slurm.partitions":
            return {"partitions": result}
        if source_action == "slurm.accounting":
            return {"jobs": result}
    return result


def _merge_child_summary(summary: dict[str, Any], child: NormalizedSlurmResult) -> None:
    """Handle the internal merge child summary helper path for this module.

    Inputs:
        Receives summary, child for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._merge_child_summary.
    """
    data = child.summary or {}
    if child.kind == SlurmResultKind.CLUSTER_STATUS:
        summary.update(data)
    elif child.kind == SlurmResultKind.QUEUE_JOBS:
        state = str(data.get("state") or "").upper()
        if state == "RUNNING":
            summary["running_jobs"] = child.total_count
        elif state == "PENDING":
            summary["pending_jobs"] = child.total_count
        else:
            summary["queue_count"] = child.total_count
    elif child.kind == SlurmResultKind.QUEUE_COUNTS:
        summary.update(data)
    elif child.kind == SlurmResultKind.PROBLEMATIC_NODES:
        summary["problematic_nodes"] = data.get("problematic_unique_count")
        summary["problematic_partition_rows"] = data.get("problematic_partition_row_count")
    elif child.kind == SlurmResultKind.GPU_SUMMARY:
        summary["gpu_available"] = data.get("available")
        summary["total_gpus"] = data.get("total_gpus")
    elif child.kind == SlurmResultKind.SLURMDBD_HEALTH:
        summary["slurmdbd_status"] = data.get("status")
    elif child.kind == SlurmResultKind.ACCOUNTING_AGGREGATE:
        summary["accounting_runtime"] = data


def _source_for_compound_key(key: str) -> str:
    """Handle the internal source for compound key helper path for this module.

    Inputs:
        Receives key for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._source_for_compound_key.
    """
    lowered = key.lower()
    if "runtime" in lowered or "aggregate" in lowered:
        return "slurm.accounting_aggregate"
    if "job" in lowered or "queue" in lowered:
        return "slurm.queue" if "accounting" not in lowered else "slurm.accounting"
    if "node" in lowered:
        return "slurm.nodes"
    if "partition" in lowered:
        return "slurm.partitions"
    if "slurmdbd" in lowered or "health" in lowered and "account" in lowered:
        return "slurm.slurmdbd_health"
    return "slurm.metrics"


def _queue_title(state: str) -> str:
    """Handle the internal queue title helper path for this module.

    Inputs:
        Receives state for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._queue_title.
    """
    if state == "RUNNING":
        return "Running SLURM Jobs"
    if state == "PENDING":
        return "Pending SLURM Jobs"
    return "SLURM Queue"


def _queue_grouped_title(state: str, group_by: str) -> str:
    """Handle the internal queue grouped title helper path for this module.

    Inputs:
        Receives state, group_by for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._queue_grouped_title.
    """
    prefix = "Running Jobs" if state == "RUNNING" else "Pending Jobs" if state == "PENDING" else "Jobs"
    return f"{prefix} by {group_by.replace('_', ' ').title()}"


def _accounting_title(filters: dict[str, Any]) -> str:
    """Handle the internal accounting title helper path for this module.

    Inputs:
        Receives filters for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._accounting_title.
    """
    state = str(filters.get("state") or "").upper()
    if state == "FAILED" and filters.get("start"):
        return "Failed Jobs Since Yesterday" if "00:00:00" in str(filters.get("start")) else "Failed SLURM Accounting Jobs"
    if filters.get("min_elapsed_seconds"):
        return "SLURM Jobs Longer Than Requested Duration"
    if state:
        return f"{state.title()} SLURM Accounting Jobs"
    return "SLURM Accounting Jobs"


def _truncation_warnings(label: str, total: int | None, returned: int | None, truncated: bool) -> list[str]:
    """Handle the internal truncation warnings helper path for this module.

    Inputs:
        Receives label, total, returned, truncated for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._truncation_warnings.
    """
    if truncated and total is not None and returned is not None:
        return [f"Showing {returned} of {total} {label}."]
    return []


def _count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    """Handle the internal count by helper path for this module.

    Inputs:
        Receives rows, field for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._count_by.
    """
    counter: Counter[str] = Counter()
    for row in rows:
        value = str(row.get(field) or "unknown")
        counter[value] += 1
    return dict(sorted(counter.items()))


def _dict_rows(value: Any) -> list[dict[str, Any]]:
    """Handle the internal dict rows helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._dict_rows.
    """
    return [dict(row) for row in value or [] if isinstance(row, dict)]


def _to_int(value: Any) -> int | None:
    """Handle the internal to int helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._to_int.
    """
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _small_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Handle the internal small summary helper path for this module.

    Inputs:
        Receives payload for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._small_summary.
    """
    return {str(key): value for key, value in payload.items() if isinstance(value, (str, int, float, bool)) or value is None}


def _drop_none(value: dict[str, Any]) -> dict[str, Any]:
    """Handle the internal drop none helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._drop_none.
    """
    return {key: item for key, item in value.items() if item is not None}


def _dedupe(values: list[Any]) -> list[Any]:
    """Handle the internal dedupe helper path for this module.

    Inputs:
        Receives values for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._dedupe.
    """
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _looks_like_mutation_refusal(value: str) -> bool:
    """Handle the internal looks like mutation refusal helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_result_normalizer._looks_like_mutation_refusal.
    """
    lowered = value.lower()
    return "read-only slurm" in lowered and ("unsupported operation" in lowered or "supports read-only" in lowered)


@dataclass
class _ChildContext:
    """Represent child context within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by _ChildContext.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.slurm_result_normalizer._ChildContext and related tests.
    """
    source_action: str
    source_args: dict[str, Any] = field(default_factory=dict)
