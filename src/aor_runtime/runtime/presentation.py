from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from aor_runtime.runtime.markdown import cell as md_cell
from aor_runtime.runtime.markdown import code_block as md_code_block
from aor_runtime.runtime.markdown import section as md_section
from aor_runtime.runtime.markdown import table as md_table
from aor_runtime.runtime.slurm_result_normalizer import NormalizedSlurmResult, SlurmResultKind, normalize_slurm_result


PresentationMode = Literal["user", "debug", "raw"]

INTERNAL_KEYS = {
    "coverage",
    "semantic_frame",
    "slurm_semantic_frame",
    "sql_semantic_frame",
    "constraints",
    "unresolved_constraints",
    "covered_constraint_ids",
    "missing_constraint_ids",
    "covered_request_ids",
    "missing_request_ids",
    "request_ids",
    "planning_metadata",
    "raw_output",
    "stdout",
    "stderr",
    "command",
}

LLM_FORBIDDEN_KEY_PARTS = {
    "password",
    "token",
    "secret",
    "credential",
    "key",
    "raw_output",
    "stdout",
    "stderr",
    "semantic_frame",
    "coverage",
    "telemetry",
    "environment",
    "env",
    "rows",
    "payload",
}


@dataclass
class PresentationContext:
    mode: PresentationMode = "user"
    enable_llm_summary: bool = False
    max_rows: int = 20
    max_items: int = 20
    include_telemetry: bool = False
    include_raw: bool = False
    source_action: str | None = None
    source_args: dict[str, Any] = field(default_factory=dict)
    output_mode: str = "text"
    goal: str = ""
    include_row_samples: bool = False
    include_paths: bool = False


@dataclass
class PresentationResult:
    markdown: str
    summary: dict[str, Any] = field(default_factory=dict)
    redacted: bool = True
    raw_available: bool = False
    warnings: list[str] = field(default_factory=list)


def present_result(result: Any, context: PresentationContext | None = None) -> PresentationResult:
    ctx = context or PresentationContext()
    if ctx.mode == "raw":
        return PresentationResult(markdown=_render_raw(result), summary={"raw": True}, redacted=False, raw_available=True)
    if _is_slurm_result(result, ctx):
        return present_slurm_result(_as_dict(result), ctx)
    if _is_sql_result(result, ctx):
        return present_sql_result(result, ctx)
    if _is_filesystem_result(result, ctx):
        return present_filesystem_result(result, ctx)
    return present_generic_result(result, ctx)


def present_slurm_result(result: dict[str, Any], context: PresentationContext) -> PresentationResult:
    clean = strip_internal_telemetry(result) if context.mode == "user" else dict(result)
    normalized = normalize_slurm_result(clean, context)
    lines = _render_normalized_slurm(normalized, context)

    if context.mode == "debug" or context.include_telemetry:
        lines.extend(["", *md_section("Debug Metadata")])
        lines.extend(md_code_block("json", _json_dumps(_compact_debug(clean))))

    return PresentationResult(
        markdown="\n".join(lines).strip(),
        summary=normalized.summary,
        redacted=context.mode == "user",
        raw_available=True,
        warnings=list(normalized.warnings),
    )


def _render_normalized_slurm(normalized: NormalizedSlurmResult, context: PresentationContext) -> list[str]:
    if normalized.kind == SlurmResultKind.COMPOUND:
        return _render_slurm_compound(normalized, context)
    if normalized.kind == SlurmResultKind.QUEUE_JOBS:
        return _render_slurm_queue_jobs(normalized, context)
    if normalized.kind == SlurmResultKind.QUEUE_GROUPED_COUNTS:
        return _render_slurm_grouped_counts(normalized)
    if normalized.kind in {SlurmResultKind.PROBLEMATIC_NODES, SlurmResultKind.NODE_ROWS}:
        return _render_slurm_nodes(normalized, context)
    if normalized.kind == SlurmResultKind.PARTITION_SUMMARY:
        return _render_slurm_partitions(normalized, context)
    if normalized.kind == SlurmResultKind.ACCOUNTING_JOBS:
        return _render_slurm_accounting_jobs(normalized, context)
    if normalized.kind == SlurmResultKind.ACCOUNTING_COUNTS:
        return _render_slurm_grouped_counts(normalized)
    if normalized.kind == SlurmResultKind.ACCOUNTING_AGGREGATE:
        return _render_slurm_accounting_aggregate(normalized, context)
    if normalized.kind == SlurmResultKind.GPU_SUMMARY:
        return _render_slurm_gpu(normalized)
    if normalized.kind == SlurmResultKind.SLURMDBD_HEALTH:
        return _render_slurmdbd(normalized)
    if normalized.kind == SlurmResultKind.UNSUPPORTED_MUTATION:
        return _render_slurm_refusal(normalized)
    return _render_slurm_summary(normalized)


def _render_slurm_compound(normalized: NormalizedSlurmResult, context: PresentationContext) -> list[str]:
    lines = md_section(normalized.title)
    overall = _slurm_overall_rows(normalized.summary)
    if overall:
        lines.extend(["", *md_section("Overall")])
        lines.extend(f"- {label}: {value}" for label, value in overall)
    children = [child for child in normalized.grouped.get("children", []) if isinstance(child, NormalizedSlurmResult)]
    for child in children:
        child_lines = _render_normalized_slurm(child, context)
        if child_lines:
            lines.extend(["", *child_lines])
    _append_slurm_notes(lines, normalized.warnings)
    return lines


def _render_slurm_queue_jobs(normalized: NormalizedSlurmResult, context: PresentationContext) -> list[str]:
    total = normalized.total_count if normalized.total_count is not None else len(normalized.rows)
    returned = normalized.returned_count if normalized.returned_count is not None else len(normalized.rows)
    state = str(normalized.summary.get("state") or "").lower()
    noun = f"{state} jobs" if state else "jobs"
    lines = md_section(normalized.title)
    if total == 0:
        lines.append("")
        lines.append("No matching SLURM jobs were found.")
        return lines
    if normalized.truncated and normalized.total_count is not None:
        lines.append("")
        lines.append(f"Showing **{_format_number(returned)}** of **{_format_number(total)}** {noun}.")
    else:
        lines.append("")
        lines.append(f"Found **{_format_number(total)}** {noun}.")
    rows, truncated = _limit_rows(normalized.rows, context.max_rows)
    if rows:
        lines.extend(["", *md_table(
            ["Job ID", "User", "State", "Partition", "Name", "Time", "Nodes", "Reason"],
            [
                [
                    row.get("job_id"),
                    row.get("user"),
                    row.get("state"),
                    row.get("partition"),
                    row.get("name"),
                    row.get("time") or row.get("elapsed"),
                    row.get("nodes") or row.get("alloc_cpus"),
                    row.get("reason") or row.get("exit_code"),
                ]
                for row in rows
            ],
            alignments=["left", "left", "left", "left", "left", "right", "right", "left"],
        )])
    if truncated:
        lines.append(f"\nShowing first {len(rows)} rendered rows.")
    return lines


def _render_slurm_grouped_counts(normalized: NormalizedSlurmResult) -> list[str]:
    lines = md_section(normalized.title)
    if not normalized.grouped:
        lines.extend(["", "No matching SLURM jobs were found."])
        return lines
    lines.extend(["", *md_table(
        ["Group", "Count"],
        [[key, _format_number(count)] for key, count in sorted(normalized.grouped.items(), key=lambda item: str(item[0]))],
        alignments=["left", "right"],
    )])
    _append_slurm_notes(lines, normalized.warnings)
    return lines


def _render_slurm_nodes(normalized: NormalizedSlurmResult, context: PresentationContext) -> list[str]:
    lines = md_section(normalized.title)
    summary = normalized.summary
    if normalized.kind == SlurmResultKind.PROBLEMATIC_NODES:
        unique_count = int(summary.get("problematic_unique_count") or summary.get("unique_count") or len(normalized.rows))
        partition_rows = int(summary.get("problematic_partition_row_count") or summary.get("partition_row_count") or len(normalized.rows))
        lines.extend(["", f"Found **{_format_number(unique_count)} unique problematic nodes** affecting **{_format_number(partition_rows)} partition rows**."])
    else:
        lines.extend(["", f"Found **{_format_number(summary.get('unique_count', len(normalized.rows)))} unique nodes**."])
    rows, truncated = _limit_rows(normalized.rows, context.max_rows)
    if rows:
        lines.extend(["", *md_table(
            ["Node", "States Seen", "Partitions", "GRES"],
            [
                [
                    row.get("name"),
                    ", ".join(row.get("states_seen")) if isinstance(row.get("states_seen"), list) else row.get("state"),
                    ", ".join(row.get("partitions")) if isinstance(row.get("partitions"), list) else row.get("partition"),
                    _short_gres(row.get("gres")),
                ]
                for row in rows
            ],
        )])
    if truncated:
        lines.append(f"\nShowing first {len(rows)} of {len(normalized.rows)} nodes.")
    _append_slurm_notes(lines, normalized.warnings)
    return lines


def _render_slurm_partitions(normalized: NormalizedSlurmResult, context: PresentationContext) -> list[str]:
    lines = md_section(normalized.title)
    grouped_items = list(normalized.grouped.items())
    rows, truncated = _limit_rows(grouped_items, context.max_rows)
    if not rows:
        lines.extend(["", "No SLURM partitions were found."])
        return lines
    lines.extend(["", *md_table(
        ["Partition", "Rows", "States", "Nodes", "CPU Summary", "GRES Summary"],
        [
            [
                partition,
                _format_number(data.get("rows")),
                ", ".join(f"{state}:{count}" for state, count in dict(data.get("states") or {}).items()),
                _format_number(data.get("nodes")),
                ", ".join(list(data.get("cpu_summaries") or [])[:3]),
                _short_gres(", ".join(list(dict(data.get("gres") or {}).keys())[:3])),
            ]
            for partition, data in rows
        ],
        alignments=["left", "right", "left", "right", "left", "left"],
    )])
    if truncated:
        lines.append(f"\nShowing first {len(rows)} of {len(grouped_items)} partitions.")
    return lines


def _render_slurm_accounting_jobs(normalized: NormalizedSlurmResult, context: PresentationContext) -> list[str]:
    lines = md_section(normalized.title)
    total = normalized.total_count if normalized.total_count is not None else len(normalized.rows)
    if total == 0:
        filters = dict(normalized.summary.get("filters") or {})
        if filters.get("min_elapsed_seconds"):
            lines.extend(["", "No jobs longer than the requested duration were found in the selected accounting window."])
        else:
            lines.extend(["", "No matching accounting jobs were found."])
        return lines
    lines.extend(["", f"Found **{_format_number(total)}** accounting jobs."])
    rows, truncated = _limit_rows(normalized.rows, context.max_rows)
    if rows:
        lines.extend(["", *md_table(
            ["Job ID", "User", "State", "Partition", "Name", "Elapsed", "Start", "End", "Exit Code"],
            [
                [
                    row.get("job_id"),
                    row.get("user"),
                    row.get("state"),
                    row.get("partition"),
                    row.get("name"),
                    row.get("elapsed"),
                    row.get("start"),
                    row.get("end"),
                    row.get("exit_code"),
                ]
                for row in rows
            ],
            alignments=["left", "left", "left", "left", "left", "right", "left", "left", "left"],
        )])
    if normalized.truncated:
        lines.append(f"\nShowing **{_format_number(normalized.returned_count)}** of **{_format_number(total)}** accounting jobs.")
    elif truncated:
        lines.append(f"\nShowing first {len(rows)} rendered rows.")
    return lines


def _render_slurm_accounting_aggregate(normalized: NormalizedSlurmResult, context: PresentationContext) -> list[str]:
    del context
    summary = normalized.summary
    metric = str(summary.get("metric") or "average_elapsed")
    partition = summary.get("partition")
    include_all_states = bool(summary.get("include_all_states") or False)
    default_state_applied = bool(summary.get("default_state_applied") or False)
    state = _slurm_runtime_state_label(summary.get("state"), include_all_states, default_state_applied)
    job_count = int(summary.get("job_count") or 0)
    lines = md_section(normalized.title)
    lines.append("")
    if job_count == 0:
        target = f" for `{partition}`" if partition else ""
        lines.append(f"No matching accounting jobs were found{target}.")
    else:
        label = _runtime_metric_label(metric)
        value = _runtime_metric_value(metric, summary)
        target = f" on `{partition}`" if partition else ""
        lines.append(f"**{label}{target}:** {value}")
    rows = [
        ("Jobs included", _format_number(job_count)),
        ("Average elapsed", summary.get("average_elapsed_human")),
        ("Minimum elapsed", summary.get("min_elapsed_human")),
        ("Maximum elapsed", summary.get("max_elapsed_human")),
        ("Total elapsed", summary.get("sum_elapsed_human")),
        ("State filter", state),
        ("Partition", partition),
        ("Time window", summary.get("time_window_label") or _time_window_text(summary)),
    ]
    if metric == "count_longer_than":
        rows.insert(1, ("Threshold", summary.get("threshold_human")))
    lines.extend(["", *md_table(
        ["Metric", "Value"],
        [[key, value] for key, value in rows if value not in (None, "")],
        alignments=["left", "right"],
    )])
    groups = list(normalized.rows or [])
    if groups:
        lines.extend(["", *md_section("Groups")])
        lines.extend(["", *md_table(
            ["Group", "Jobs", "Average Elapsed", "Min", "Max", "Total"],
            [
                [
                    row.get("key"),
                    _format_number(row.get("job_count")),
                    row.get("average_elapsed_human"),
                    row.get("min_elapsed_human"),
                    row.get("max_elapsed_human"),
                    row.get("sum_elapsed_human"),
                ]
                for row in groups
            ],
            alignments=["left", "right", "right", "right", "right", "right"],
        )])
    if include_all_states:
        normalized.warnings.append("Included all job states; no completed-only filter was applied.")
    elif default_state_applied:
        normalized.warnings.append("Defaulted to completed jobs for runtime calculation.")
    _append_slurm_notes(lines, normalized.warnings)
    return lines


def _render_slurm_gpu(normalized: NormalizedSlurmResult) -> list[str]:
    summary = normalized.summary
    lines = md_section(normalized.title)
    lines.append("")
    lines.append(f"- GPU availability: {_yes_no(summary.get('available'))}")
    if "nodes_with_gpu" in summary:
        lines.append(f"- GPU-capable nodes: {_format_number(summary.get('nodes_with_gpu'))}")
    if "total_gpus" in summary:
        lines.append(f"- Total GPUs: {_format_unknown_number(summary.get('total_gpus'))}")
    if summary.get("gpu_memory_gres"):
        lines.append("- GPU memory GRES: available in debug/raw view")
    return lines


def _render_slurmdbd(normalized: NormalizedSlurmResult) -> list[str]:
    summary = normalized.summary
    lines = md_section(normalized.title)
    lines.append("")
    lines.append(f"- Available: {_yes_no(summary.get('available'))}")
    lines.append(f"- Status: {_cell(str(summary.get('status') or 'unknown').upper())}")
    if summary.get("message"):
        lines.append(f"- Message: {_cell(summary.get('message'))}")
    return lines


def _render_slurm_refusal(normalized: NormalizedSlurmResult) -> list[str]:
    return [
        *md_section("Request Not Allowed"),
        "",
        "I can inspect SLURM jobs and cluster status, but I cannot cancel, submit, drain, resume, or modify jobs/nodes.",
        "",
        "Try:",
        "- Show pending SLURM jobs.",
        "- Count pending SLURM jobs.",
        "- Show SLURM node status.",
    ]


def _render_slurm_summary(normalized: NormalizedSlurmResult) -> list[str]:
    lines = md_section(normalized.title)
    overall = _slurm_overall_rows(normalized.summary)
    if overall:
        lines.extend(["", *md_section("Overall")])
        lines.extend(f"- {label}: {value}" for label, value in overall)
    _append_slurm_notes(lines, normalized.warnings)
    return lines


def _append_slurm_notes(lines: list[str], warnings: list[str]) -> None:
    if not warnings:
        return
    lines.extend(["", *md_section("Notes")])
    lines.extend(f"- {warning}" for warning in warnings)


def _runtime_metric_label(metric: str) -> str:
    return {
        "average_elapsed": "Average runtime",
        "min_elapsed": "Minimum runtime",
        "max_elapsed": "Maximum runtime",
        "sum_elapsed": "Total runtime",
        "count": "Matching jobs",
        "count_longer_than": "Jobs longer than threshold",
        "runtime_summary": "Average runtime",
    }.get(metric, "Runtime")


def _runtime_metric_value(metric: str, summary: dict[str, Any]) -> str:
    if metric in {"count", "count_longer_than"}:
        return _format_number(summary.get("job_count"))
    if metric == "min_elapsed":
        return str(summary.get("min_elapsed_human") or "Unknown")
    if metric == "max_elapsed":
        return str(summary.get("max_elapsed_human") or "Unknown")
    if metric == "sum_elapsed":
        return str(summary.get("sum_elapsed_human") or "Unknown")
    return str(summary.get("average_elapsed_human") or summary.get("value_human") or "Unknown")


def _slurm_runtime_state_label(state: Any, include_all_states: bool, default_state_applied: bool) -> str:
    if include_all_states:
        return "All states"
    normalized = str(state or "").strip()
    if not normalized:
        return "All states"
    if default_state_applied and normalized.upper() == "COMPLETED":
        return "Completed jobs (default)"
    return normalized


def _time_window_text(summary: dict[str, Any]) -> str:
    start = summary.get("start")
    end = summary.get("end")
    if start and end:
        return f"{start} to {end}"
    if start:
        return f"Since {start}"
    if end:
        return f"Until {end}"
    return ""


def present_sql_result(result: Any, context: PresentationContext) -> PresentationResult:
    rows: list[Any]
    row_count: int
    database = ""
    if isinstance(result, dict) and "rows" in result:
        rows = list(result.get("rows") or [])
        row_count = int(result.get("row_count", len(rows)) or 0)
        database = str(result.get("database") or "")
    elif isinstance(result, list):
        rows = result
        row_count = len(rows)
    else:
        scalar = _scalar_value(result)
        markdown = f"Count: {_format_number(scalar)}" if context.output_mode == "count" else str(scalar)
        return PresentationResult(markdown=markdown, summary={"value": scalar}, raw_available=True)

    scalar = _single_row_scalar(rows)
    if scalar is not None and context.output_mode == "count":
        return PresentationResult(
            markdown=f"Count: {_format_number(scalar)}",
            summary={"database": database, "row_count": row_count, "count": scalar},
            raw_available=True,
        )

    rendered_table = _markdown_table(rows, context.max_rows)
    lines = md_section("SQL Results", [f"- Rows returned: {_format_number(row_count)}"])
    if database:
        lines.append(f"- Database: `{database}`")
    if rendered_table:
        lines.extend(["", rendered_table])
    else:
        lines.append("\nNo rows returned.")
    if len(rows) > context.max_rows:
        lines.append(f"\nShowing first {context.max_rows} of {len(rows)} rows.")
    if context.mode == "debug":
        lines.extend(["", *md_section("Debug Metadata")])
        lines.extend(md_code_block("json", _json_dumps({"database": database, "row_count": row_count})))
    return PresentationResult(markdown="\n".join(lines).strip(), summary={"database": database, "row_count": row_count}, raw_available=True)


def present_filesystem_result(result: Any, context: PresentationContext) -> PresentationResult:
    payload = _as_dict(result)
    if {"file_count", "total_size_bytes"} <= set(payload):
        file_count = int(payload.get("file_count") or 0)
        total = int(payload.get("total_size_bytes") or 0)
        display_size = str(payload.get("display_size") or f"{total} bytes")
        lines = [f"Found {_format_number(file_count)} files totaling {display_size}."]
        matches = list(payload.get("matches") or [])
        if matches:
            rows, truncated = _limit_rows(matches, context.max_rows)
            lines.extend(["", *md_table(
                ["Path", "Size"],
                [[f"`{_cell(item.get('relative_path') or item.get('path') or item.get('name'))}`", _format_number(item.get("size_bytes"))] for item in rows],
                alignments=["left", "right"],
            )])
            if truncated:
                lines.append(f"\nShowing first {len(rows)} of {len(matches)} files.")
        return PresentationResult(
            markdown="\n".join(lines),
            summary={"file_count": file_count, "total_size_bytes": total},
            raw_available=True,
        )
    if "matches" in payload:
        matches = list(payload.get("matches") or [])
        lines = [f"Found {_format_number(len(matches))} matches."]
        for item in matches[: context.max_items]:
            lines.append(f"- `{item}`")
        if len(matches) > context.max_items:
            lines.append(f"\nShowing first {context.max_items} of {len(matches)} matches.")
        return PresentationResult(markdown="\n".join(lines), summary={"match_count": len(matches)}, raw_available=True)
    return present_generic_result(result, context)


def present_generic_result(result: Any, context: PresentationContext) -> PresentationResult:
    clean = strip_internal_telemetry(result) if context.mode == "user" else result
    if isinstance(clean, (str, int, float, bool)) or clean is None:
        return PresentationResult(markdown=str(clean if clean is not None else ""), summary={"type": type(clean).__name__})
    if isinstance(clean, list) and all(isinstance(item, dict) for item in clean):
        table = _markdown_table(clean, context.max_rows)
        if table:
            suffix = f"\n\nShowing first {context.max_rows} of {len(clean)} rows." if len(clean) > context.max_rows else ""
            return PresentationResult(markdown=f"{table}{suffix}", summary={"row_count": len(clean)}, raw_available=True)
    if isinstance(clean, dict):
        if _is_small(clean):
            lines = md_table(
                ["Field", "Value"],
                [[f"`{_cell(key)}`", _compact_value(value)] for key, value in clean.items()],
            )
            return PresentationResult(markdown="\n".join(lines), summary={"keys": list(clean)}, raw_available=True)
        keys = ", ".join(f"`{key}`" for key in list(clean)[: context.max_items])
        return PresentationResult(
            markdown=f"Structured result with top-level fields: {keys}.\n\nRaw output is available in debug/raw mode.",
            summary={"keys": list(clean), "large": True},
            raw_available=True,
        )
    return PresentationResult(markdown=_render_raw(clean), summary={"type": type(clean).__name__}, raw_available=True)


def strip_internal_telemetry(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): strip_internal_telemetry(item)
            for key, item in value.items()
            if str(key) not in INTERNAL_KEYS and not str(key).startswith(("sql_", "slurm_"))
        }
    if isinstance(value, list):
        return [strip_internal_telemetry(item) for item in value]
    return value


def build_sanitized_presentation_facts(
    result: Any,
    actions: list[Any],
    context: PresentationContext,
) -> dict[str, Any]:
    clean = strip_internal_telemetry(result)
    action_tools = [_action_attr(action, "tool") for action in actions if _action_attr(action, "tool") and _action_attr(action, "tool") != "runtime.return"]
    source_action = str(context.source_action or (action_tools[-1] if action_tools else ""))

    if source_action == "sql.query" or isinstance(clean, dict) and "database" in clean and "rows" in clean:
        rows = list(clean.get("rows") or []) if isinstance(clean, dict) else []
        facts: dict[str, Any] = {
            "domain": "sql",
            "database": clean.get("database") if isinstance(clean, dict) else None,
            "row_count": clean.get("row_count", len(rows)) if isinstance(clean, dict) else len(rows),
            "tools": action_tools,
        }
        scalar = _single_row_scalar(rows)
        if scalar is not None:
            facts["result"] = {"count" if context.output_mode == "count" else "value": scalar}
        if context.include_row_samples and rows:
            facts["row_samples"] = [_compact_row(row) for row in rows[: min(context.max_rows, 5)] if isinstance(row, dict)]
        return _drop_empty(facts)

    if source_action.startswith("slurm.") or _is_slurm_result(clean, context):
        rendered = present_slurm_result(_as_dict(clean), PresentationContext(mode="user", max_rows=context.max_rows))
        summary = dict(rendered.summary or {})
        gpu = dict(summary.get("gpu") or {})
        slurmdbd = dict(summary.get("slurmdbd") or {})
        facts = {
            "domain": "slurm",
            "queue": {
                "total_jobs": summary.get("queue_count"),
                "running_jobs": summary.get("running_jobs"),
                "pending_jobs": summary.get("pending_jobs"),
            },
            "nodes": {
                "total_nodes": summary.get("node_count"),
                "problematic_nodes": summary.get("problematic_nodes"),
                "states": summary.get("node_states"),
            },
            "gpu": {
                "available": gpu.get("available"),
                "gpu_capable_nodes": gpu.get("nodes_with_gpu"),
                "total_gpus": gpu.get("total_gpus"),
            },
            "accounting": {"slurmdbd_status": slurmdbd.get("status")},
            "tools": action_tools,
            "warnings": _slurm_notes(summary),
        }
        return _drop_empty(facts)

    if source_action.startswith("fs.") or isinstance(clean, dict) and any(key in clean for key in ("file_count", "total_size_bytes", "matches", "entries")):
        payload = _as_dict(clean)
        facts = {
            "domain": "filesystem",
            "operation": source_action or "filesystem",
            "pattern": payload.get("pattern"),
            "file_count": payload.get("file_count"),
            "total_size_bytes": payload.get("total_size_bytes"),
            "total_size_human": payload.get("display_size") or payload.get("summary_text"),
            "tools": action_tools,
        }
        if context.include_paths and payload.get("path"):
            facts["path_scope"] = payload.get("path")
        else:
            facts["path_scope"] = "requested path" if payload.get("path") else None
        return _drop_empty(facts)

    return _drop_empty({"domain": "generic", "type": type(clean).__name__, "tools": action_tools})


def validate_presentation_facts_for_llm(
    facts: dict[str, Any],
    *,
    max_items: int = 50,
    max_string_length: int = 4000,
    max_depth: int = 6,
) -> bool:
    try:
        _validate_llm_fact_node(facts, max_items=max_items, max_string_length=max_string_length, max_depth=max_depth, depth=0)
    except ValueError:
        return False
    return True


def summarize_presented_facts_with_llm(facts: dict[str, Any], context: PresentationContext, settings: Any) -> str | None:
    if not context.enable_llm_summary:
        return None
    max_items = max(1, int(getattr(settings, "presentation_llm_max_facts", getattr(settings, "llm_summary_max_facts", 50))))
    max_input_chars = max(1, int(getattr(settings, "presentation_llm_max_input_chars", 4000)))
    max_output_chars = max(1, int(getattr(settings, "presentation_llm_max_output_chars", 1500)))
    compact = _cap_facts(facts, max_items=max_items)
    if not isinstance(compact, dict) or not validate_presentation_facts_for_llm(compact, max_items=max_items, max_string_length=max_input_chars):
        return None
    prompt_payload = _json_dumps(compact)
    if len(prompt_payload) > max_input_chars:
        return None
    try:
        from aor_runtime.llm.client import LLMClient

        summary = LLMClient(settings).complete(
            system_prompt=(
                "You are summarizing sanitized operational facts for a user. Use only the facts provided. "
                "Do not infer values that are not present. Do not change numbers. Do not mention internal telemetry "
                "or hidden data. Return concise Markdown."
            ),
            user_prompt=prompt_payload,
            temperature=0.0,
        ).strip()
        if not summary or summary.lstrip().startswith(("{", "[")):
            return None
        return summary[:max_output_chars].strip()
    except Exception:
        return None


def _is_slurm_result(result: Any, context: PresentationContext) -> bool:
    if str(context.source_action or "").startswith("slurm."):
        return True
    if not isinstance(result, dict):
        return False
    if "metric_group" in result or "slurmdbd_health" in result:
        return True
    if "results" in result and isinstance(result["results"], dict):
        return any(_is_slurm_result(item, PresentationContext(source_action=str(key))) for key, item in result["results"].items())
    return any(key in result for key in ("jobs", "nodes", "partitions")) and not any(key in result for key in ("database", "rows"))


def _is_sql_result(result: Any, context: PresentationContext) -> bool:
    if context.source_action == "sql.query":
        return True
    return isinstance(result, dict) and "rows" in result and "database" in result


def _is_filesystem_result(result: Any, context: PresentationContext) -> bool:
    if str(context.source_action or "").startswith("fs."):
        return True
    return isinstance(result, dict) and any(key in result for key in ("file_count", "total_size_bytes", "matches", "entries"))


def _collect_slurm_facts(result: dict[str, Any]) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    results = result.get("results")
    if isinstance(results, dict):
        for key, item in results.items():
            _merge_slurm_fact(facts, str(key), item)
    else:
        _merge_slurm_fact(facts, str(result.get("metric_group") or "slurm"), result)
    return facts


def _merge_slurm_fact(facts: dict[str, Any], key: str, value: Any) -> None:
    if not isinstance(value, dict):
        return
    payload = value.get("payload") if isinstance(value.get("payload"), dict) else value
    metric_group = str(value.get("metric_group") or key)
    if metric_group == "cluster_summary":
        facts.update({name: payload.get(name) for name in (
            "queue_count",
            "running_jobs",
            "pending_jobs",
            "node_count",
            "idle_nodes",
            "allocated_nodes",
            "mixed_nodes",
            "down_nodes",
            "drained_nodes",
        ) if name in payload})
        if "problematic_nodes" in payload:
            facts["problematic_node_count"] = payload.get("problematic_nodes")
        facts["gpu"] = {"available": payload.get("gpu_available"), "total_gpus": payload.get("total_gpus")}
    elif metric_group in {"queue_summary", "scheduler_health"}:
        facts["queue_count"] = payload.get("job_count", facts.get("queue_count"))
        facts["running_jobs"] = payload.get("running_jobs", facts.get("running_jobs"))
        facts["pending_jobs"] = payload.get("pending_jobs", facts.get("pending_jobs"))
    elif metric_group == "node_summary":
        facts["node_count"] = payload.get("node_count", facts.get("node_count"))
        facts["node_states"] = payload.get("by_state") or facts.get("node_states")
    elif metric_group == "problematic_nodes" or key == "problematic_nodes":
        facts["problematic_nodes"] = payload.get("nodes") or value.get("nodes") or facts.get("problematic_nodes")
        facts["problematic_node_count"] = payload.get("count") or value.get("count")
        facts["node_states"] = payload.get("by_state") or value.get("summary") or facts.get("node_states")
    elif metric_group == "partition_summary" or "partitions" in payload:
        facts["partitions"] = payload.get("partitions") or value.get("partitions") or facts.get("partitions")
    elif metric_group == "gpu_summary":
        facts["gpu"] = payload
    elif metric_group in {"slurmdbd_health", "accounting_health"} or "available" in value and "status" in value:
        facts["slurmdbd"] = {
            "available": value.get("available"),
            "status": value.get("status"),
            "message": value.get("message"),
        }
    elif "jobs" in value:
        count = value.get("count", len(value.get("jobs") or []))
        states = {str(job.get("state") or "").upper() for job in value.get("jobs") or [] if isinstance(job, dict)}
        if "RUNNING" in states or "running" in key:
            facts["running_jobs"] = count
        elif "PENDING" in states or "pending" in key:
            facts["pending_jobs"] = count
        else:
            facts["queue_count"] = count
    elif "nodes" in value:
        nodes = value.get("nodes") or []
        facts["problematic_nodes"] = nodes if "problematic" in key else facts.get("problematic_nodes")
        facts["node_count"] = value.get("count", len(nodes))
        facts["node_states"] = value.get("summary") or facts.get("node_states")


def _slurm_overall_rows(facts: dict[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    mapping = [
        ("Queue", "queue_count", "jobs total"),
        ("Running", "running_jobs", None),
        ("Pending", "pending_jobs", None),
        ("Nodes", "node_count", "total"),
        ("Problematic nodes", "problematic_nodes", None),
        ("Problematic partition rows", "problematic_partition_rows", None),
    ]
    for label, key, suffix in mapping:
        if key in facts and facts[key] is not None:
            value = _format_number(facts[key])
            rows.append((label, f"{value} {suffix}".strip() if suffix else value))
    if "problematic_nodes" not in facts and isinstance(facts.get("problematic_nodes"), list):
        rows.append(("Problematic nodes", _format_number(len(facts["problematic_nodes"]))))
    gpu = dict(facts.get("gpu") or {})
    if gpu:
        rows.append(("GPU availability", _yes_no(gpu.get("available"))))
    slurmdbd = dict(facts.get("slurmdbd") or {})
    if slurmdbd:
        rows.append(("SLURM accounting", str(slurmdbd.get("status") or "unknown").upper()))
    return rows


def _slurm_notes(facts: dict[str, Any]) -> list[str]:
    notes = []
    problematic_count = facts.get("problematic_nodes")
    if problematic_count is None and isinstance(facts.get("problematic_nodes"), list):
        problematic_count = len(facts["problematic_nodes"])
    if isinstance(problematic_count, int) and problematic_count > 0:
        notes.append("Some nodes are unavailable or drained. Check node details before scheduling large jobs.")
    slurmdbd = dict(facts.get("slurmdbd") or {})
    if slurmdbd and not bool(slurmdbd.get("available", True)):
        notes.append("SLURM accounting appears unavailable.")
    return notes


def _markdown_table(rows: list[Any], max_rows: int) -> str:
    dict_rows = [row for row in rows if isinstance(row, dict)]
    if not dict_rows:
        return ""
    headers = list(dict_rows[0].keys())[:8]
    return "\n".join(md_table(headers, [[row.get(header) for header in headers] for row in dict_rows[:max_rows]]))


def _limit_rows(rows: list[Any], max_rows: int) -> tuple[list[Any], bool]:
    limit = max(1, int(max_rows))
    return rows[:limit], len(rows) > limit


def _single_row_scalar(rows: list[Any]) -> Any | None:
    if len(rows) != 1 or not isinstance(rows[0], dict) or len(rows[0]) != 1:
        return None
    return next(iter(rows[0].values()))


def _scalar_value(value: Any) -> Any:
    if isinstance(value, dict) and len(value) == 1:
        return next(iter(value.values()))
    return value


def _format_number(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def _format_unknown_number(value: Any) -> str:
    if value is None or value == "Unknown":
        return "Unknown"
    return _format_number(value)


def _yes_no(value: Any) -> str:
    if value is None:
        return "Unknown"
    return "Yes" if bool(value) else "No"


def _cell(value: Any) -> str:
    return md_cell(value)


def _short_gres(value: Any) -> str:
    text = _cell(value)
    if len(text) > 60:
        return f"{text[:57]}..."
    return text


def _title(value: str) -> str:
    return value.replace("_", " ").title()


def _compact_debug(value: Any) -> Any:
    clean = strip_internal_telemetry(value)
    if isinstance(clean, dict):
        return {key: _compact_value(item) for key, item in clean.items()}
    return clean


def _compact_value(value: Any) -> Any:
    if isinstance(value, list):
        return f"{len(value)} items"
    if isinstance(value, dict):
        return f"{len(value)} fields"
    return value


def _is_small(value: dict[str, Any]) -> bool:
    if len(value) > 12:
        return False
    return all(not isinstance(item, (dict, list)) for item in value.values())


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {"value": value}


def _render_raw(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return _json_dumps(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _cap_facts(value: Any, *, max_items: int) -> Any:
    if isinstance(value, dict):
        capped: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                break
            capped[str(key)] = _cap_facts(item, max_items=max_items)
        return capped
    if isinstance(value, list):
        return [_cap_facts(item, max_items=max_items) for item in value[:max_items]]
    return value


def _action_attr(action: Any, name: str) -> Any:
    if isinstance(action, dict):
        return action.get(name)
    return getattr(action, name, None)


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): _compact_value(value)
        for key, value in row.items()
        if str(key).lower() not in {"raw", "content", "stdout", "stderr"}
    }


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            nested = _drop_empty(item)
            if nested in ({}, [], None, ""):
                continue
            cleaned[str(key)] = nested
        return cleaned
    if isinstance(value, list):
        return [item for item in (_drop_empty(item) for item in value) if item not in ({}, [], None, "")]
    return value


def _validate_llm_fact_node(
    value: Any,
    *,
    max_items: int,
    max_string_length: int,
    max_depth: int,
    depth: int,
) -> None:
    if depth > max_depth:
        raise ValueError("Presentation facts are too deeply nested for LLM summary.")
    if isinstance(value, dict):
        if len(value) > max_items:
            raise ValueError("Presentation facts contain too many fields for LLM summary.")
        for key, item in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in LLM_FORBIDDEN_KEY_PARTS):
                raise ValueError(f"Presentation facts contain forbidden key: {key}")
            _validate_llm_fact_node(item, max_items=max_items, max_string_length=max_string_length, max_depth=max_depth, depth=depth + 1)
        return
    if isinstance(value, list):
        if len(value) > max_items:
            raise ValueError("Presentation facts contain too many list items for LLM summary.")
        for item in value:
            _validate_llm_fact_node(item, max_items=max_items, max_string_length=max_string_length, max_depth=max_depth, depth=depth + 1)
        return
    if isinstance(value, str):
        if len(value) > max_string_length:
            raise ValueError("Presentation facts contain an oversized string.")
        stripped = value.strip()
        if len(stripped) > 20 and stripped[0:1] in {"{", "["}:
            raise ValueError("Presentation facts contain a raw JSON-looking string.")
