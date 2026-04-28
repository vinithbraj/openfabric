from __future__ import annotations

import os
import re
import shlex
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import ToolSpec
from aor_runtime.runtime.slurm_aggregations import aggregate_slurm_accounting_jobs
from aor_runtime.tools.base import BaseTool, ToolArgsModel, ToolExecutionError, ToolResultModel
from aor_runtime.tools.gateway import GatewayExecStreamChunk, execute_gateway_command, resolve_execution_node, stream_gateway_command


SLURM_FIXTURE_DIR_ENV = "AOR_SLURM_FIXTURE_DIR"
SLURM_TIMEOUT_SECONDS = 10
SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9._-]+$")
JOB_ID_RE = re.compile(r"^\d+(?:[._][A-Za-z0-9_-]+)*$")
NODE_RE = re.compile(r"^[A-Za-z0-9._-]+$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?$")
SQUEUE_FORMAT = "%i|%u|%T|%P|%j|%M|%D|%R"
SINFO_NODE_FORMAT = "%N|%t|%P|%c|%m|%G"
SINFO_PARTITION_FORMAT = "%P|%a|%l|%D|%t|%C|%G"
SACCT_FORMAT = "JobID,User,State,Partition,JobName,Elapsed,AllocCPUS,ReqMem,Submit,Start,End,ExitCode"
ACCOUNTING_AGGREGATE_METRICS = {
    "average_elapsed",
    "min_elapsed",
    "max_elapsed",
    "sum_elapsed",
    "count",
    "count_longer_than",
    "runtime_summary",
}
ACCOUNTING_AGGREGATE_GROUP_BY = {"partition", "user", "state", "job_name", None}


class _AggregateIntent:
    def __init__(
        self,
        *,
        user: str | None,
        state: str | None,
        partition: str | None,
        start: str | None,
        end: str | None,
        metric: str,
        group_by: str | None,
        threshold_seconds: int | None,
        time_window_label: str | None,
        include_all_states: bool = False,
        excluded_states: list[str] | None = None,
        default_state_applied: bool = False,
    ) -> None:
        self.user = user
        self.state = state
        self.include_all_states = include_all_states
        self.excluded_states = list(excluded_states or [])
        self.default_state_applied = default_state_applied
        self.partition = partition
        self.start = start
        self.end = end
        self.metric = metric
        self.group_by = group_by
        self.threshold_seconds = threshold_seconds
        self.time_window_label = time_window_label


def slurm_queue(
    settings: Settings,
    *,
    user: str | None = None,
    state: str | None = None,
    partition: str | None = None,
    group_by: str | None = None,
    limit: int | None = 100,
    gateway_node: str | None = None,
) -> dict[str, Any]:
    normalized_user = _validate_safe_token(user, field_name="user")
    normalized_partition = _validate_safe_token(partition, field_name="partition")
    normalized_state = _validate_safe_token(state, field_name="state")
    normalized_group_by = _validate_safe_token(group_by, field_name="group_by")
    normalized_limit = _validate_limit(limit)
    result = _run_command(
        settings,
        ["squeue", "-h", "-o", SQUEUE_FORMAT],
        gateway_node=gateway_node,
        fixture_name="squeue.txt",
    )
    return _queue_result_from_stdout(
        result["stdout"],
        user=normalized_user,
        state=normalized_state,
        partition=normalized_partition,
        group_by=normalized_group_by,
        limit=normalized_limit,
    )


def slurm_job_detail(settings: Settings, *, job_id: str, gateway_node: str | None = None) -> dict[str, Any]:
    normalized_job_id = _validate_job_id(job_id)
    result = _run_command(
        settings,
        ["scontrol", "show", "job", normalized_job_id],
        gateway_node=gateway_node,
        fixture_name=f"scontrol_job_{normalized_job_id}.txt",
    )
    return _job_detail_result_from_stdout(result["stdout"], job_id=normalized_job_id)


def slurm_nodes(
    settings: Settings,
    *,
    node: str | None = None,
    partition: str | None = None,
    state: str | None = None,
    state_group: str | None = None,
    gpu_only: bool = False,
    gateway_node: str | None = None,
) -> dict[str, Any]:
    normalized_node = _validate_node_name(node) if node is not None else None
    normalized_partition = _validate_safe_token(partition, field_name="partition")
    normalized_state = _validate_safe_token(state, field_name="state")
    normalized_state_group = _validate_safe_token(state_group, field_name="state_group")
    result = _run_command(
        settings,
        ["sinfo", "-h", "-N", "-o", SINFO_NODE_FORMAT],
        gateway_node=gateway_node,
        fixture_name="sinfo_nodes.txt",
    )
    return _nodes_result_from_stdout(
        result["stdout"],
        node=normalized_node,
        partition=normalized_partition,
        state=normalized_state,
        state_group=normalized_state_group,
        gpu_only=bool(gpu_only),
    )


def slurm_node_detail(settings: Settings, *, node: str, gateway_node: str | None = None) -> dict[str, Any]:
    normalized_node = _validate_node_name(node)
    result = _run_command(
        settings,
        ["scontrol", "show", "node", normalized_node],
        gateway_node=gateway_node,
        fixture_name=f"scontrol_node_{normalized_node}.txt",
    )
    return _node_detail_result_from_stdout(result["stdout"], node=normalized_node)


def slurm_partitions(settings: Settings, *, partition: str | None = None, gateway_node: str | None = None) -> dict[str, Any]:
    normalized_partition = _validate_safe_token(partition, field_name="partition")
    result = _run_command(
        settings,
        ["sinfo", "-h", "-o", SINFO_PARTITION_FORMAT],
        gateway_node=gateway_node,
        fixture_name="sinfo_partitions.txt",
    )
    return _partitions_result_from_stdout(result["stdout"], partition=normalized_partition)


def slurm_accounting(
    settings: Settings,
    *,
    user: str | None = None,
    state: str | None = None,
    partition: str | None = None,
    start: str | None = None,
    end: str | None = None,
    min_elapsed_seconds: int | None = None,
    max_elapsed_seconds: int | None = None,
    group_by: str | None = None,
    limit: int | None = 100,
    gateway_node: str | None = None,
) -> dict[str, Any]:
    normalized_user = _validate_safe_token(user, field_name="user")
    normalized_state = _validate_safe_token(state, field_name="state")
    normalized_partition = _validate_safe_token(partition, field_name="partition")
    normalized_start = _validate_time_value(start, field_name="start")
    normalized_end = _validate_time_value(end, field_name="end")
    normalized_group_by = _validate_safe_token(group_by, field_name="group_by")
    normalized_limit = _validate_limit(limit)

    command = ["sacct", "-X", "-P", f"--format={SACCT_FORMAT}"]
    if normalized_user:
        command.append(f"--user={normalized_user}")
    if normalized_state:
        command.append(f"--state={normalized_state}")
    if normalized_partition:
        command.append(f"--partition={normalized_partition}")
    if normalized_start:
        command.append(f"--starttime={normalized_start}")
    if normalized_end:
        command.append(f"--endtime={normalized_end}")

    result = _run_command(
        settings,
        command,
        gateway_node=gateway_node,
        fixture_name="sacct.txt",
    )
    return _accounting_result_from_stdout(
        result["stdout"],
        user=normalized_user,
        state=normalized_state,
        partition=normalized_partition,
        start=normalized_start,
        end=normalized_end,
        min_elapsed_seconds=min_elapsed_seconds,
        max_elapsed_seconds=max_elapsed_seconds,
        group_by=normalized_group_by,
        limit=normalized_limit,
    )


def slurm_accounting_aggregate(
    settings: Settings,
    *,
    user: str | None = None,
    state: str | None = None,
    include_all_states: bool = False,
    excluded_states: list[str] | None = None,
    default_state_applied: bool = False,
    partition: str | None = None,
    start: str | None = None,
    end: str | None = None,
    metric: str = "average_elapsed",
    group_by: str | None = None,
    threshold_seconds: int | None = None,
    limit: int | None = 1000,
    gateway_node: str | None = None,
    time_window_label: str | None = None,
) -> dict[str, Any]:
    normalized_metric = _validate_aggregate_metric(metric)
    normalized_group_by = _validate_aggregate_group_by(group_by)
    normalized_threshold = _validate_nonnegative_int(threshold_seconds, field_name="threshold_seconds")
    normalized_include_all_states = bool(include_all_states)
    normalized_excluded_states = [
        state_value
        for state_value in (_validate_safe_token(state, field_name="excluded_state") for state in list(excluded_states or []))
        if state_value
    ]
    normalized_state = None if normalized_include_all_states else state
    accounting = slurm_accounting(
        settings,
        user=user,
        state=normalized_state,
        partition=partition,
        start=start,
        end=end,
        min_elapsed_seconds=None,
        max_elapsed_seconds=None,
        group_by=None,
        limit=limit,
        gateway_node=gateway_node,
    )
    intent = _AggregateIntent(
        user=_validate_safe_token(user, field_name="user"),
        state=_validate_safe_token(normalized_state, field_name="state"),
        include_all_states=normalized_include_all_states,
        excluded_states=normalized_excluded_states,
        default_state_applied=bool(default_state_applied and not normalized_include_all_states),
        partition=_validate_safe_token(partition, field_name="partition"),
        start=_validate_time_value(start, field_name="start"),
        end=_validate_time_value(end, field_name="end"),
        metric=normalized_metric,
        group_by=normalized_group_by,
        threshold_seconds=normalized_threshold,
        time_window_label=time_window_label,
    )
    aggregate = aggregate_slurm_accounting_jobs(list(accounting.get("jobs") or []), intent)
    warnings = list(aggregate.get("warnings") or [])
    if accounting.get("truncated"):
        warnings.append(
            f"Aggregation used the first {accounting.get('returned_count')} of {accounting.get('total_count')} matching accounting rows."
        )
    aggregate.update(
        {
            "value_seconds": aggregate.get("value_seconds"),
            "value_human": aggregate.get("value_human"),
            "count": aggregate.get("count"),
            "count_longer_than": aggregate.get("count_longer_than"),
            "threshold_human": aggregate.get("threshold_human"),
            "filters": dict(accounting.get("filters") or {}),
            "source": "sacct",
            "include_all_states": normalized_include_all_states,
            "excluded_states": normalized_excluded_states,
            "default_state_applied": bool(default_state_applied and not normalized_include_all_states),
            "state": _validate_safe_token(normalized_state, field_name="state"),
            "total_count": accounting.get("total_count"),
            "returned_count": accounting.get("returned_count"),
            "limit": accounting.get("limit"),
            "truncated": accounting.get("truncated"),
            "warnings": warnings,
        }
    )
    return aggregate


def slurm_metrics(
    settings: Settings,
    *,
    metric_group: Literal[
        "cluster_summary",
        "queue_summary",
        "node_summary",
        "problematic_nodes",
        "partition_summary",
        "gpu_summary",
        "accounting_summary",
        "scheduler_health",
        "accounting_health",
    ] = "cluster_summary",
    start: str | None = None,
    end: str | None = None,
    gateway_node: str | None = None,
) -> dict[str, Any]:
    normalized_start = _validate_time_value(start, field_name="start")
    normalized_end = _validate_time_value(end, field_name="end")

    if metric_group == "cluster_summary":
        queue = slurm_queue(settings, limit=None, gateway_node=gateway_node)
        nodes = slurm_nodes(settings, gateway_node=gateway_node)
        gpu_summary = summarize_gpu_gres(nodes["nodes"])
        node_summary = summarize_slurm_nodes(nodes["nodes"], unique_by_name=True)
        problematic_summary = summarize_problematic_nodes(nodes["nodes"], unique_by_name=True)
        payload = {
            "queue_count": int(queue["count"]),
            "running_jobs": summarize_jobs_by_state(queue["jobs"]).get("RUNNING", 0),
            "pending_jobs": summarize_jobs_by_state(queue["jobs"]).get("PENDING", 0),
            "node_count": int(node_summary["unique_count"]),
            "partition_node_rows": int(node_summary["partition_row_count"]),
            "idle_nodes": int(node_summary["summary"]["idle"]),
            "allocated_nodes": int(node_summary["summary"]["allocated"]),
            "mixed_nodes": int(node_summary["summary"]["mixed"]),
            "down_nodes": int(node_summary["summary"]["down"]),
            "drained_nodes": int(node_summary["summary"]["drained"]),
            "problematic_nodes": int(problematic_summary["unique_count"]),
            "problematic_partition_rows": int(problematic_summary["partition_row_count"]),
            "gpu_available": bool(gpu_summary["available"]),
            "total_gpus": gpu_summary["total_gpus"],
        }
    elif metric_group == "queue_summary":
        queue = slurm_queue(settings, limit=None, gateway_node=gateway_node)
        by_state = summarize_jobs_by_state(queue["jobs"])
        payload = {
            "job_count": int(queue["count"]),
            "by_state": by_state,
            "by_user": _summarize_field(queue["jobs"], "user"),
            "by_partition": _summarize_field(queue["jobs"], "partition"),
            "pending_jobs": by_state.get("PENDING", 0),
            "running_jobs": by_state.get("RUNNING", 0),
        }
    elif metric_group == "node_summary":
        nodes = slurm_nodes(settings, gateway_node=gateway_node)
        node_summary = summarize_slurm_nodes(nodes["nodes"], unique_by_name=True)
        payload = {
            "node_count": int(node_summary["unique_count"]),
            "partition_node_rows": int(node_summary["partition_row_count"]),
            "by_state": dict(node_summary["summary"]),
            "by_partition": _summarize_field(nodes["nodes"], "partition"),
        }
    elif metric_group == "problematic_nodes":
        nodes = slurm_nodes(settings, state_group="problematic", gateway_node=gateway_node)
        problematic_summary = summarize_problematic_nodes(nodes["nodes"], unique_by_name=True)
        payload = {
            "count": int(problematic_summary["unique_count"]),
            "partition_row_count": int(problematic_summary["partition_row_count"]),
            "nodes": nodes["nodes"],
            "unique_nodes": problematic_summary["nodes"],
            "by_state": dict(problematic_summary["summary"]),
        }
    elif metric_group == "partition_summary":
        partitions = slurm_partitions(settings, gateway_node=gateway_node)
        payload = {"partitions": partitions["partitions"], "count": len(partitions["partitions"])}
    elif metric_group == "gpu_summary":
        nodes = slurm_nodes(settings, gateway_node=gateway_node)
        payload = summarize_gpu_gres(nodes["nodes"])
    elif metric_group in {"queue_summary", "scheduler_health"}:
        queue = slurm_queue(settings, limit=None, gateway_node=gateway_node)
        payload = {
            "job_count": int(queue["count"]),
            "by_state": summarize_jobs_by_state(queue["jobs"]),
            "by_user": _summarize_field(queue["jobs"], "user"),
            "by_partition": _summarize_field(queue["jobs"], "partition"),
            "pending_jobs": summarize_jobs_by_state(queue["jobs"]).get("PENDING", 0),
            "running_jobs": summarize_jobs_by_state(queue["jobs"]).get("RUNNING", 0),
        }
    else:
        accounting = slurm_accounting(
            settings,
            start=normalized_start,
            end=normalized_end,
            limit=None,
            gateway_node=gateway_node,
        )
        payload = {
            "job_count": int(accounting["count"]),
            "by_state": summarize_jobs_by_state(accounting["jobs"]),
            "by_user": _summarize_field(accounting["jobs"], "user"),
            "by_partition": _summarize_field(accounting["jobs"], "partition"),
        }

    return {
        "metric_group": metric_group,
        "payload": payload,
        "text_lines": _metric_text_lines(metric_group, payload),
    }


def slurm_slurmdbd_health(settings: Settings, *, gateway_node: str | None = None) -> dict[str, Any]:
    try:
        cluster_probe = _run_command(
            settings,
            ["sacctmgr", "show", "cluster", "-P"],
            gateway_node=gateway_node,
            fixture_name="sacctmgr_show_cluster.txt",
            allow_nonzero=True,
        )
    except ToolExecutionError as exc:
        return _probe_accounting_fallback(settings, str(exc), gateway_node=gateway_node)

    if cluster_probe["returncode"] == 0 and cluster_probe["stdout"].strip():
        clusters = [line.strip() for line in cluster_probe["stdout"].splitlines() if line.strip()]
        result = {
            "available": True,
            "status": "ok",
            "message": "SLURM accounting is available.",
            "clusters": clusters,
        }
        result["text_lines"] = _health_text_lines(result)
        return result
    return _probe_accounting_fallback(
        settings,
        cluster_probe["stderr"].strip() or "sacctmgr probe failed",
        gateway_node=gateway_node,
    )


def parse_squeue_output(text: str) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split("|")
        if len(parts) < 8:
            continue
        job_id, user, state, partition, name, time_used, nodes, reason = [part.strip() for part in parts[:8]]
        jobs.append(
            {
                "job_id": job_id,
                "user": user,
                "state": state,
                "partition": _normalize_partition_name(partition),
                "name": name,
                "time": time_used,
                "nodes": nodes,
                "reason": reason,
            }
        )
    return jobs


def parse_sinfo_nodes_output(text: str) -> list[dict[str, str]]:
    nodes: list[dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split("|")
        if len(parts) < 6:
            continue
        name, state, partition, cpus, memory, gres = [part.strip() for part in parts[:6]]
        nodes.append(
            {
                "name": name,
                "state": state,
                "partition": _normalize_partition_name(partition),
                "cpus": cpus,
                "memory": memory,
                "gres": gres,
            }
        )
    return nodes


def summarize_node_states(nodes: list[dict[str, str]]) -> dict[str, int]:
    summary = {"idle": 0, "allocated": 0, "mixed": 0, "down": 0, "drained": 0, "other": 0}
    for node in nodes:
        canonical = _canonical_node_state(str(node.get("state", "")))
        if canonical in summary:
            summary[canonical] += 1
        else:
            summary["other"] += 1
    return summary


def dedupe_slurm_nodes_by_name(nodes: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in nodes:
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        entry = grouped.setdefault(
            name,
            {
                "name": name,
                "states_seen": [],
                "partitions": [],
                "gres_values": [],
                "cpus": row.get("cpus", ""),
                "memory": row.get("memory", ""),
            },
        )
        state = str(row.get("state", "")).strip()
        partition = str(row.get("partition", "")).strip()
        gres = str(row.get("gres", "")).strip()
        if state and state not in entry["states_seen"]:
            entry["states_seen"].append(state)
        if partition and partition not in entry["partitions"]:
            entry["partitions"].append(partition)
        if gres and gres not in entry["gres_values"]:
            entry["gres_values"].append(gres)
    normalized = []
    for entry in grouped.values():
        states = entry.pop("states_seen")
        partitions = entry.pop("partitions")
        gres_values = entry.pop("gres_values")
        entry["state"] = ",".join(states)
        entry["states_seen"] = states
        entry["partition"] = ",".join(partitions)
        entry["partitions"] = partitions
        entry["gres"] = "; ".join(gres_values)
        entry["gres_values"] = gres_values
        normalized.append(entry)
    return sorted(normalized, key=lambda item: str(item.get("name", "")))


def summarize_slurm_nodes(nodes: list[dict[str, str]], unique_by_name: bool = True) -> dict[str, Any]:
    selected = dedupe_slurm_nodes_by_name(nodes) if unique_by_name else list(nodes)
    state_rows: list[dict[str, str]] = []
    for node in selected:
        states = node.get("states_seen") if isinstance(node, dict) else None
        if isinstance(states, list) and states:
            state = str(states[0])
        else:
            state = str(node.get("state", "")) if isinstance(node, dict) else ""
        state_rows.append({"state": state})
    return {
        "nodes": selected,
        "unique_count": len(dedupe_slurm_nodes_by_name(nodes)),
        "partition_row_count": len(nodes),
        "summary": summarize_node_states(state_rows),
    }


def summarize_problematic_nodes(nodes: list[dict[str, str]], unique_by_name: bool = True) -> dict[str, Any]:
    problematic_rows = [node for node in nodes if is_problematic_node_state(str(node.get("state", "")))]
    summary = summarize_slurm_nodes(problematic_rows, unique_by_name=unique_by_name)
    summary["partition_rows"] = problematic_rows
    return summary


def parse_sinfo_partitions_output(text: str) -> list[dict[str, str]]:
    partitions: list[dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split("|")
        if len(parts) < 7:
            continue
        partition, availability, time_limit, nodes, state, cpus, gres = [part.strip() for part in parts[:7]]
        partitions.append(
            {
                "partition": _normalize_partition_name(partition),
                "availability": availability,
                "time_limit": time_limit,
                "nodes": nodes,
                "state": state,
                "cpus": cpus,
                "gres": gres,
            }
        )
    return partitions


def parse_sacct_output(text: str) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split("|")
        if len(parts) < 12:
            continue
        job_id, user, state, partition, name, elapsed, alloc_cpus, req_mem, submit, start, end, exit_code = [
            part.strip() for part in parts[:12]
        ]
        jobs.append(
            {
                "job_id": job_id,
                "user": user,
                "state": state,
                "partition": _normalize_partition_name(partition),
                "name": name,
                "elapsed": elapsed,
                "elapsed_seconds": parse_elapsed_to_seconds(elapsed),
                "alloc_cpus": alloc_cpus,
                "req_mem": req_mem,
                "submit": submit,
                "start": start,
                "end": end,
                "exit_code": exit_code,
            }
        )
    return jobs


def parse_scontrol_kv_output(text: str) -> dict[str, str]:
    normalized = " ".join(line.strip() for line in text.splitlines() if line.strip())
    matches = re.finditer(r"([A-Za-z][A-Za-z0-9_]*)=([^=]+?)(?=\s+[A-Za-z][A-Za-z0-9_]*=|$)", normalized)
    parsed: dict[str, str] = {}
    for match in matches:
        parsed[match.group(1)] = match.group(2).strip()
    return parsed


def summarize_jobs_by_state(jobs: list[dict[str, str]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for job in jobs:
        counter[_canonical_job_state(str(job.get("state", "")))] += 1
    return dict(sorted(counter.items()))


def summarize_gpu_gres(nodes: list[dict[str, str]]) -> dict[str, Any]:
    nodes_with_gpu = 0
    total_gpus = 0
    unknown_gpu_count = False
    by_gres: Counter[str] = Counter()
    gpu_memory_gres: Counter[str] = Counter()
    for node in nodes:
        gres = str(node.get("gres", "") or "")
        if not gres or gres in {"(null)", "N/A", "none"}:
            continue
        gpu_count = 0
        node_has_gpu = False
        for token in [part.strip() for part in gres.split(",") if part.strip()]:
            cleaned = token.split("(")[0]
            lowered = cleaned.lower()
            if lowered.startswith("gpu_mem:"):
                gpu_memory_gres[cleaned] += 1
                continue
            if not lowered.startswith("gpu"):
                continue
            by_gres[cleaned] += 1
            count = _extract_gpu_count(cleaned)
            if count is None:
                unknown_gpu_count = True
                node_has_gpu = True
                continue
            if count > 0:
                node_has_gpu = True
                gpu_count += count
        if node_has_gpu:
            nodes_with_gpu += 1
            total_gpus += gpu_count
    summary = {
        "available": total_gpus > 0 or unknown_gpu_count,
        "nodes_with_gpu": nodes_with_gpu,
        "total_gpus": "Unknown" if unknown_gpu_count else total_gpus,
        "by_gres": dict(sorted(by_gres.items())),
    }
    if gpu_memory_gres:
        summary["gpu_memory_gres"] = dict(sorted(gpu_memory_gres.items()))
    return summary


def parse_elapsed_to_seconds(value: str) -> int | None:
    text = str(value or "").strip()
    if not text or text in {"Unknown", "N/A"}:
        return None
    days = 0
    time_part = text
    if "-" in text:
        day_part, time_part = text.split("-", 1)
        if day_part.isdigit():
            days = int(day_part)
    parts = [part for part in time_part.split(":") if part]
    if not parts or not all(part.isdigit() for part in parts):
        return None
    if len(parts) == 3:
        hours, minutes, seconds = [int(part) for part in parts]
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = [int(part) for part in parts]
    else:
        hours = 0
        minutes = 0
        seconds = int(parts[0])
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def is_problematic_node_state(state: str | None) -> bool:
    normalized = str(state or "").strip().lower().replace("-", "_")
    if not normalized:
        return False
    return any(token in normalized for token in ("down", "drain", "fail", "unknown", "no_respond", "not_respond"))


class SlurmQueueTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        user: str | None = None
        state: str | None = None
        partition: str | None = None
        group_by: str | None = None
        limit: int | None = 100
        gateway_node: str | None = None

    class ToolResult(ToolResultModel):
        jobs: list[dict[str, Any]]
        count: int
        total_count: int
        returned_count: int
        limit: int | None = None
        truncated: bool = False
        filters: dict[str, Any]
        group_by: str | None = None
        grouped: dict[str, int] | None = None

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="slurm.queue",
            description="Inspect the current SLURM queue in a read-only way.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "user": {"type": ["string", "null"]},
                    "state": {"type": ["string", "null"]},
                    "partition": {"type": ["string", "null"]},
                    "group_by": {"type": ["string", "null"], "enum": ["state", "user", "partition", "node", None]},
                    "limit": {"type": ["integer", "null"], "minimum": 1},
                    "gateway_node": {"type": ["string", "null"]},
                },
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult.model_validate(
            slurm_queue(
                self.settings,
                user=arguments.user,
                state=arguments.state,
                partition=arguments.partition,
                group_by=arguments.group_by,
                limit=arguments.limit,
                gateway_node=arguments.gateway_node,
            )
        )

    def stream(self, arguments: ToolArgs) -> Iterator[dict[str, Any]]:
        return _stream_command(
            self.settings,
            ["squeue", "-h", "-o", SQUEUE_FORMAT],
            gateway_node=arguments.gateway_node,
            fixture_name="squeue.txt",
        )

    def preview_command(self, arguments: ToolArgs) -> str:
        return _join_argv(["squeue", "-h", "-o", SQUEUE_FORMAT])

    def build_stream_result(
        self,
        arguments: ToolArgs,
        *,
        stdout: str,
        stderr: str,
        returncode: int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _raise_for_returncode(returncode, stderr, "squeue")
        return slurm_queue(
            self.settings,
            user=arguments.user,
            state=arguments.state,
            partition=arguments.partition,
            group_by=arguments.group_by,
            limit=arguments.limit,
            gateway_node=arguments.gateway_node,
        ) if _is_fixture_mode() else _queue_result_from_stdout(
            stdout,
            user=_validate_safe_token(arguments.user, field_name="user"),
            state=_validate_safe_token(arguments.state, field_name="state"),
            partition=_validate_safe_token(arguments.partition, field_name="partition"),
            group_by=_validate_safe_token(arguments.group_by, field_name="group_by"),
            limit=_validate_limit(arguments.limit),
        )


class SlurmJobDetailTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        job_id: str
        gateway_node: str | None = None

    class ToolResult(ToolResultModel):
        job_id: str
        raw: str
        fields: dict[str, str]
        field_rows: list[dict[str, str]]

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="slurm.job_detail",
            description="Show read-only details for a SLURM job.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "gateway_node": {"type": ["string", "null"]},
                },
                "required": ["job_id"],
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult.model_validate(
            slurm_job_detail(self.settings, job_id=arguments.job_id, gateway_node=arguments.gateway_node)
        )

    def stream(self, arguments: ToolArgs) -> Iterator[dict[str, Any]]:
        normalized_job_id = _validate_job_id(arguments.job_id)
        return _stream_command(
            self.settings,
            ["scontrol", "show", "job", normalized_job_id],
            gateway_node=arguments.gateway_node,
            fixture_name=f"scontrol_job_{normalized_job_id}.txt",
        )

    def preview_command(self, arguments: ToolArgs) -> str:
        return _join_argv(["scontrol", "show", "job", _validate_job_id(arguments.job_id)])

    def build_stream_result(
        self,
        arguments: ToolArgs,
        *,
        stdout: str,
        stderr: str,
        returncode: int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _raise_for_returncode(returncode, stderr, "scontrol")
        return _job_detail_result_from_stdout(stdout, job_id=_validate_job_id(arguments.job_id))


class SlurmNodesTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        node: str | None = None
        partition: str | None = None
        state: str | None = None
        state_group: str | None = None
        gpu_only: bool = False
        gateway_node: str | None = None

    class ToolResult(ToolResultModel):
        nodes: list[dict[str, str]]
        count: int
        partition_row_count: int
        unique_count: int
        summary: dict[str, int]
        unique_summary: dict[str, Any]
        problematic_summary: dict[str, Any]
        filters: dict[str, Any]

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="slurm.nodes",
            description="Inspect SLURM node status in a read-only way.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "node": {"type": ["string", "null"]},
                    "partition": {"type": ["string", "null"]},
                    "state": {"type": ["string", "null"]},
                    "state_group": {"type": ["string", "null"]},
                    "gpu_only": {"type": "boolean"},
                    "gateway_node": {"type": ["string", "null"]},
                },
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult.model_validate(
            slurm_nodes(
                self.settings,
                node=arguments.node,
                partition=arguments.partition,
                state=arguments.state,
                state_group=arguments.state_group,
                gpu_only=arguments.gpu_only,
                gateway_node=arguments.gateway_node,
            )
        )

    def stream(self, arguments: ToolArgs) -> Iterator[dict[str, Any]]:
        return _stream_command(
            self.settings,
            ["sinfo", "-h", "-N", "-o", SINFO_NODE_FORMAT],
            gateway_node=arguments.gateway_node,
            fixture_name="sinfo_nodes.txt",
        )

    def preview_command(self, arguments: ToolArgs) -> str:
        return _join_argv(["sinfo", "-h", "-N", "-o", SINFO_NODE_FORMAT])

    def build_stream_result(
        self,
        arguments: ToolArgs,
        *,
        stdout: str,
        stderr: str,
        returncode: int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _raise_for_returncode(returncode, stderr, "sinfo")
        normalized_node = _validate_node_name(arguments.node) if arguments.node is not None else None
        return _nodes_result_from_stdout(
            stdout,
            node=normalized_node,
            partition=_validate_safe_token(arguments.partition, field_name="partition"),
            state=_validate_safe_token(arguments.state, field_name="state"),
            state_group=_validate_safe_token(arguments.state_group, field_name="state_group"),
            gpu_only=bool(arguments.gpu_only),
        )


class SlurmNodeDetailTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        node: str
        gateway_node: str | None = None

    class ToolResult(ToolResultModel):
        node: str
        raw: str
        fields: dict[str, str]
        field_rows: list[dict[str, str]]

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="slurm.node_detail",
            description="Show read-only details for a SLURM node.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "node": {"type": "string"},
                    "gateway_node": {"type": ["string", "null"]},
                },
                "required": ["node"],
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult.model_validate(
            slurm_node_detail(self.settings, node=arguments.node, gateway_node=arguments.gateway_node)
        )

    def stream(self, arguments: ToolArgs) -> Iterator[dict[str, Any]]:
        normalized_node = _validate_node_name(arguments.node)
        return _stream_command(
            self.settings,
            ["scontrol", "show", "node", normalized_node],
            gateway_node=arguments.gateway_node,
            fixture_name=f"scontrol_node_{normalized_node}.txt",
        )

    def preview_command(self, arguments: ToolArgs) -> str:
        return _join_argv(["scontrol", "show", "node", _validate_node_name(arguments.node)])

    def build_stream_result(
        self,
        arguments: ToolArgs,
        *,
        stdout: str,
        stderr: str,
        returncode: int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _raise_for_returncode(returncode, stderr, "scontrol")
        return _node_detail_result_from_stdout(stdout, node=_validate_node_name(arguments.node))


class SlurmPartitionsTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        partition: str | None = None
        gateway_node: str | None = None

    class ToolResult(ToolResultModel):
        partitions: list[dict[str, str]]

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="slurm.partitions",
            description="Inspect SLURM partition status in a read-only way.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "partition": {"type": ["string", "null"]},
                    "gateway_node": {"type": ["string", "null"]},
                },
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult.model_validate(
            slurm_partitions(self.settings, partition=arguments.partition, gateway_node=arguments.gateway_node)
        )

    def stream(self, arguments: ToolArgs) -> Iterator[dict[str, Any]]:
        return _stream_command(
            self.settings,
            ["sinfo", "-h", "-o", SINFO_PARTITION_FORMAT],
            gateway_node=arguments.gateway_node,
            fixture_name="sinfo_partitions.txt",
        )

    def preview_command(self, arguments: ToolArgs) -> str:
        return _join_argv(["sinfo", "-h", "-o", SINFO_PARTITION_FORMAT])

    def build_stream_result(
        self,
        arguments: ToolArgs,
        *,
        stdout: str,
        stderr: str,
        returncode: int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _raise_for_returncode(returncode, stderr, "sinfo")
        return _partitions_result_from_stdout(
            stdout,
            partition=_validate_safe_token(arguments.partition, field_name="partition"),
        )


class SlurmAccountingTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        user: str | None = None
        state: str | None = None
        partition: str | None = None
        start: str | None = None
        end: str | None = None
        min_elapsed_seconds: int | None = None
        max_elapsed_seconds: int | None = None
        group_by: str | None = None
        limit: int | None = 100
        gateway_node: str | None = None

    class ToolResult(ToolResultModel):
        jobs: list[dict[str, Any]]
        count: int
        total_count: int
        returned_count: int
        limit: int | None = None
        truncated: bool = False
        filters: dict[str, Any]
        group_by: str | None = None
        grouped: dict[str, int] | None = None

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="slurm.accounting",
            description="Inspect read-only SLURM accounting records with sacct.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "user": {"type": ["string", "null"]},
                    "state": {"type": ["string", "null"]},
                    "partition": {"type": ["string", "null"]},
                    "start": {"type": ["string", "null"]},
                    "end": {"type": ["string", "null"]},
                    "min_elapsed_seconds": {"type": ["integer", "null"], "minimum": 0},
                    "max_elapsed_seconds": {"type": ["integer", "null"], "minimum": 0},
                    "group_by": {"type": ["string", "null"], "enum": ["state", "user", "partition", "node", "job_name", None]},
                    "limit": {"type": ["integer", "null"], "minimum": 1},
                    "gateway_node": {"type": ["string", "null"]},
                },
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult.model_validate(
            slurm_accounting(
                self.settings,
                user=arguments.user,
                state=arguments.state,
                partition=arguments.partition,
                start=arguments.start,
                end=arguments.end,
                min_elapsed_seconds=arguments.min_elapsed_seconds,
                max_elapsed_seconds=arguments.max_elapsed_seconds,
                group_by=arguments.group_by,
                limit=arguments.limit,
                gateway_node=arguments.gateway_node,
            )
        )

    def stream(self, arguments: ToolArgs) -> Iterator[dict[str, Any]]:
        command = _accounting_argv(arguments)
        return _stream_command(
            self.settings,
            command,
            gateway_node=arguments.gateway_node,
            fixture_name="sacct.txt",
        )

    def preview_command(self, arguments: ToolArgs) -> str:
        return _join_argv(_accounting_argv(arguments))

    def build_stream_result(
        self,
        arguments: ToolArgs,
        *,
        stdout: str,
        stderr: str,
        returncode: int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _raise_for_returncode(returncode, stderr, "sacct")
        return _accounting_result_from_stdout(
            stdout,
            user=_validate_safe_token(arguments.user, field_name="user"),
            state=_validate_safe_token(arguments.state, field_name="state"),
            partition=_validate_safe_token(arguments.partition, field_name="partition"),
            start=_validate_time_value(arguments.start, field_name="start"),
            end=_validate_time_value(arguments.end, field_name="end"),
            min_elapsed_seconds=arguments.min_elapsed_seconds,
            max_elapsed_seconds=arguments.max_elapsed_seconds,
            group_by=_validate_safe_token(arguments.group_by, field_name="group_by"),
            limit=_validate_limit(arguments.limit),
        )


class SlurmAccountingAggregateTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        user: str | None = None
        state: str | None = None
        include_all_states: bool = False
        excluded_states: list[str] = []
        default_state_applied: bool = False
        partition: str | None = None
        start: str | None = None
        end: str | None = None
        metric: str = "average_elapsed"
        group_by: str | None = None
        threshold_seconds: int | None = None
        limit: int | None = 1000
        gateway_node: str | None = None
        time_window_label: str | None = None

    class ToolResult(ToolResultModel):
        result_kind: str
        metric: str
        source: str | None = None
        partition: str | None = None
        user: str | None = None
        state: str | None = None
        include_all_states: bool = False
        excluded_states: list[str] = []
        default_state_applied: bool = False
        start: str | None = None
        end: str | None = None
        time_window_label: str | None = None
        group_by: str | None = None
        threshold_seconds: int | None = None
        job_count: int
        average_elapsed_seconds: float | None = None
        average_elapsed_human: str | None = None
        min_elapsed_seconds: int | None = None
        min_elapsed_human: str | None = None
        max_elapsed_seconds: int | None = None
        max_elapsed_human: str | None = None
        sum_elapsed_seconds: int | None = None
        sum_elapsed_human: str | None = None
        value_seconds: float | int | None = None
        value_human: str | None = None
        count: int | None = None
        count_longer_than: int | None = None
        threshold_human: str | None = None
        groups: list[dict[str, Any]] = []
        warnings: list[str] = []
        filters: dict[str, Any] = {}
        total_count: int | None = None
        returned_count: int | None = None
        limit: int | None = None
        truncated: bool = False

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="slurm.accounting_aggregate",
            description="Compute read-only SLURM accounting runtime aggregates from sacct.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "user": {"type": ["string", "null"]},
                    "state": {"type": ["string", "null"]},
                    "include_all_states": {"type": "boolean"},
                    "excluded_states": {"type": "array", "items": {"type": "string"}},
                    "default_state_applied": {"type": "boolean"},
                    "partition": {"type": ["string", "null"]},
                    "start": {"type": ["string", "null"]},
                    "end": {"type": ["string", "null"]},
                    "metric": {
                        "type": "string",
                        "enum": sorted(ACCOUNTING_AGGREGATE_METRICS),
                    },
                    "group_by": {"type": ["string", "null"], "enum": ["partition", "user", "state", "job_name", None]},
                    "threshold_seconds": {"type": ["integer", "null"], "minimum": 0},
                    "limit": {"type": ["integer", "null"], "minimum": 1},
                    "gateway_node": {"type": ["string", "null"]},
                    "time_window_label": {"type": ["string", "null"]},
                },
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult.model_validate(
            slurm_accounting_aggregate(
                self.settings,
                user=arguments.user,
                state=arguments.state,
                include_all_states=arguments.include_all_states,
                excluded_states=arguments.excluded_states,
                default_state_applied=arguments.default_state_applied,
                partition=arguments.partition,
                start=arguments.start,
                end=arguments.end,
                metric=arguments.metric,
                group_by=arguments.group_by,
                threshold_seconds=arguments.threshold_seconds,
                limit=arguments.limit,
                gateway_node=arguments.gateway_node,
                time_window_label=arguments.time_window_label,
            )
        )

    def stream(self, arguments: ToolArgs) -> Iterator[dict[str, Any]]:
        command = _accounting_argv(arguments)
        return _stream_command(
            self.settings,
            command,
            gateway_node=arguments.gateway_node,
            fixture_name="sacct.txt",
        )

    def preview_command(self, arguments: ToolArgs) -> str:
        return _join_argv(_accounting_argv(arguments))

    def build_stream_result(
        self,
        arguments: ToolArgs,
        *,
        stdout: str,
        stderr: str,
        returncode: int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _raise_for_returncode(returncode, stderr, "sacct")
        normalized_include_all_states = bool(arguments.include_all_states)
        normalized_state = None if normalized_include_all_states else _validate_safe_token(arguments.state, field_name="state")
        accounting = _accounting_result_from_stdout(
            stdout,
            user=_validate_safe_token(arguments.user, field_name="user"),
            state=normalized_state,
            partition=_validate_safe_token(arguments.partition, field_name="partition"),
            start=_validate_time_value(arguments.start, field_name="start"),
            end=_validate_time_value(arguments.end, field_name="end"),
            min_elapsed_seconds=None,
            max_elapsed_seconds=None,
            group_by=None,
            limit=_validate_limit(arguments.limit),
        )
        intent = _AggregateIntent(
            user=_validate_safe_token(arguments.user, field_name="user"),
            state=normalized_state,
            include_all_states=normalized_include_all_states,
            excluded_states=list(arguments.excluded_states or []),
            default_state_applied=bool(arguments.default_state_applied and not normalized_include_all_states),
            partition=_validate_safe_token(arguments.partition, field_name="partition"),
            start=_validate_time_value(arguments.start, field_name="start"),
            end=_validate_time_value(arguments.end, field_name="end"),
            metric=_validate_aggregate_metric(arguments.metric),
            group_by=_validate_aggregate_group_by(arguments.group_by),
            threshold_seconds=_validate_nonnegative_int(arguments.threshold_seconds, field_name="threshold_seconds"),
            time_window_label=arguments.time_window_label,
        )
        aggregate = aggregate_slurm_accounting_jobs(list(accounting.get("jobs") or []), intent)
        warnings = list(aggregate.get("warnings") or [])
        if accounting.get("truncated"):
            warnings.append(
                f"Aggregation used the first {accounting.get('returned_count')} of {accounting.get('total_count')} matching accounting rows."
            )
        aggregate.update(
        {
            "value_seconds": aggregate.get("value_seconds"),
            "value_human": aggregate.get("value_human"),
            "count": aggregate.get("count"),
            "count_longer_than": aggregate.get("count_longer_than"),
            "threshold_human": aggregate.get("threshold_human"),
                "filters": dict(accounting.get("filters") or {}),
                "source": "sacct",
                "include_all_states": normalized_include_all_states,
                "excluded_states": list(arguments.excluded_states or []),
                "default_state_applied": bool(arguments.default_state_applied and not normalized_include_all_states),
                "state": normalized_state,
                "total_count": accounting.get("total_count"),
                "returned_count": accounting.get("returned_count"),
                "limit": accounting.get("limit"),
                "truncated": accounting.get("truncated"),
                "warnings": warnings,
            }
        )
        return aggregate


class SlurmMetricsTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        metric_group: Literal[
            "cluster_summary",
            "queue_summary",
            "node_summary",
            "problematic_nodes",
            "partition_summary",
            "gpu_summary",
            "accounting_summary",
            "scheduler_health",
            "accounting_health",
        ] = "cluster_summary"
        start: str | None = None
        end: str | None = None
        gateway_node: str | None = None

    class ToolResult(ToolResultModel):
        metric_group: str
        payload: dict[str, Any]
        text_lines: list[str]

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="slurm.metrics",
            description="Collect read-only SLURM summary metrics suitable for dashboards.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "metric_group": {
                        "type": "string",
                        "enum": [
                            "cluster_summary",
                            "queue_summary",
                            "node_summary",
                            "problematic_nodes",
                            "partition_summary",
                            "gpu_summary",
                            "accounting_summary",
                            "scheduler_health",
                            "accounting_health",
                        ],
                    },
                    "start": {"type": ["string", "null"]},
                    "end": {"type": ["string", "null"]},
                    "gateway_node": {"type": ["string", "null"]},
                },
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult.model_validate(
            slurm_metrics(
                self.settings,
                metric_group=arguments.metric_group,
                start=arguments.start,
                end=arguments.end,
                gateway_node=arguments.gateway_node,
            )
        )


class SlurmDBDHealthTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        gateway_node: str | None = None

    class ToolResult(ToolResultModel):
        available: bool
        status: str
        message: str
        clusters: list[str] = []
        text_lines: list[str]

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="slurm.slurmdbd_health",
            description="Check read-only SLURM accounting/SLURMDBD health.",
            arguments_schema={
                "type": "object",
                "properties": {"gateway_node": {"type": ["string", "null"]}},
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult.model_validate(
            slurm_slurmdbd_health(self.settings, gateway_node=arguments.gateway_node)
        )

    def preview_command(self, arguments: ToolArgs) -> str:
        return _join_argv(["sacctmgr", "show", "cluster", "-P"])


def _queue_result_from_stdout(
    stdout: str,
    *,
    user: str | None,
    state: str | None,
    partition: str | None,
    group_by: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    jobs = parse_squeue_output(stdout)
    filtered = _filter_jobs(jobs, user=user, state=state, partition=partition)
    total_count = len(filtered)
    if limit is not None:
        filtered = filtered[:limit]
    result: dict[str, Any] = {
        "jobs": filtered,
        "count": total_count,
        "total_count": total_count,
        "returned_count": len(filtered),
        "limit": limit,
        "truncated": len(filtered) < total_count,
        "filters": {"user": user, "state": state, "partition": partition},
        "group_by": group_by,
        "grouped": None,
    }
    if group_by:
        result["grouped"] = _summarize_field(filtered, group_by)
    return result


def _job_detail_result_from_stdout(stdout: str, *, job_id: str) -> dict[str, Any]:
    fields = parse_scontrol_kv_output(stdout)
    return {
        "job_id": job_id,
        "raw": stdout.strip(),
        "fields": fields,
        "field_rows": _field_rows(fields),
    }


def _nodes_result_from_stdout(
    stdout: str,
    *,
    node: str | None,
    partition: str | None,
    state: str | None,
    state_group: str | None = None,
    gpu_only: bool = False,
) -> dict[str, Any]:
    nodes = parse_sinfo_nodes_output(stdout)
    filtered = _filter_nodes(nodes, node=node, partition=partition, state=state, state_group=state_group, gpu_only=gpu_only)
    unique_summary = summarize_slurm_nodes(filtered, unique_by_name=True)
    problematic_summary = summarize_problematic_nodes(filtered, unique_by_name=True)
    return {
        "nodes": filtered,
        "count": len(filtered),
        "partition_row_count": len(filtered),
        "unique_count": int(unique_summary["unique_count"]),
        "summary": summarize_node_states(filtered),
        "unique_summary": unique_summary,
        "problematic_summary": problematic_summary,
        "filters": {"node": node, "partition": partition, "state": state, "state_group": state_group, "gpu_only": gpu_only},
    }


def _node_detail_result_from_stdout(stdout: str, *, node: str) -> dict[str, Any]:
    fields = parse_scontrol_kv_output(stdout)
    return {
        "node": node,
        "raw": stdout.strip(),
        "fields": fields,
        "field_rows": _field_rows(fields),
    }


def _partitions_result_from_stdout(stdout: str, *, partition: str | None) -> dict[str, Any]:
    partitions = parse_sinfo_partitions_output(stdout)
    if partition:
        partitions = [item for item in partitions if _normalize_partition_name(item["partition"]) == partition]
    return {"partitions": partitions}


def _accounting_result_from_stdout(
    stdout: str,
    *,
    user: str | None,
    state: str | None,
    partition: str | None,
    start: str | None,
    end: str | None,
    min_elapsed_seconds: int | None = None,
    max_elapsed_seconds: int | None = None,
    group_by: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    jobs = parse_sacct_output(stdout)
    filtered = _filter_jobs(jobs, user=user, state=state, partition=partition)
    filtered = _filter_jobs_by_time(filtered, start=start, end=end)
    filtered = _filter_jobs_by_elapsed(
        filtered,
        min_elapsed_seconds=min_elapsed_seconds,
        max_elapsed_seconds=max_elapsed_seconds,
    )
    total_count = len(filtered)
    if limit is not None:
        filtered = filtered[:limit]
    result: dict[str, Any] = {
        "jobs": filtered,
        "count": total_count,
        "total_count": total_count,
        "returned_count": len(filtered),
        "limit": limit,
        "truncated": len(filtered) < total_count,
        "filters": {
            "user": user,
            "state": state,
            "partition": partition,
            "start": start,
            "end": end,
            "min_elapsed_seconds": min_elapsed_seconds,
            "max_elapsed_seconds": max_elapsed_seconds,
        },
        "group_by": group_by,
        "grouped": None,
    }
    if group_by:
        result["grouped"] = _summarize_field(filtered, group_by)
    return result


def _accounting_argv(arguments: Any) -> list[str]:
    normalized_user = _validate_safe_token(getattr(arguments, "user", None), field_name="user")
    include_all_states = bool(getattr(arguments, "include_all_states", False))
    normalized_state = None if include_all_states else _validate_safe_token(getattr(arguments, "state", None), field_name="state")
    normalized_partition = _validate_safe_token(getattr(arguments, "partition", None), field_name="partition")
    normalized_start = _validate_time_value(getattr(arguments, "start", None), field_name="start")
    normalized_end = _validate_time_value(getattr(arguments, "end", None), field_name="end")
    command = ["sacct", "-X", "-P", f"--format={SACCT_FORMAT}"]
    if normalized_user:
        command.append(f"--user={normalized_user}")
    if normalized_state:
        command.append(f"--state={normalized_state}")
    if normalized_partition:
        command.append(f"--partition={normalized_partition}")
    if normalized_start:
        command.append(f"--starttime={normalized_start}")
    if normalized_end:
        command.append(f"--endtime={normalized_end}")
    return command


def _run_command(
    settings: Settings,
    argv: list[str],
    *,
    gateway_node: str | None = None,
    fixture_name: str | None = None,
    allow_nonzero: bool = False,
) -> dict[str, Any]:
    fixture_result = _maybe_fixture_result(fixture_name)
    if fixture_result is not None:
        return fixture_result

    resolved_node = resolve_execution_node(settings, str(gateway_node or ""))
    command = _join_argv(argv)
    completed = execute_gateway_command(settings, node=resolved_node, command=command)
    result = {"stdout": completed.stdout, "stderr": completed.stderr, "returncode": int(completed.exit_code)}
    if not allow_nonzero and result["returncode"] != 0:
        raise ToolExecutionError(result["stderr"].strip() or f"SLURM command failed: {argv[0]}")
    return result


def _stream_command(
    settings: Settings,
    argv: list[str],
    *,
    gateway_node: str | None = None,
    fixture_name: str | None = None,
    allow_nonzero: bool = False,
) -> Iterator[dict[str, Any]]:
    fixture_result = _maybe_fixture_result(fixture_name)
    command = _join_argv(argv)
    if fixture_result is not None:
        if fixture_result["stdout"]:
            yield {"type": "stdout", "text": fixture_result["stdout"], "node": str(gateway_node or ""), "command": command}
        if fixture_result["stderr"]:
            yield {"type": "stderr", "text": fixture_result["stderr"], "node": str(gateway_node or ""), "command": command}
        yield {"type": "completed", "exit_code": int(fixture_result["returncode"]), "node": str(gateway_node or ""), "command": command}
        return

    resolved_node = resolve_execution_node(settings, str(gateway_node or ""))
    for chunk in stream_gateway_command(settings, node=resolved_node, command=command):
        payload = chunk.model_dump()
        payload["node"] = resolved_node
        payload["command"] = command
        yield payload

    if allow_nonzero:
        return


def _probe_accounting_fallback(settings: Settings, message: str, *, gateway_node: str | None = None) -> dict[str, Any]:
    try:
        probe = _run_command(
            settings,
            ["sacct", "-X", "-P", "--format=JobID", "--starttime", datetime.now().strftime("%Y-%m-%d")],
            gateway_node=gateway_node,
            fixture_name="sacct_probe.txt",
            allow_nonzero=True,
        )
    except ToolExecutionError as exc:
        result = {
            "available": False,
            "status": "unavailable_or_permission_denied",
            "message": message or str(exc),
            "clusters": [],
        }
        result["text_lines"] = _health_text_lines(result)
        return result

    if probe["returncode"] == 0:
        result = {
            "available": True,
            "status": "sacct_probe_ok",
            "message": "SLURM accounting responded to a read-only sacct probe.",
            "clusters": [],
        }
        result["text_lines"] = _health_text_lines(result)
        return result

    result = {
        "available": False,
        "status": "unavailable_or_permission_denied",
        "message": message or probe["stderr"].strip() or "SLURM accounting probe failed.",
        "clusters": [],
    }
    result["text_lines"] = _health_text_lines(result)
    return result


def _maybe_fixture_result(fixture_name: str | None) -> dict[str, Any] | None:
    fixture_dir = os.getenv(SLURM_FIXTURE_DIR_ENV, "").strip()
    if not fixture_dir or not fixture_name:
        return None
    candidate = Path(fixture_dir) / fixture_name
    if not candidate.exists():
        return None
    return {"stdout": candidate.read_text(), "stderr": "", "returncode": 0}


def _is_fixture_mode() -> bool:
    return bool(str(os.getenv(SLURM_FIXTURE_DIR_ENV, "")).strip())


def _join_argv(argv: list[str]) -> str:
    return shlex.join(argv)


def _raise_for_returncode(returncode: int, stderr: str, binary: str) -> None:
    if int(returncode) != 0:
        raise ToolExecutionError(str(stderr).strip() or f"SLURM command failed: {binary}")


def _filter_jobs(
    jobs: list[dict[str, Any]],
    *,
    user: str | None = None,
    state: str | None = None,
    partition: str | None = None,
) -> list[dict[str, Any]]:
    normalized_state = _canonical_job_state(state) if state else None
    filtered: list[dict[str, Any]] = []
    for job in jobs:
        if user and str(job.get("user", "")) != user:
            continue
        if partition and _normalize_partition_name(str(job.get("partition", ""))) != partition:
            continue
        if normalized_state and _canonical_job_state(str(job.get("state", ""))) != normalized_state:
            continue
        filtered.append(job)
    return filtered


def _filter_nodes(
    nodes: list[dict[str, str]],
    *,
    node: str | None = None,
    partition: str | None = None,
    state: str | None = None,
    state_group: str | None = None,
    gpu_only: bool = False,
) -> list[dict[str, str]]:
    normalized_state = _canonical_node_state(state) if state else None
    normalized_state_group = str(state_group or "").strip().lower() or None
    filtered: list[dict[str, str]] = []
    for entry in nodes:
        if node and str(entry.get("name", "")) != node:
            continue
        if partition and _normalize_partition_name(str(entry.get("partition", ""))) != partition:
            continue
        if normalized_state and _canonical_node_state(str(entry.get("state", ""))) != normalized_state:
            continue
        if normalized_state_group:
            canonical = _canonical_node_state(str(entry.get("state", "")))
            if normalized_state_group == "problematic":
                if not is_problematic_node_state(str(entry.get("state", ""))):
                    continue
            elif normalized_state_group != "all" and canonical != normalized_state_group:
                continue
        if gpu_only and not _node_has_gpu(entry):
            continue
        filtered.append(entry)
    return filtered


def _filter_jobs_by_time(jobs: list[dict[str, Any]], *, start: str | None, end: str | None) -> list[dict[str, Any]]:
    if not start and not end:
        return jobs
    start_dt = _parse_time_value(start) if start else None
    end_dt = _parse_time_value(end) if end else None
    filtered: list[dict[str, Any]] = []
    for job in jobs:
        candidate = _parse_time_value(str(job.get("submit", ""))) or _parse_time_value(str(job.get("start", "")))
        if candidate is None:
            filtered.append(job)
            continue
        if start_dt and candidate < start_dt:
            continue
        if end_dt and candidate > end_dt:
            continue
        filtered.append(job)
    return filtered


def _filter_jobs_by_elapsed(
    jobs: list[dict[str, Any]],
    *,
    min_elapsed_seconds: int | None = None,
    max_elapsed_seconds: int | None = None,
) -> list[dict[str, Any]]:
    if min_elapsed_seconds is None and max_elapsed_seconds is None:
        return jobs
    filtered: list[dict[str, Any]] = []
    for job in jobs:
        elapsed = job.get("elapsed_seconds")
        if elapsed is None:
            elapsed = parse_elapsed_to_seconds(str(job.get("elapsed", "")))
        if elapsed is None:
            continue
        if min_elapsed_seconds is not None and elapsed <= int(min_elapsed_seconds):
            continue
        if max_elapsed_seconds is not None and elapsed >= int(max_elapsed_seconds):
            continue
        filtered.append(job)
    return filtered


def _summarize_field(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    field = "name" if key == "job_name" else key
    for item in items:
        value = str(item.get(field, "")).strip()
        if value:
            counter[value] += 1
    return dict(sorted(counter.items()))


def _metric_text_lines(metric_group: str, payload: dict[str, Any]) -> list[str]:
    if metric_group in {"cluster_summary", "queue_summary", "node_summary", "accounting_summary"}:
        return [f"{key}: {payload[key]}" for key in sorted(payload)]
    if metric_group == "partition_summary":
        partitions = list(payload.get("partitions", []))
        return [
            f"{item.get('partition', '')}: availability={item.get('availability', '')} state={item.get('state', '')} cpus={item.get('cpus', '')} gres={item.get('gres', '')}"
            for item in partitions
        ]
    return [f"{key}: {payload[key]}" for key in sorted(payload)]


def _health_text_lines(payload: dict[str, Any]) -> list[str]:
    lines = [f"available: {str(bool(payload.get('available'))).lower()}", f"status: {payload.get('status', '')}"]
    message = str(payload.get("message", "")).strip()
    if message:
        lines.append(f"message: {message}")
    return lines


def _extract_gpu_count(gres: str) -> int | None:
    normalized = gres.split("(")[0].strip()
    parts = [part for part in normalized.split(":") if part]
    if not parts or parts[0].lower() != "gpu":
        return 0
    if parts[-1].isdigit():
        return int(parts[-1])
    return None


def _node_has_gpu(node: dict[str, str]) -> bool:
    gres = str(node.get("gres", "") or "")
    if not gres or gres in {"(null)", "N/A", "none"}:
        return False
    for part in gres.split(","):
        count = _extract_gpu_count(part)
        if count is None or count > 0:
            return True
    return False


def _canonical_job_state(state: str | None) -> str:
    normalized = str(state or "").strip().upper()
    if not normalized:
        return ""
    if normalized.startswith("RUN"):
        return "RUNNING"
    if normalized.startswith("PEND"):
        return "PENDING"
    if normalized.startswith("COMP"):
        return "COMPLETED"
    if normalized.startswith("FAIL"):
        return "FAILED"
    if normalized.startswith("CANCEL"):
        return "CANCELLED"
    if normalized.startswith("TIMEOUT"):
        return "TIMEOUT"
    return normalized


def _canonical_node_state(state: str | None) -> str:
    normalized = str(state or "").strip().upper()
    if not normalized:
        return ""
    if normalized.startswith("IDLE"):
        return "idle"
    if normalized.startswith("ALLOC"):
        return "allocated"
    if normalized.startswith("MIX"):
        return "mixed"
    if normalized.startswith("DOWN"):
        return "down"
    if normalized.startswith("DRAIN"):
        return "drained"
    if normalized.startswith("FAIL") or normalized.startswith("UNKNOWN") or normalized.startswith("NO_RESPOND"):
        return "down"
    return "other"


def _normalize_partition_name(partition: str) -> str:
    return str(partition or "").strip().rstrip("*")


def _field_rows(fields: dict[str, str]) -> list[dict[str, str]]:
    return [{"field": key, "value": value} for key, value in fields.items()]


def _validate_job_id(job_id: str) -> str:
    normalized = str(job_id or "").strip()
    if not normalized or not JOB_ID_RE.match(normalized):
        raise ToolExecutionError("slurm.job_detail job_id must be a valid SLURM job identifier.")
    return normalized


def _validate_node_name(node: str) -> str:
    normalized = str(node or "").strip()
    if not normalized or not NODE_RE.match(normalized):
        raise ToolExecutionError("slurm node names may contain only letters, numbers, dot, dash, and underscore.")
    return normalized


def _validate_safe_token(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if not SAFE_TOKEN_RE.match(normalized):
        raise ToolExecutionError(f"slurm {field_name} contains unsafe characters.")
    return normalized


def _validate_time_value(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if not DATE_RE.match(normalized):
        raise ToolExecutionError(f"slurm {field_name} must be an ISO-like date or datetime.")
    return normalized


def _validate_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    normalized = int(limit)
    if normalized <= 0:
        raise ToolExecutionError("slurm limit must be greater than zero.")
    return normalized


def _validate_nonnegative_int(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    normalized = int(value)
    if normalized < 0:
        raise ToolExecutionError(f"slurm {field_name} must be zero or greater.")
    return normalized


def _validate_aggregate_metric(metric: str | None) -> str:
    normalized = _validate_safe_token(metric or "average_elapsed", field_name="metric") or "average_elapsed"
    if normalized not in ACCOUNTING_AGGREGATE_METRICS:
        raise ToolExecutionError(f"Unsupported SLURM accounting aggregate metric: {metric}.")
    return normalized


def _validate_aggregate_group_by(group_by: str | None) -> str | None:
    normalized = _validate_safe_token(group_by, field_name="group_by")
    if normalized not in ACCOUNTING_AGGREGATE_GROUP_BY:
        raise ToolExecutionError(f"Unsupported SLURM accounting aggregate group_by: {group_by}.")
    return normalized


def _parse_time_value(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None
