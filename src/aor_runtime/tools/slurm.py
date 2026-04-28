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


def slurm_queue(
    settings: Settings,
    *,
    user: str | None = None,
    state: str | None = None,
    partition: str | None = None,
    limit: int | None = 100,
    gateway_node: str | None = None,
) -> dict[str, Any]:
    normalized_user = _validate_safe_token(user, field_name="user")
    normalized_partition = _validate_safe_token(partition, field_name="partition")
    normalized_state = _validate_safe_token(state, field_name="state")
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
    limit: int | None = 100,
    gateway_node: str | None = None,
) -> dict[str, Any]:
    normalized_user = _validate_safe_token(user, field_name="user")
    normalized_state = _validate_safe_token(state, field_name="state")
    normalized_partition = _validate_safe_token(partition, field_name="partition")
    normalized_start = _validate_time_value(start, field_name="start")
    normalized_end = _validate_time_value(end, field_name="end")
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
        limit=normalized_limit,
    )


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
        problematic_nodes = [node for node in nodes["nodes"] if is_problematic_node_state(str(node.get("state", "")))]
        payload = {
            "queue_count": int(queue["count"]),
            "running_jobs": summarize_jobs_by_state(queue["jobs"]).get("RUNNING", 0),
            "pending_jobs": summarize_jobs_by_state(queue["jobs"]).get("PENDING", 0),
            "node_count": int(nodes["count"]),
            "idle_nodes": int(nodes["summary"]["idle"]),
            "allocated_nodes": int(nodes["summary"]["allocated"]),
            "mixed_nodes": int(nodes["summary"]["mixed"]),
            "down_nodes": int(nodes["summary"]["down"]),
            "drained_nodes": int(nodes["summary"]["drained"]),
            "problematic_nodes": len(problematic_nodes),
            "gpu_available": bool(gpu_summary["available"]),
            "total_gpus": int(gpu_summary["total_gpus"]),
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
        payload = {
            "node_count": int(nodes["count"]),
            "by_state": dict(nodes["summary"]),
            "by_partition": _summarize_field(nodes["nodes"], "partition"),
        }
    elif metric_group == "problematic_nodes":
        nodes = slurm_nodes(settings, state_group="problematic", gateway_node=gateway_node)
        payload = {
            "count": int(nodes["count"]),
            "nodes": nodes["nodes"],
            "by_state": dict(nodes["summary"]),
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


def parse_sacct_output(text: str) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
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
    by_gres: Counter[str] = Counter()
    for node in nodes:
        gres = str(node.get("gres", "") or "")
        if not gres or gres in {"(null)", "N/A", "none"}:
            continue
        gpu_count = 0
        for token in [part.strip() for part in gres.split(",") if part.strip()]:
            cleaned = token.split("(")[0]
            if "gpu" not in cleaned.lower():
                continue
            by_gres[cleaned] += 1
            gpu_count += _extract_gpu_count(cleaned)
        if gpu_count > 0:
            nodes_with_gpu += 1
            total_gpus += gpu_count
    return {
        "available": total_gpus > 0,
        "nodes_with_gpu": nodes_with_gpu,
        "total_gpus": total_gpus,
        "by_gres": dict(sorted(by_gres.items())),
    }


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
        limit: int | None = 100
        gateway_node: str | None = None

    class ToolResult(ToolResultModel):
        jobs: list[dict[str, str]]
        count: int

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
            limit=arguments.limit,
            gateway_node=arguments.gateway_node,
        ) if _is_fixture_mode() else _queue_result_from_stdout(
            stdout,
            user=_validate_safe_token(arguments.user, field_name="user"),
            state=_validate_safe_token(arguments.state, field_name="state"),
            partition=_validate_safe_token(arguments.partition, field_name="partition"),
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
        summary: dict[str, int]

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
        limit: int | None = 100
        gateway_node: str | None = None

    class ToolResult(ToolResultModel):
        jobs: list[dict[str, str]]
        count: int

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
            limit=_validate_limit(arguments.limit),
        )


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
    limit: int | None = None,
) -> dict[str, Any]:
    jobs = parse_squeue_output(stdout)
    filtered = _filter_jobs(jobs, user=user, state=state, partition=partition)
    if limit is not None:
        filtered = filtered[:limit]
    return {"jobs": filtered, "count": len(filtered)}


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
    return {"nodes": filtered, "count": len(filtered), "summary": summarize_node_states(filtered)}


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
    if limit is not None:
        filtered = filtered[:limit]
    return {"jobs": filtered, "count": len(filtered)}


def _accounting_argv(arguments: Any) -> list[str]:
    normalized_user = _validate_safe_token(getattr(arguments, "user", None), field_name="user")
    normalized_state = _validate_safe_token(getattr(arguments, "state", None), field_name="state")
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
    jobs: list[dict[str, str]],
    *,
    user: str | None = None,
    state: str | None = None,
    partition: str | None = None,
) -> list[dict[str, str]]:
    normalized_state = _canonical_job_state(state) if state else None
    filtered: list[dict[str, str]] = []
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


def _filter_jobs_by_time(jobs: list[dict[str, str]], *, start: str | None, end: str | None) -> list[dict[str, str]]:
    if not start and not end:
        return jobs
    start_dt = _parse_time_value(start) if start else None
    end_dt = _parse_time_value(end) if end else None
    filtered: list[dict[str, str]] = []
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
    jobs: list[dict[str, str]],
    *,
    min_elapsed_seconds: int | None = None,
    max_elapsed_seconds: int | None = None,
) -> list[dict[str, str]]:
    if min_elapsed_seconds is None and max_elapsed_seconds is None:
        return jobs
    filtered: list[dict[str, str]] = []
    for job in jobs:
        elapsed = parse_elapsed_to_seconds(str(job.get("elapsed", "")))
        if elapsed is None:
            continue
        if min_elapsed_seconds is not None and elapsed <= int(min_elapsed_seconds):
            continue
        if max_elapsed_seconds is not None and elapsed >= int(max_elapsed_seconds):
            continue
        filtered.append(job)
    return filtered


def _summarize_field(items: list[dict[str, str]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in items:
        value = str(item.get(key, "")).strip()
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


def _extract_gpu_count(gres: str) -> int:
    normalized = gres.split("(")[0].strip()
    if "gpu" not in normalized.lower():
        return 0
    parts = [part for part in normalized.split(":") if part]
    if not parts:
        return 0
    if parts[-1].isdigit():
        return int(parts[-1])
    if any(part.lower() == "gpu" for part in parts):
        return 1
    return 0


def _node_has_gpu(node: dict[str, str]) -> bool:
    gres = str(node.get("gres", "") or "")
    if not gres or gres in {"(null)", "N/A", "none"}:
        return False
    return any("gpu" in part.lower() and _extract_gpu_count(part) > 0 for part in gres.split(","))


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
