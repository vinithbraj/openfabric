from __future__ import annotations

import getpass
import re
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel

from aor_runtime.core.contracts import ExecutionPlan
from aor_runtime.runtime.capabilities.base import CapabilityPack, ClassificationContext, CompileContext, CompiledIntentPlan
from aor_runtime.runtime.intents import IntentResult
from aor_runtime.runtime.output_contract import build_output_contract


SLURM_KEYWORD_RE = re.compile(r"\b(?:slurm|squeue|sacct|sinfo|scontrol|slurmdbd|accounting)\b", re.IGNORECASE)
SLURM_MUTATION_RE = re.compile(
    r"\b(?:sbatch|submit(?:\s+a)?\s+job|run\s+this\s+job|scancel|cancel\s+job|drain\s+node|resume\s+node|requeue\s+job|update\s+node|change\s+partition|kill\s+job)\b",
    re.IGNORECASE,
)
OUTPUT_JSON_RE = re.compile(r"\b(?:json|json only|as json)\b", re.IGNORECASE)
OUTPUT_CSV_RE = re.compile(r"\b(?:csv|csv only|as csv|comma separated)\b", re.IGNORECASE)
OUTPUT_COUNT_RE = re.compile(r"\b(?:count only|number only|how many|count)\b", re.IGNORECASE)
USER_FOR_RE = re.compile(r"\bfor\s+user\s+([A-Za-z0-9._-]+)\b", re.IGNORECASE)
USER_RE = re.compile(r"\buser\s+([A-Za-z0-9._-]+)\b", re.IGNORECASE)
PARTITION_RE = re.compile(r"\b(?:partition|in partition|on partition)\s+([A-Za-z0-9._-]+)\b", re.IGNORECASE)
JOB_ID_RE = re.compile(r"\bjob\s+(\d+(?:[._][A-Za-z0-9_-]+)*)\b", re.IGNORECASE)
NODE_DETAIL_RE = re.compile(r"\bnode\s+details?\s+for\s+([A-Za-z0-9._-]+)\b", re.IGNORECASE)
NODE_STATUS_FOR_RE = re.compile(r"\bnode\s+status\s+for\s+([A-Za-z0-9._-]+)\b", re.IGNORECASE)
NODE_NAME_RE = re.compile(r"\bshow\s+(?:slurm\s+)?node\s+([A-Za-z0-9._-]+)\b", re.IGNORECASE)
CLUSTER_ROUTE_RE = re.compile(r"\b(?:on|via|through)\s+cluster\s+([A-Za-z0-9._-]+)\b", re.IGNORECASE)
Bare_ROUTE_RE = re.compile(r"\bon\s+([A-Za-z0-9._-]+)\s*$", re.IGNORECASE)


class SlurmQueueIntent(BaseModel):
    user: str | None = None
    state: str | None = None
    partition: str | None = None
    limit: int | None = 100
    gateway_node: str | None = None
    output_mode: Literal["text", "csv", "json"] = "text"


class SlurmJobDetailIntent(BaseModel):
    job_id: str | None = None
    user: str | None = None
    state: str | None = None
    partition: str | None = None
    limit: int | None = 100
    gateway_node: str | None = None
    output_mode: Literal["text", "csv", "json"] = "text"


class SlurmAccountingIntent(BaseModel):
    user: str | None = None
    state: str | None = None
    partition: str | None = None
    start: str | None = None
    end: str | None = None
    limit: int | None = 100
    gateway_node: str | None = None
    output_mode: Literal["text", "csv", "json"] = "text"


class SlurmJobCountIntent(BaseModel):
    user: str | None = None
    state: str | None = None
    partition: str | None = None
    source: Literal["squeue", "sacct"] = "squeue"
    start: str | None = None
    end: str | None = None
    gateway_node: str | None = None
    output_mode: Literal["count", "json"] = "count"


class SlurmNodeStatusIntent(BaseModel):
    node: str | None = None
    partition: str | None = None
    state: str | None = None
    gateway_node: str | None = None
    output_mode: Literal["text", "csv", "json"] = "text"


class SlurmNodeDetailIntent(BaseModel):
    node: str | None = None
    gateway_node: str | None = None
    output_mode: Literal["text", "csv", "json"] = "text"


class SlurmPartitionSummaryIntent(BaseModel):
    partition: str | None = None
    gateway_node: str | None = None
    output_mode: Literal["text", "csv", "json"] = "text"


class SlurmMetricsIntent(BaseModel):
    metric_group: Literal[
        "cluster_summary",
        "queue_summary",
        "node_summary",
        "partition_summary",
        "gpu_summary",
        "accounting_summary",
        "slurmdbd_health",
    ] = "cluster_summary"
    start: str | None = None
    end: str | None = None
    gateway_node: str | None = None
    output_mode: Literal["text", "json"] = "json"


class SlurmUnsupportedMutationIntent(BaseModel):
    operation: str
    reason: str


class SlurmCapabilityPack(CapabilityPack):
    name = "slurm"
    intent_types = (
        SlurmQueueIntent,
        SlurmJobDetailIntent,
        SlurmAccountingIntent,
        SlurmJobCountIntent,
        SlurmNodeStatusIntent,
        SlurmNodeDetailIntent,
        SlurmPartitionSummaryIntent,
        SlurmMetricsIntent,
        SlurmUnsupportedMutationIntent,
    )

    def classify(self, goal: str, context: ClassificationContext) -> IntentResult:
        prompt = str(goal or "").strip()
        if not prompt:
            return IntentResult(matched=False, reason="slurm_no_match")
        if SLURM_MUTATION_RE.search(prompt) and SLURM_KEYWORD_RE.search(prompt):
            operation = SLURM_MUTATION_RE.search(prompt)
            return IntentResult(
                matched=True,
                intent=SlurmUnsupportedMutationIntent(
                    operation=str(operation.group(0) if operation is not None else "mutation"),
                    reason="This runtime supports read-only SLURM inspection and metrics only.",
                ),
            )
        if not SLURM_KEYWORD_RE.search(prompt):
            return IntentResult(matched=False, reason="slurm_no_match")

        routed_prompt, gateway_node = _extract_gateway_node(prompt, context.settings.available_nodes)
        output_mode = _detect_output_mode(routed_prompt)
        user = _detect_user(routed_prompt)
        partition = _detect_partition(routed_prompt)
        state = _detect_job_state(routed_prompt)
        start, end = _detect_time_range(routed_prompt)

        if re.search(r"\b(?:slurmdbd\s+health|accounting\s+health)\b", routed_prompt, re.IGNORECASE):
            return IntentResult(
                matched=True,
                intent=SlurmMetricsIntent(
                    metric_group="slurmdbd_health",
                    gateway_node=gateway_node,
                    output_mode=_detect_metrics_output_mode(routed_prompt),
                ),
            )

        if re.search(r"\b(?:cluster\s+metrics|cluster\s+summary|dashboard)\b", routed_prompt, re.IGNORECASE):
            return IntentResult(
                matched=True,
                intent=SlurmMetricsIntent(
                    metric_group="cluster_summary",
                    gateway_node=gateway_node,
                    output_mode=_detect_metrics_output_mode(routed_prompt),
                ),
            )

        if re.search(r"\b(?:gpu|gres)\s+(?:availability|summary)\b", routed_prompt, re.IGNORECASE):
            return IntentResult(
                matched=True,
                intent=SlurmMetricsIntent(
                    metric_group="gpu_summary",
                    gateway_node=gateway_node,
                    output_mode=_detect_metrics_output_mode(routed_prompt),
                ),
            )

        if re.search(r"\bsummar(?:ize|y)\s+slurm\s+partition\b|\bpartition\s+cpu\s+allocation\b", routed_prompt, re.IGNORECASE):
            return IntentResult(
                matched=True,
                intent=SlurmPartitionSummaryIntent(partition=partition, gateway_node=gateway_node, output_mode=output_mode),
            )

        if re.search(r"\bsummar(?:ize|y)\s+slurm\s+nodes?\b|\bnode\s+summary\b", routed_prompt, re.IGNORECASE):
            return IntentResult(
                matched=True,
                intent=SlurmMetricsIntent(
                    metric_group="node_summary",
                    gateway_node=gateway_node,
                    output_mode=_detect_metrics_output_mode(routed_prompt),
                ),
            )

        if re.search(r"\bsummar(?:ize|y)\s+(?:jobs|queue)\s+by\s+(?:state|user|partition)\b", routed_prompt, re.IGNORECASE):
            metric_group = "accounting_summary" if _looks_like_accounting_prompt(routed_prompt) else "queue_summary"
            return IntentResult(
                matched=True,
                intent=SlurmMetricsIntent(
                    metric_group=metric_group,
                    start=start,
                    end=end,
                    gateway_node=gateway_node,
                    output_mode=_detect_metrics_output_mode(routed_prompt),
                ),
            )

        node_name = _detect_node_name(routed_prompt)
        if node_name and re.search(r"\bdetails?\b", routed_prompt, re.IGNORECASE):
            return IntentResult(
                matched=True,
                intent=SlurmNodeDetailIntent(node=node_name, gateway_node=gateway_node, output_mode=output_mode),
            )

        job_id_match = JOB_ID_RE.search(routed_prompt)
        if job_id_match and re.search(r"\b(?:details?|show)\b", routed_prompt, re.IGNORECASE):
            return IntentResult(
                matched=True,
                intent=SlurmJobDetailIntent(job_id=job_id_match.group(1), gateway_node=gateway_node, output_mode=output_mode),
            )

        if _looks_like_accounting_prompt(routed_prompt):
            if _detect_count_mode(routed_prompt):
                return IntentResult(
                    matched=True,
                    intent=SlurmJobCountIntent(
                        user=user,
                        state=state,
                        partition=partition,
                        source="sacct",
                        start=start,
                        end=end,
                        gateway_node=gateway_node,
                        output_mode="json" if output_mode == "json" else "count",
                    ),
                )
            return IntentResult(
                matched=True,
                intent=SlurmAccountingIntent(
                    user=user,
                    state=state,
                    partition=partition,
                    start=start,
                    end=end,
                    gateway_node=gateway_node,
                    output_mode=output_mode,
                ),
            )

        if re.search(r"\bpartitions?\b", routed_prompt, re.IGNORECASE):
            return IntentResult(
                matched=True,
                intent=SlurmPartitionSummaryIntent(partition=partition, gateway_node=gateway_node, output_mode=output_mode),
            )

        if re.search(r"\bnodes?\b", routed_prompt, re.IGNORECASE):
            if node_name:
                return IntentResult(
                    matched=True,
                    intent=SlurmNodeStatusIntent(
                        node=node_name,
                        partition=partition,
                        state=_detect_node_state(routed_prompt),
                        gateway_node=gateway_node,
                        output_mode=output_mode,
                    ),
                )
            return IntentResult(
                matched=True,
                intent=SlurmNodeStatusIntent(
                    partition=partition,
                    state=_detect_node_state(routed_prompt),
                    gateway_node=gateway_node,
                    output_mode=output_mode,
                ),
            )

        if _detect_count_mode(routed_prompt):
            return IntentResult(
                matched=True,
                intent=SlurmJobCountIntent(
                    user=user,
                    state=state,
                    partition=partition,
                    source="squeue",
                    gateway_node=gateway_node,
                    output_mode="json" if output_mode == "json" else "count",
                ),
            )

        if job_id_match:
            return IntentResult(
                matched=True,
                intent=SlurmJobDetailIntent(job_id=job_id_match.group(1), gateway_node=gateway_node, output_mode=output_mode),
            )

        if re.search(r"\b(?:queue|jobs?)\b", routed_prompt, re.IGNORECASE):
            return IntentResult(
                matched=True,
                intent=SlurmQueueIntent(
                    user=user,
                    state=state,
                    partition=partition,
                    gateway_node=gateway_node,
                    output_mode=output_mode,
                ),
            )

        return IntentResult(matched=False, reason="slurm_no_match")

    def compile(self, intent: Any, context: CompileContext) -> CompiledIntentPlan | None:
        if not isinstance(intent, self.intent_types):
            return None

        if isinstance(intent, SlurmUnsupportedMutationIntent):
            return CompiledIntentPlan(
                plan=ExecutionPlan.model_validate(
                    {
                        "steps": [
                            {
                                "id": 1,
                                "action": "runtime.return",
                                "args": {
                                    "value": f"{intent.reason} Unsupported operation: {intent.operation}.",
                                    "mode": "text",
                                    "output_contract": build_output_contract(mode="text"),
                                },
                            }
                        ]
                    }
                ),
                metadata={"capability_pack": self.name, "intent_type": type(intent).__name__},
            )

        steps = _compile_slurm_intent(intent, context.allowed_tools)
        return CompiledIntentPlan(
            plan=ExecutionPlan.model_validate({"steps": steps}),
            metadata={"capability_pack": self.name, "intent_type": type(intent).__name__},
        )


def _compile_slurm_intent(intent: Any, allowed_tools: list[str]) -> list[dict[str, Any]]:
    if isinstance(intent, SlurmQueueIntent):
        _require_tools(allowed_tools, "slurm.queue")
        return [
            {
                "id": 1,
                "action": "slurm.queue",
                "args": {
                    "user": intent.user,
                    "state": intent.state,
                    "partition": intent.partition,
                    "limit": intent.limit,
                    "gateway_node": intent.gateway_node,
                },
                "output": "slurm_queue_jobs",
            },
            _rows_return_step(
                step_id=2,
                alias="slurm_queue_jobs",
                collection_path="jobs",
                mode=intent.output_mode,
                wrapper_key="jobs",
            ),
        ]

    if isinstance(intent, SlurmAccountingIntent):
        _require_tools(allowed_tools, "slurm.accounting")
        return [
            {
                "id": 1,
                "action": "slurm.accounting",
                "args": {
                    "user": intent.user,
                    "state": intent.state,
                    "partition": intent.partition,
                    "start": intent.start,
                    "end": intent.end,
                    "limit": intent.limit,
                    "gateway_node": intent.gateway_node,
                },
                "output": "slurm_accounting_jobs",
            },
            _rows_return_step(
                step_id=2,
                alias="slurm_accounting_jobs",
                collection_path="jobs",
                mode=intent.output_mode,
                wrapper_key="jobs",
            ),
        ]

    if isinstance(intent, SlurmJobCountIntent):
        action = "slurm.accounting" if intent.source == "sacct" else "slurm.queue"
        _require_tools(allowed_tools, action)
        args = {
            "user": intent.user,
            "state": intent.state,
            "partition": intent.partition,
            "limit": None,
            "gateway_node": intent.gateway_node,
        }
        if intent.source == "sacct":
            args["start"] = intent.start
            args["end"] = intent.end
        return [
            {"id": 1, "action": action, "args": args, "output": "slurm_job_count"},
            {
                "id": 2,
                "action": "runtime.return",
                "input": ["slurm_job_count"],
                "args": {
                    "value": {"$ref": "slurm_job_count", "path": "count"},
                    "mode": intent.output_mode,
                    "output_contract": build_output_contract(
                        mode=intent.output_mode,
                        json_shape="count" if intent.output_mode == "json" else None,
                    ),
                },
            },
        ]

    if isinstance(intent, SlurmJobDetailIntent):
        if not intent.job_id:
            raise ValueError("Deterministic SLURM job detail requires a job_id.")
        _require_tools(allowed_tools, "slurm.job_detail")
        if intent.output_mode == "text":
            value = {"$ref": "slurm_job_detail", "path": "raw"}
            contract = build_output_contract(mode="text")
        elif intent.output_mode == "json":
            value = {"$ref": "slurm_job_detail"}
            contract = build_output_contract(mode="json")
        else:
            value = {"$ref": "slurm_job_detail", "path": "field_rows"}
            contract = build_output_contract(mode="csv", json_shape="rows")
        return [
            {
                "id": 1,
                "action": "slurm.job_detail",
                "args": {"job_id": intent.job_id, "gateway_node": intent.gateway_node},
                "output": "slurm_job_detail",
            },
            {"id": 2, "action": "runtime.return", "input": ["slurm_job_detail"], "args": {"value": value, "mode": intent.output_mode, "output_contract": contract}},
        ]

    if isinstance(intent, SlurmNodeStatusIntent):
        _require_tools(allowed_tools, "slurm.nodes")
        return [
            {
                "id": 1,
                "action": "slurm.nodes",
                "args": {"node": intent.node, "partition": intent.partition, "state": intent.state, "gateway_node": intent.gateway_node},
                "output": "slurm_nodes",
            },
            _rows_return_step(
                step_id=2,
                alias="slurm_nodes",
                collection_path="nodes",
                mode=intent.output_mode,
                wrapper_key="nodes",
            ),
        ]

    if isinstance(intent, SlurmNodeDetailIntent):
        if not intent.node:
            raise ValueError("Deterministic SLURM node detail requires a node name.")
        _require_tools(allowed_tools, "slurm.node_detail")
        if intent.output_mode == "text":
            value = {"$ref": "slurm_node_detail", "path": "raw"}
            contract = build_output_contract(mode="text")
        elif intent.output_mode == "json":
            value = {"$ref": "slurm_node_detail"}
            contract = build_output_contract(mode="json")
        else:
            value = {"$ref": "slurm_node_detail", "path": "field_rows"}
            contract = build_output_contract(mode="csv", json_shape="rows")
        return [
            {
                "id": 1,
                "action": "slurm.node_detail",
                "args": {"node": intent.node, "gateway_node": intent.gateway_node},
                "output": "slurm_node_detail",
            },
            {"id": 2, "action": "runtime.return", "input": ["slurm_node_detail"], "args": {"value": value, "mode": intent.output_mode, "output_contract": contract}},
        ]

    if isinstance(intent, SlurmPartitionSummaryIntent):
        _require_tools(allowed_tools, "slurm.partitions")
        return [
            {
                "id": 1,
                "action": "slurm.partitions",
                "args": {"partition": intent.partition, "gateway_node": intent.gateway_node},
                "output": "slurm_partitions",
            },
            _rows_return_step(
                step_id=2,
                alias="slurm_partitions",
                collection_path="partitions",
                mode=intent.output_mode,
                wrapper_key="partitions",
            ),
        ]

    if isinstance(intent, SlurmMetricsIntent):
        if intent.metric_group == "slurmdbd_health":
            _require_tools(allowed_tools, "slurm.slurmdbd_health")
            value = {"$ref": "slurm_health"} if intent.output_mode == "json" else {"$ref": "slurm_health", "path": "text_lines"}
            return [
                {"id": 1, "action": "slurm.slurmdbd_health", "args": {"gateway_node": intent.gateway_node}, "output": "slurm_health"},
                {
                    "id": 2,
                    "action": "runtime.return",
                    "input": ["slurm_health"],
                    "args": {
                        "value": value,
                        "mode": intent.output_mode,
                        "output_contract": build_output_contract(mode=intent.output_mode),
                    },
                },
            ]
        _require_tools(allowed_tools, "slurm.metrics")
        value = {"$ref": "slurm_metrics", "path": "payload"} if intent.output_mode == "json" else {"$ref": "slurm_metrics", "path": "text_lines"}
        return [
            {
                "id": 1,
                "action": "slurm.metrics",
                "args": {
                    "metric_group": intent.metric_group,
                    "start": intent.start,
                    "end": intent.end,
                    "gateway_node": intent.gateway_node,
                },
                "output": "slurm_metrics",
            },
            {
                "id": 2,
                "action": "runtime.return",
                "input": ["slurm_metrics"],
                "args": {
                    "value": value,
                    "mode": intent.output_mode,
                    "output_contract": build_output_contract(mode=intent.output_mode),
                },
            },
        ]

    raise ValueError(f"Unsupported SLURM intent: {type(intent).__name__}")


def _rows_return_step(*, step_id: int, alias: str, collection_path: str, mode: str, wrapper_key: str) -> dict[str, Any]:
    if mode == "json":
        value = {wrapper_key: {"$ref": alias, "path": collection_path}}
        contract = build_output_contract(mode="json")
    else:
        value = {"$ref": alias, "path": collection_path}
        contract = build_output_contract(mode=mode, json_shape="rows")
    return {
        "id": step_id,
        "action": "runtime.return",
        "input": [alias],
        "args": {
            "value": value,
            "mode": mode,
            "output_contract": contract,
        },
    }


def _require_tools(allowed_tools: list[str], *required: str) -> None:
    missing = [tool for tool in required if tool not in allowed_tools]
    if missing:
        raise ValueError(f"Deterministic intent requires unavailable tools: {', '.join(missing)}")


def _detect_output_mode(prompt: str) -> Literal["text", "csv", "json"]:
    if OUTPUT_JSON_RE.search(prompt):
        return "json"
    if OUTPUT_CSV_RE.search(prompt):
        return "csv"
    return "text"


def _detect_metrics_output_mode(prompt: str) -> Literal["text", "json"]:
    return "json" if OUTPUT_JSON_RE.search(prompt) else "text"


def _detect_count_mode(prompt: str) -> bool:
    return OUTPUT_COUNT_RE.search(prompt) is not None


def _detect_user(prompt: str) -> str | None:
    if re.search(r"\bmy\b", prompt, re.IGNORECASE):
        return getpass.getuser()
    for pattern in (USER_FOR_RE, USER_RE):
        match = pattern.search(prompt)
        if match is not None:
            return match.group(1)
    return None


def _detect_partition(prompt: str) -> str | None:
    match = PARTITION_RE.search(prompt)
    if match is None:
        return None
    candidate = match.group(1)
    if candidate.lower() in {"status", "summary", "cpu", "allocation"}:
        return None
    return candidate


def _extract_gateway_node(prompt: str, available_nodes: list[str]) -> tuple[str, str | None]:
    explicit_match = CLUSTER_ROUTE_RE.search(prompt)
    if explicit_match is not None:
        gateway_node = explicit_match.group(1)
        cleaned = CLUSTER_ROUTE_RE.sub("", prompt, count=1)
        return _collapse_spaces(cleaned), gateway_node

    trailing_match = Bare_ROUTE_RE.search(prompt)
    if trailing_match is not None:
        gateway_node = trailing_match.group(1)
        if gateway_node in set(available_nodes):
            cleaned = Bare_ROUTE_RE.sub("", prompt, count=1)
            return _collapse_spaces(cleaned), gateway_node

    return prompt, None


def _detect_node_name(prompt: str) -> str | None:
    for pattern in (NODE_DETAIL_RE, NODE_STATUS_FOR_RE, NODE_NAME_RE):
        match = pattern.search(prompt)
        if match is not None:
            candidate = match.group(1)
            if candidate.lower() in {"status", "details", "detail", "node", "nodes"}:
                continue
            return candidate
    return None


def _detect_job_state(prompt: str) -> str | None:
    lower = prompt.lower()
    if "running" in lower:
        return "RUNNING"
    if "pending" in lower:
        return "PENDING"
    if "completed" in lower:
        return "COMPLETED"
    if "failed" in lower:
        return "FAILED"
    if "cancelled" in lower or "canceled" in lower:
        return "CANCELLED"
    if "timed out" in lower or "timeout" in lower:
        return "TIMEOUT"
    return None


def _detect_node_state(prompt: str) -> str | None:
    lower = prompt.lower()
    if "idle" in lower:
        return "idle"
    if "allocated" in lower or re.search(r"\balloc\b", lower):
        return "allocated"
    if "mixed" in lower:
        return "mixed"
    if "drained" in lower or "drain" in lower:
        return "drained"
    if "down" in lower:
        return "down"
    return None


def _looks_like_accounting_prompt(prompt: str) -> bool:
    lower = prompt.lower()
    return any(
        token in lower
        for token in (
            "sacct",
            "accounting",
            "completed",
            "failed",
            "cancelled",
            "canceled",
            "recent",
            "since yesterday",
            "last 24 hours",
            "last 7 days",
            "this week",
        )
    )


def _detect_time_range(prompt: str) -> tuple[str | None, str | None]:
    lower = prompt.lower()
    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)
    if "since yesterday" in lower:
        return yesterday_start.strftime("%Y-%m-%d %H:%M:%S"), None
    if re.search(r"\byesterday\b", lower):
        return yesterday_start.strftime("%Y-%m-%d %H:%M:%S"), today_start.strftime("%Y-%m-%d %H:%M:%S")
    if re.search(r"\btoday\b", lower):
        return today_start.strftime("%Y-%m-%d %H:%M:%S"), None
    if "last 24 hours" in lower:
        return (now - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S"), None
    if "last 7 days" in lower:
        return (today_start - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S"), None
    if "this week" in lower:
        start_of_week = today_start - timedelta(days=today_start.weekday())
        return start_of_week.strftime("%Y-%m-%d %H:%M:%S"), None
    return None, None


def _collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
