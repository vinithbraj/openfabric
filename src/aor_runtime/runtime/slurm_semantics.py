from __future__ import annotations

import getpass
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal


SlurmRequestKind = Literal[
    "queue_status",
    "job_count",
    "job_listing",
    "job_detail",
    "accounting_jobs",
    "node_status",
    "problematic_nodes",
    "partition_status",
    "gpu_availability",
    "cluster_health",
    "scheduler_health",
    "slurmdbd_health",
    "accounting_health",
    "resource_summary",
    "compound",
    "unsupported_mutation",
    "unknown",
]
SlurmConstraintKind = Literal[
    "job_state",
    "job_user",
    "job_partition",
    "job_name",
    "job_id",
    "time_window",
    "duration_comparison",
    "node_state",
    "node_name",
    "gpu_required",
    "gres_filter",
    "partition_filter",
    "group_by",
    "limit",
    "output_mode",
    "unknown",
]
SlurmQueryType = Literal[
    "queue",
    "jobs",
    "accounting",
    "nodes",
    "partitions",
    "metrics",
    "health",
    "compound",
    "unsupported_mutation",
    "unknown",
]


@dataclass
class SlurmRequest:
    id: str
    kind: SlurmRequestKind
    raw_text: str
    filters: dict[str, Any] = field(default_factory=dict)
    output: Literal["count", "rows", "summary", "json"] = "summary"
    requires_tool: (
        Literal[
            "slurm.queue",
            "slurm.accounting",
            "slurm.nodes",
            "slurm.partitions",
            "slurm.metrics",
            "slurm.job_detail",
            "slurm.node_detail",
            "slurm.slurmdbd_health",
            "runtime.return",
        ]
        | None
    ) = None
    confidence: float = 1.0
    covered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SlurmSemanticConstraint:
    id: str
    kind: SlurmConstraintKind
    raw_text: str
    subject: str | None = None
    operator: Literal["eq", "neq", "gt", "gte", "lt", "lte", "between", "contains"] | None = None
    value: Any | None = None
    unit: str | None = None
    resolved_field: str | None = None
    confidence: float = 1.0
    covered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SlurmSemanticFrame:
    query_type: SlurmQueryType
    requests: list[SlurmRequest] = field(default_factory=list)
    constraints: list[SlurmSemanticConstraint] = field(default_factory=list)
    output_mode: Literal["text", "csv", "json", "count"] = "text"
    unresolved_requests: list[SlurmRequest] = field(default_factory=list)
    unresolved_constraints: list[SlurmSemanticConstraint] = field(default_factory=list)
    covered_request_ids: list[str] = field(default_factory=list)
    missing_request_ids: list[str] = field(default_factory=list)
    covered_constraint_ids: list[str] = field(default_factory=list)
    missing_constraint_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type,
            "requests": [request.to_dict() for request in self.requests],
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "output_mode": self.output_mode,
            "unresolved_requests": [request.to_dict() for request in self.unresolved_requests],
            "unresolved_constraints": [constraint.to_dict() for constraint in self.unresolved_constraints],
            "covered_request_ids": list(self.covered_request_ids),
            "missing_request_ids": list(self.missing_request_ids),
            "covered_constraint_ids": list(self.covered_constraint_ids),
            "missing_constraint_ids": list(self.missing_constraint_ids),
        }


_MUTATION_RE = re.compile(
    r"\b(?:sbatch|submit(?:\s+a)?\s+job|scancel|cancel(?:\s+my)?\s+job|drain\s+nodes?|resume\s+nodes?|"
    r"requeue\s+jobs?|suspend\s+jobs?|hold\s+jobs?|release\s+jobs?|scontrol\s+update|update\s+nodes?|"
    r"change\s+partition|delete\s+jobs?|kill(?:\s+my)?\s+jobs?|restart(?:\s+slurm|\s+services?)?|"
    r"systemctl\s+restart|stop\s+service|start\s+service)\b",
    re.IGNORECASE,
)
_SLURM_DOMAIN_RE = re.compile(
    r"\b(?:slurm|squeue|sacct|sinfo|scontrol|slurmdbd|scheduler|accounting|partition|partitions|"
    r"node|nodes|queue|cluster|gpu|gpus|gres|jobs?)\b",
    re.IGNORECASE,
)
_JSON_RE = re.compile(r"\b(?:json|as\s+json|json\s+only)\b", re.IGNORECASE)
_CSV_RE = re.compile(r"\b(?:csv|as\s+csv|csv\s+only|comma\s+separated)\b", re.IGNORECASE)
_COUNT_RE = re.compile(r"\b(?:how\s+many|count|number\s+of)\b", re.IGNORECASE)
_MY_RE = re.compile(r"\bmy\b", re.IGNORECASE)
_USER_RE = re.compile(r"\b(?:for\s+user|user)\s+([A-Za-z0-9._-]+)\b", re.IGNORECASE)
_PARTITION_RE = re.compile(r"\b(?:partition|in\s+partition|on\s+partition)\s+([A-Za-z0-9._-]+)\b", re.IGNORECASE)
_JOB_ID_RE = re.compile(r"\bjob\s+(\d+(?:[._][A-Za-z0-9_-]+)*)\b", re.IGNORECASE)
_GROUP_BY_RE = re.compile(r"\b(?:by|group(?:ed)?\s+by)\s+(state|user|partition|node)\b", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\b(?:top|latest|recent|first|last)\s+(\d+)\b", re.IGNORECASE)
_DURATION_RE = re.compile(r"\b(?:longer\s+than|over|more\s+than)\s+(\d+)\s*(seconds?|minutes?|hours?|days?)\b", re.IGNORECASE)


def extract_slurm_semantic_frame(goal: str, context: dict[str, Any] | None = None) -> SlurmSemanticFrame:
    del context
    prompt = str(goal or "").strip()
    output_mode = _detect_output_mode(prompt)
    if not prompt:
        return SlurmSemanticFrame(query_type="unknown", output_mode=output_mode)

    if _MUTATION_RE.search(prompt):
        operation = _MUTATION_RE.search(prompt)
        request = SlurmRequest(
            id="req_1",
            kind="unsupported_mutation",
            raw_text=str(operation.group(0) if operation else prompt),
            output="summary",
            requires_tool="runtime.return",
        )
        return SlurmSemanticFrame(query_type="unsupported_mutation", requests=[request], output_mode="text")

    if not _looks_like_slurm_prompt(prompt):
        return SlurmSemanticFrame(query_type="unknown", output_mode=output_mode)

    builder = _FrameBuilder(prompt, output_mode)
    builder.add_common_constraints()
    builder.extract_requests()
    return builder.frame()


def _looks_like_slurm_prompt(prompt: str) -> bool:
    lower = prompt.lower()
    if _SLURM_DOMAIN_RE.search(prompt):
        return True
    return any(
        phrase in lower
        for phrase in (
            "what is wrong with the cluster",
            "are gpus available",
            "are gpus free",
            "are jobs stuck",
            "queue pressure",
            "unhealthy things",
        )
    )


class _FrameBuilder:
    def __init__(self, prompt: str, output_mode: Literal["text", "csv", "json", "count"]) -> None:
        self.prompt = prompt
        self.lower = prompt.lower()
        self.output_mode = output_mode
        self.requests: list[SlurmRequest] = []
        self.constraints: list[SlurmSemanticConstraint] = []
        self._seen_requests: set[tuple[str, tuple[tuple[str, str], ...]]] = set()

    def frame(self) -> SlurmSemanticFrame:
        query_type = _query_type_for_requests(self.requests)
        return SlurmSemanticFrame(
            query_type=query_type,
            requests=self.requests,
            constraints=self.constraints,
            output_mode=self.output_mode,
        )

    def add_common_constraints(self) -> None:
        user = _detect_user(self.prompt)
        if user:
            self._constraint("job_user", "my" if user == getpass.getuser() and _MY_RE.search(self.prompt) else user, value=user, resolved_field="user")
        partition = _detect_partition(self.prompt)
        if partition:
            self._constraint("job_partition", partition, value=partition, resolved_field="partition")
            self._constraint("partition_filter", partition, value=partition, resolved_field="partition")
        for state in _detect_job_states(self.prompt):
            self._constraint("job_state", state.lower(), value=state, resolved_field="state")
        node_state = _detect_node_state_group(self.prompt)
        if node_state:
            self._constraint("node_state", node_state, value=node_state, resolved_field="state_group")
        job_id = _detect_job_id(self.prompt)
        if job_id:
            self._constraint("job_id", job_id, value=job_id, resolved_field="job_id")
        start, end = detect_time_range(self.prompt)
        if start or end:
            self._constraint("time_window", "time window", value={"start": start, "end": end}, resolved_field="start,end")
        group_by = _detect_group_by(self.prompt)
        if group_by:
            self._constraint("group_by", f"by {group_by}", value=group_by, resolved_field="group_by")
        limit = _detect_limit(self.prompt)
        if limit is not None:
            self._constraint("limit", f"limit {limit}", value=limit, resolved_field="limit")
        duration = _detect_duration(self.prompt)
        if duration is not None:
            seconds, raw = duration
            self._constraint(
                "duration_comparison",
                raw,
                operator="gt",
                value=seconds,
                unit="seconds",
                resolved_field="elapsed",
            )
        if self.output_mode in {"json", "csv", "count"}:
            self._constraint("output_mode", self.output_mode, value=self.output_mode, resolved_field="output_mode")

    def extract_requests(self) -> None:
        prompt = self.prompt
        lower = self.lower
        output = _request_output(prompt)
        user = _detect_user(prompt)
        partition = _detect_partition(prompt)
        group_by = _detect_group_by(prompt)
        start, end = detect_time_range(prompt)
        limit = _detect_limit(prompt)
        states = _detect_job_states(prompt)
        job_id = _detect_job_id(prompt)
        duration = _detect_duration(prompt)
        job_detail_requested = bool(job_id and re.search(r"\b(?:details?|detail|show)\b", lower))
        node_detail_requested = _mentions_node_detail(lower)

        if _is_broad_cluster_health(lower):
            self._request("cluster_health", "cluster health", output="json" if self.output_mode == "json" else "summary", requires_tool="slurm.metrics")

        if _mentions_scheduler_health(lower):
            self._request("scheduler_health", "scheduler health", output="summary", requires_tool="slurm.metrics")

        if _mentions_slurmdbd_health(lower):
            self._request("slurmdbd_health", "slurmdbd/accounting health", output="summary", requires_tool="slurm.slurmdbd_health")

        if _mentions_accounting_health(lower):
            self._request("accounting_health", "accounting health", output="summary", requires_tool="slurm.slurmdbd_health")

        if _mentions_gpu(lower):
            self._request("gpu_availability", "gpu availability", output="json" if self.output_mode == "json" else "summary", requires_tool="slurm.metrics")

        if _mentions_partitions(lower):
            self._request(
                "partition_status",
                "partition status",
                filters=_clean_filters({"partition": partition}),
                output="json" if self.output_mode == "json" else "rows",
                requires_tool="slurm.partitions",
            )

        if _mentions_problematic_nodes(lower) or _mentions_unhealthy_things(lower):
            self._request("problematic_nodes", "problematic nodes", output="json" if self.output_mode == "json" else "rows", requires_tool="slurm.nodes")

        node_group = _detect_node_state_group(prompt)
        if _mentions_nodes(lower) and not _mentions_problematic_nodes(lower) and not node_detail_requested:
            self._request(
                "node_status",
                "node status",
                filters=_clean_filters({"state_group": node_group, "partition": partition}),
                output="json" if self.output_mode == "json" else "rows",
                requires_tool="slurm.nodes",
            )

        if _mentions_accounting_jobs(lower) or duration is not None:
            state = _first_state(states) or ("FAILED" if "fail" in lower else None)
            elapsed_seconds = duration[0] if duration else None
            self._request(
                "accounting_jobs",
                "accounting jobs",
                filters=_clean_filters(
                    {
                        "user": user,
                        "state": state,
                        "partition": partition,
                        "start": start,
                        "end": end,
                        "min_elapsed_seconds": elapsed_seconds,
                        "limit": limit,
                    }
                ),
                output="count" if output == "count" else ("json" if self.output_mode == "json" else "rows"),
                requires_tool="slurm.accounting",
            )

        if job_id and job_detail_requested:
            self._request(
                "job_detail",
                f"job {job_id}",
                filters={"job_id": job_id},
                output="json" if self.output_mode == "json" else "summary",
                requires_tool="slurm.job_detail",
            )

        queue_status_requested = _mentions_queue_status(lower) or bool(group_by and re.search(r"\b(?:summarize|summary|jobs?\s+by)\b", lower))
        if queue_status_requested:
            self._request("queue_status", "queue status", output="json" if self.output_mode == "json" else "summary", requires_tool="slurm.metrics")

        if _mentions_jobs(lower) and not _mentions_accounting_jobs(lower) and not queue_status_requested and not job_detail_requested:
            requested_states = states or ([None] if not (_COUNT_RE.search(prompt) or re.search(r"\b(?:running|pending)\b", lower)) else [])
            for state in requested_states:
                kind: SlurmRequestKind = "job_count" if output == "count" else "job_listing"
                self._request(
                    kind,
                    f"{state.lower() if state else ''} jobs".strip(),
                    filters=_clean_filters(
                        {
                            "user": user,
                            "state": state,
                            "partition": partition,
                            "group_by": group_by,
                            "limit": limit,
                        }
                    ),
                    output=output,
                    requires_tool="slurm.queue",
                )

        if not self.requests and _mentions_queue(lower):
            self._request(
                "job_listing",
                "queue jobs",
                filters=_clean_filters({"user": user, "partition": partition, "limit": limit}),
                output="json" if self.output_mode == "json" else "rows",
                requires_tool="slurm.queue",
            )

    def _request(
        self,
        kind: SlurmRequestKind,
        raw_text: str,
        *,
        filters: dict[str, Any] | None = None,
        output: Literal["count", "rows", "summary", "json"] = "summary",
        requires_tool: str | None = None,
    ) -> None:
        normalized_filters = _clean_filters(filters or {})
        key = (kind, tuple(sorted((str(k), str(v)) for k, v in normalized_filters.items())))
        if key in self._seen_requests:
            return
        self._seen_requests.add(key)
        self.requests.append(
            SlurmRequest(
                id=f"req_{len(self.requests) + 1}",
                kind=kind,
                raw_text=raw_text,
                filters=normalized_filters,
                output=output,
                requires_tool=requires_tool,
            )
        )

    def _constraint(
        self,
        kind: SlurmConstraintKind,
        raw_text: str,
        *,
        subject: str | None = None,
        operator: Literal["eq", "neq", "gt", "gte", "lt", "lte", "between", "contains"] | None = "eq",
        value: Any | None = None,
        unit: str | None = None,
        resolved_field: str | None = None,
    ) -> None:
        key = (kind, str(value), str(resolved_field))
        for existing in self.constraints:
            if (existing.kind, str(existing.value), str(existing.resolved_field)) == key:
                return
        self.constraints.append(
            SlurmSemanticConstraint(
                id=f"constraint_{len(self.constraints) + 1}",
                kind=kind,
                raw_text=raw_text,
                subject=subject,
                operator=operator,
                value=value,
                unit=unit,
                resolved_field=resolved_field,
            )
        )


def detect_time_range(prompt: str) -> tuple[str | None, str | None]:
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


def _detect_output_mode(prompt: str) -> Literal["text", "csv", "json", "count"]:
    if _JSON_RE.search(prompt):
        return "json"
    if _CSV_RE.search(prompt):
        return "csv"
    if re.search(r"\b(?:count\s+only|number\s+only)\b", prompt, re.IGNORECASE):
        return "count"
    return "text"


def _request_output(prompt: str) -> Literal["count", "rows", "summary", "json"]:
    if _COUNT_RE.search(prompt):
        return "count"
    if _JSON_RE.search(prompt):
        return "json"
    if re.search(r"\b(?:show|list|display|which)\b", prompt, re.IGNORECASE):
        return "rows"
    return "summary"


def _detect_user(prompt: str) -> str | None:
    lower = prompt.lower()
    if _MY_RE.search(prompt) and re.search(r"\b(?:job|jobs|queue|squeue|accounting|sacct)\b", lower):
        return getpass.getuser()
    match = _USER_RE.search(prompt)
    return match.group(1) if match else None


def _detect_partition(prompt: str) -> str | None:
    match = _PARTITION_RE.search(prompt)
    if match is None:
        return None
    candidate = match.group(1)
    if candidate.lower() in {"status", "summary", "cpu", "allocation", "state"}:
        return None
    return candidate


def _detect_job_states(prompt: str) -> list[str]:
    lower = prompt.lower()
    states: list[str] = []
    mapping = [
        ("RUNNING", r"\brunning\b"),
        ("PENDING", r"\b(?:pending|waiting|stuck)\b"),
        ("COMPLETED", r"\bcompleted\b"),
        ("FAILED", r"\b(?:failed|failures?)\b"),
        ("CANCELLED", r"\b(?:cancelled|canceled)\b"),
        ("TIMEOUT", r"\b(?:timed\s+out|timeout)\b"),
    ]
    for state, pattern in mapping:
        if re.search(pattern, lower):
            states.append(state)
    return states


def _first_state(states: list[str]) -> str | None:
    return states[0] if states else None


def _detect_node_state_group(prompt: str) -> str | None:
    lower = prompt.lower()
    if re.search(r"\b(?:problematic|unhealthy|unavailable|not\s+responding|no_respond|down\s+or\s+drained)\b", lower):
        return "problematic"
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


def _detect_job_id(prompt: str) -> str | None:
    match = _JOB_ID_RE.search(prompt)
    return match.group(1) if match else None


def _detect_group_by(prompt: str) -> str | None:
    match = _GROUP_BY_RE.search(prompt)
    return match.group(1).lower() if match else None


def _detect_limit(prompt: str) -> int | None:
    match = _LIMIT_RE.search(prompt)
    if match is None:
        return None
    return int(match.group(1))


def _detect_duration(prompt: str) -> tuple[int, str] | None:
    match = _DURATION_RE.search(prompt)
    if match is None:
        return None
    value = int(match.group(1))
    unit = match.group(2).lower()
    multiplier = 1
    if unit.startswith("minute"):
        multiplier = 60
    elif unit.startswith("hour"):
        multiplier = 3600
    elif unit.startswith("day"):
        multiplier = 86400
    return value * multiplier, match.group(0)


def _is_broad_cluster_health(lower: str) -> bool:
    return (
        ("cluster" in lower or "slurm cluster" in lower)
        and re.search(r"\b(?:status|health|healthy|unhealthy|wrong|summary|summarize|operational|busy)\b", lower) is not None
    ) or re.search(r"\bwhat\s+is\s+wrong\s+with\s+(?:the\s+)?cluster\b", lower) is not None


def _mentions_scheduler_health(lower: str) -> bool:
    return "scheduler" in lower and re.search(r"\b(?:health|healthy|status)\b", lower) is not None


def _mentions_slurmdbd_health(lower: str) -> bool:
    return "slurmdbd" in lower


def _mentions_accounting_health(lower: str) -> bool:
    return "accounting" in lower and re.search(r"\b(?:health|healthy|status|available|availability)\b", lower) is not None


def _mentions_gpu(lower: str) -> bool:
    return re.search(r"\b(?:gpu|gpus|gres)\b", lower) is not None


def _mentions_partitions(lower: str) -> bool:
    return re.search(r"\b(?:partition|partitions)\b", lower) is not None


def _mentions_nodes(lower: str) -> bool:
    return re.search(r"\b(?:node|nodes)\b", lower) is not None


def _mentions_node_detail(lower: str) -> bool:
    return re.search(r"\bnode\s+details?\b|\bdetails?\s+for\s+[-A-Za-z0-9_.]*node", lower) is not None


def _mentions_problematic_nodes(lower: str) -> bool:
    return re.search(r"\b(?:problematic|unhealthy|unavailable|down\s+or\s+drained|down|drained|no_respond|not\s+responding)\s+nodes?\b", lower) is not None


def _mentions_unhealthy_things(lower: str) -> bool:
    return re.search(r"\b(?:anything\s+unhealthy|unhealthy\s+things|what\s+is\s+wrong)\b", lower) is not None


def _mentions_accounting_jobs(lower: str) -> bool:
    return any(token in lower for token in ("sacct", "accounting", "failed", "completed", "cancelled", "canceled", "timed out", "timeout", "yesterday", "recent"))


def _mentions_jobs(lower: str) -> bool:
    return re.search(r"\b(?:job|jobs|queue)\b", lower) is not None


def _mentions_queue(lower: str) -> bool:
    return "queue" in lower or "squeue" in lower


def _mentions_queue_status(lower: str) -> bool:
    return re.search(r"\b(?:(?:summarize|summary\s+of|show)\s+queue|queue\s+(?:status|summary|health|pressure)|scheduler\s+health)\b", lower) is not None


def _query_type_for_requests(requests: list[SlurmRequest]) -> SlurmQueryType:
    if not requests:
        return "unknown"
    kinds = {request.kind for request in requests}
    if "unsupported_mutation" in kinds:
        return "unsupported_mutation"
    if len(requests) > 1 or "cluster_health" in kinds:
        return "compound"
    kind = next(iter(kinds))
    if kind in {"job_count", "job_listing", "job_detail"}:
        return "jobs"
    if kind == "accounting_jobs":
        return "accounting"
    if kind in {"node_status", "problematic_nodes"}:
        return "nodes"
    if kind == "partition_status":
        return "partitions"
    if kind in {"cluster_health", "scheduler_health", "gpu_availability", "resource_summary"}:
        return "metrics"
    if kind in {"slurmdbd_health", "accounting_health"}:
        return "health"
    return "unknown"


def _clean_filters(filters: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in filters.items() if value is not None}
