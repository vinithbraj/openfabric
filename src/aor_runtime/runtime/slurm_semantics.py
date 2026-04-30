"""OpenFABRIC Runtime Module: aor_runtime.runtime.slurm_semantics

Purpose:
    Extract SLURM semantic intent details used by planners, validators, and fixtures.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

import getpass
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

from aor_runtime.runtime.output_shape import grouped_count_field_for_goal


SlurmRequestKind = Literal[
    "queue_status",
    "job_count",
    "job_listing",
    "job_detail",
    "accounting_jobs",
    "accounting_aggregate",
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
    "source",
    "metric",
    "job_state",
    "job_state_all",
    "job_state_negation",
    "job_state_default",
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
SlurmDataSource = Literal["squeue", "sacct", "sinfo", "scontrol", "sacctmgr", "derived", "unknown"]


@dataclass
class SlurmRequest:
    """Represent slurm request within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SlurmRequest.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.slurm_semantics.SlurmRequest and related tests.
    """
    id: str
    kind: SlurmRequestKind
    raw_text: str
    filters: dict[str, Any] = field(default_factory=dict)
    output: Literal["count", "rows", "summary", "json"] = "summary"
    requires_tool: (
        Literal[
            "slurm.queue",
            "slurm.accounting",
            "slurm.accounting_aggregate",
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
        """To dict for SlurmRequest instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SlurmRequest.to_dict calls and related tests.
        """
        return asdict(self)


@dataclass
class SlurmSemanticConstraint:
    """Represent slurm semantic constraint within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SlurmSemanticConstraint.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.slurm_semantics.SlurmSemanticConstraint and related tests.
    """
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
        """To dict for SlurmSemanticConstraint instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SlurmSemanticConstraint.to_dict calls and related tests.
        """
        return asdict(self)


@dataclass
class SlurmSemanticFrame:
    """Represent slurm semantic frame within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SlurmSemanticFrame.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.slurm_semantics.SlurmSemanticFrame and related tests.
    """
    query_type: SlurmQueryType
    requests: list[SlurmRequest] = field(default_factory=list)
    constraints: list[SlurmSemanticConstraint] = field(default_factory=list)
    output_mode: Literal["text", "csv", "json", "count"] = "text"
    source: SlurmDataSource = "unknown"
    metric: str | None = None
    aggregation: str | None = None
    include_all_states: bool = False
    excluded_states: list[str] = field(default_factory=list)
    explicit_state_filter: str | None = None
    default_state_applied: bool = False
    negated_filters: list[SlurmSemanticConstraint] = field(default_factory=list)
    requested_tools: list[str] = field(default_factory=list)
    compound_requests: list[dict[str, Any]] = field(default_factory=list)
    unresolved_requests: list[SlurmRequest] = field(default_factory=list)
    unresolved_constraints: list[SlurmSemanticConstraint] = field(default_factory=list)
    covered_request_ids: list[str] = field(default_factory=list)
    missing_request_ids: list[str] = field(default_factory=list)
    covered_constraint_ids: list[str] = field(default_factory=list)
    missing_constraint_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """To dict for SlurmSemanticFrame instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SlurmSemanticFrame.to_dict calls and related tests.
        """
        return {
            "query_type": self.query_type,
            "requests": [request.to_dict() for request in self.requests],
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "output_mode": self.output_mode,
            "source": self.source,
            "metric": self.metric,
            "aggregation": self.aggregation,
            "include_all_states": self.include_all_states,
            "excluded_states": list(self.excluded_states),
            "explicit_state_filter": self.explicit_state_filter,
            "default_state_applied": self.default_state_applied,
            "negated_filters": [constraint.to_dict() for constraint in self.negated_filters],
            "requested_tools": list(self.requested_tools),
            "compound_requests": list(self.compound_requests),
            "unresolved_requests": [request.to_dict() for request in self.unresolved_requests],
            "unresolved_constraints": [constraint.to_dict() for constraint in self.unresolved_constraints],
            "covered_request_ids": list(self.covered_request_ids),
            "missing_request_ids": list(self.missing_request_ids),
            "covered_constraint_ids": list(self.covered_constraint_ids),
            "missing_constraint_ids": list(self.missing_constraint_ids),
        }


_MUTATION_RE = re.compile(
    r"\b(?:sbatch|submit(?:\s+a)?\s+job|scancel|cancel(?:\s+(?:all|my|the|pending|running))*\s+jobs?|drain\s+nodes?|resume\s+nodes?|"
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
_GROUP_BY_RE = re.compile(r"\b(?:by|group(?:ed)?\s+by)\s+(state|user|partition|node|job\s+name)\b", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\b(?:top|latest|recent|first|last)\s+(\d+)\b", re.IGNORECASE)
_DURATION_RE = re.compile(r"\b(?:longer\s+than|over|more\s+than)\s+(\d+)\s*(seconds?|minutes?|hours?|days?)\b", re.IGNORECASE)
_RUNTIME_AGGREGATE_RE = re.compile(
    r"\b(?:how\s+long|average\s+(?:run\s*time|runtime|elapsed|duration)|avg\s+elapsed|mean\s+runtime|"
    r"min(?:imum)?\s+(?:run\s*time|runtime|elapsed|duration)|max(?:imum)?\s+(?:run\s*time|runtime|elapsed|duration)|"
    r"longest\s+(?:run\s*time|runtime|elapsed|duration)|total\s+(?:run\s*time|runtime|elapsed|duration)|"
    r"runtime\s+summary|elapsed\s+time|took\s+to\s+run|ran\s+longer\s+than|jobs?\s+longer\s+than)\b",
    re.IGNORECASE,
)


def extract_slurm_semantic_frame(goal: str, context: dict[str, Any] | None = None) -> SlurmSemanticFrame:
    """Extract slurm semantic frame for the surrounding runtime workflow.

    Inputs:
        Receives goal, context for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics.extract_slurm_semantic_frame.
    """
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
    """Handle the internal looks like slurm prompt helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._looks_like_slurm_prompt.
    """
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
    """Represent frame builder within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by _FrameBuilder.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.slurm_semantics._FrameBuilder and related tests.
    """
    def __init__(self, prompt: str, output_mode: Literal["text", "csv", "json", "count"]) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives prompt, output_mode for this _FrameBuilder method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by planning, execution, validation, and presentation through _FrameBuilder.__init__ calls and related tests.
        """
        self.prompt = prompt
        self.lower = prompt.lower()
        self.output_mode = output_mode
        self.requests: list[SlurmRequest] = []
        self.constraints: list[SlurmSemanticConstraint] = []
        self._seen_requests: set[tuple[str, tuple[tuple[str, str], ...]]] = set()

    def frame(self) -> SlurmSemanticFrame:
        """Frame for _FrameBuilder instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through _FrameBuilder.frame calls and related tests.
        """
        query_type = _query_type_for_requests(self.requests)
        requested_tools = [
            str(request.requires_tool)
            for request in self.requests
            if request.requires_tool and request.requires_tool != "runtime.return"
        ]
        metric = _first_filter_value(self.requests, "metric")
        explicit_state = _first_filter_value(self.requests, "state")
        include_all_states = any(bool((request.filters or {}).get("include_all_states")) for request in self.requests)
        default_state_applied = any(bool((request.filters or {}).get("default_state_applied")) for request in self.requests)
        negated_filters = [constraint for constraint in self.constraints if constraint.kind == "job_state_negation"]
        frame = SlurmSemanticFrame(
            query_type=query_type,
            requests=self.requests,
            constraints=self.constraints,
            output_mode=self.output_mode,
            metric=str(metric) if metric else None,
            aggregation=str(metric) if metric else None,
            include_all_states=include_all_states,
            explicit_state_filter=str(explicit_state) if explicit_state else None,
            default_state_applied=default_state_applied,
            negated_filters=negated_filters,
            requested_tools=requested_tools,
            compound_requests=[request.to_dict() for request in self.requests] if query_type == "compound" else [],
        )
        frame.source = select_slurm_data_source(frame)
        return frame

    def add_common_constraints(self) -> None:
        """Add common constraints for _FrameBuilder instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through _FrameBuilder.add_common_constraints calls and related tests.
        """
        user = _detect_user(self.prompt)
        if user:
            self._constraint("job_user", "my" if user == getpass.getuser() and _MY_RE.search(self.prompt) else user, value=user, resolved_field="user")
        partition = _detect_partition(self.prompt) or (_detect_runtime_partition(self.prompt) if _mentions_runtime_aggregate(self.lower) else None)
        if partition:
            self._constraint("job_partition", partition, value=partition, resolved_field="partition")
            self._constraint("partition_filter", partition, value=partition, resolved_field="partition")
        runtime_aggregate = _detect_runtime_aggregate(self.prompt)
        if runtime_aggregate is not None:
            self._constraint("source", "accounting source", value="sacct", resolved_field="source")
            self._constraint("metric", str(runtime_aggregate["metric"]), value=runtime_aggregate["metric"], resolved_field="metric")
        if _runtime_all_states(self.prompt):
            raw = _runtime_all_states_raw(self.prompt) or "all states"
            self._constraint("job_state_all", raw, value=True, resolved_field="include_all_states")
            for state in _detect_negated_state_filters(self.prompt):
                self._constraint(
                    "job_state_negation",
                    raw,
                    operator="neq",
                    value=state,
                    resolved_field="state",
                )
        else:
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
        """Extract requests for _FrameBuilder instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through _FrameBuilder.extract_requests calls and related tests.
        """
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
        runtime_aggregate = _detect_runtime_aggregate(prompt)
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

        if _mentions_gpu(lower) and runtime_aggregate is None:
            self._request("gpu_availability", "gpu availability", output="json" if self.output_mode == "json" else "summary", requires_tool="slurm.metrics")

        state_summary_requested = bool(
            group_by == "state"
            and _mentions_jobs(lower)
            and re.search(r"\b(?:summarize|summary)\b", lower)
        )
        grouped_jobs_requested = bool(group_by and _mentions_jobs(lower) and not state_summary_requested)

        if _mentions_partitions(lower) and not grouped_jobs_requested and runtime_aggregate is None:
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

        if runtime_aggregate is not None:
            include_all_states = _runtime_all_states(prompt)
            explicit_state = runtime_aggregate["state"] or _first_state(states)
            default_state_applied = False
            if include_all_states:
                state = None
            elif explicit_state:
                state = explicit_state
            else:
                state = "COMPLETED"
                default_state_applied = True
                self._constraint(
                    "job_state_default",
                    "default completed jobs",
                    value="COMPLETED",
                    resolved_field="state",
                )
            self._request(
                "accounting_aggregate",
                "SLURM accounting runtime",
                filters=_clean_filters(
                    {
                        "user": user,
                        "state": state,
                        "include_all_states": True if include_all_states else None,
                        "excluded_states": [],
                        "default_state_applied": True if default_state_applied else None,
                        "partition": runtime_aggregate["partition"] or partition,
                        "start": runtime_aggregate["start"],
                        "end": runtime_aggregate["end"],
                        "metric": runtime_aggregate["metric"],
                        "group_by": runtime_aggregate["group_by"] or group_by,
                        "threshold_seconds": runtime_aggregate["threshold_seconds"],
                        "limit": limit or 1000,
                        "time_window_label": runtime_aggregate["time_window_label"],
                    }
                ),
                output="json" if self.output_mode == "json" else "summary",
                requires_tool="slurm.accounting_aggregate",
            )

        if runtime_aggregate is None and (_mentions_accounting_jobs(lower) or duration is not None):
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

        queue_status_requested = _mentions_queue_status(lower) or bool(
            group_by and not grouped_jobs_requested and re.search(r"\b(?:summarize|summary)\b", lower)
        )
        if queue_status_requested:
            self._request("queue_status", "queue status", output="json" if self.output_mode == "json" else "summary", requires_tool="slurm.metrics")

        if _mentions_jobs(lower) and runtime_aggregate is None and not (_mentions_accounting_jobs(lower) or duration is not None) and not queue_status_requested and not job_detail_requested:
            requested_states = states or ([None] if (group_by or not (_COUNT_RE.search(prompt) or re.search(r"\b(?:running|pending)\b", lower))) else [])
            for state in requested_states:
                kind: SlurmRequestKind = "job_count" if output == "count" or group_by else "job_listing"
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
                    output="count" if group_by else output,
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
        """Handle the internal request helper path for this module.

        Inputs:
            Receives kind, raw_text, filters, output, requires_tool for this _FrameBuilder method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through _FrameBuilder._request calls and related tests.
        """
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
        """Handle the internal constraint helper path for this module.

        Inputs:
            Receives kind, raw_text, subject, operator, value, unit, resolved_field for this _FrameBuilder method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through _FrameBuilder._constraint calls and related tests.
        """
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
    """Detect time range for the surrounding runtime workflow.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics.detect_time_range.
    """
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
    """Handle the internal detect output mode helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._detect_output_mode.
    """
    if _JSON_RE.search(prompt):
        return "json"
    if _CSV_RE.search(prompt):
        return "csv"
    if re.search(r"\b(?:count\s+only|number\s+only)\b", prompt, re.IGNORECASE):
        return "count"
    return "text"


def _request_output(prompt: str) -> Literal["count", "rows", "summary", "json"]:
    """Handle the internal request output helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._request_output.
    """
    if _COUNT_RE.search(prompt):
        return "count"
    if _JSON_RE.search(prompt):
        return "json"
    if re.search(r"\b(?:show|list|display|which)\b", prompt, re.IGNORECASE):
        return "rows"
    return "summary"


def _detect_user(prompt: str) -> str | None:
    """Handle the internal detect user helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._detect_user.
    """
    lower = prompt.lower()
    if _MY_RE.search(prompt) and re.search(r"\b(?:job|jobs|queue|squeue|accounting|sacct)\b", lower):
        return getpass.getuser()
    match = _USER_RE.search(prompt)
    return match.group(1) if match else None


def _detect_partition(prompt: str) -> str | None:
    """Handle the internal detect partition helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._detect_partition.
    """
    match = _PARTITION_RE.search(prompt)
    if match is None:
        return None
    candidate = match.group(1)
    if candidate.lower() in {"status", "summary", "cpu", "allocation", "state", "in", "on", "by", "as", "json", "csv", "only"}:
        return None
    return candidate


def _detect_job_states(prompt: str) -> list[str]:
    """Handle the internal detect job states helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._detect_job_states.
    """
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
    """Handle the internal first state helper path for this module.

    Inputs:
        Receives states for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._first_state.
    """
    return states[0] if states else None


def _detect_node_state_group(prompt: str) -> str | None:
    """Handle the internal detect node state group helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._detect_node_state_group.
    """
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
    """Handle the internal detect job id helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._detect_job_id.
    """
    match = _JOB_ID_RE.search(prompt)
    return match.group(1) if match else None


def _detect_group_by(prompt: str) -> str | None:
    """Handle the internal detect group by helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._detect_group_by.
    """
    match = _GROUP_BY_RE.search(prompt)
    if match:
        return match.group(1).lower().replace(" ", "_")
    return grouped_count_field_for_goal(prompt)


def _detect_limit(prompt: str) -> int | None:
    """Handle the internal detect limit helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._detect_limit.
    """
    match = _LIMIT_RE.search(prompt)
    if match is None:
        return None
    tail = prompt[match.end() : match.end() + 12].lower()
    if re.match(r"\s*(?:seconds?|minutes?|hours?|days?|weeks?)\b", tail):
        return None
    return int(match.group(1))


def _detect_duration(prompt: str) -> tuple[int, str] | None:
    """Handle the internal detect duration helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._detect_duration.
    """
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


def _mentions_runtime_aggregate(lower: str) -> bool:
    """Handle the internal mentions runtime aggregate helper path for this module.

    Inputs:
        Receives lower for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._mentions_runtime_aggregate.
    """
    return _RUNTIME_AGGREGATE_RE.search(lower) is not None or (
        "runtime" in lower and re.search(r"\b(?:average|avg|mean|min|max|longest|total|summary|by|longer\s+than)\b", lower)
        is not None
    )


def _detect_runtime_aggregate(prompt: str) -> dict[str, Any] | None:
    """Handle the internal detect runtime aggregate helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._detect_runtime_aggregate.
    """
    lower = prompt.lower()
    if not _mentions_runtime_aggregate(lower):
        return None
    if re.search(r"\b(?:show|list|display)\b", lower) and not re.search(
        r"\b(?:average|avg|mean|min|minimum|max|maximum|longest|total|sum|summary|count|how\s+many|longer\s+than|how\s+long|took\s+to\s+run)\b",
        lower,
    ):
        return None
    if "median" in lower:
        return {
            "metric": "unsupported_median",
            "state": None,
            "partition": _detect_runtime_partition(prompt),
            "start": None,
            "end": None,
            "group_by": _detect_group_by(prompt),
            "threshold_seconds": None,
            "time_window_label": None,
        }
    duration = _detect_duration(prompt)
    metric = _detect_runtime_metric(lower, duration)
    start, end = detect_time_range(prompt)
    time_window_label = _time_window_label(prompt, start, end)
    if start is None and end is None:
        start = _default_accounting_start()
        time_window_label = "Last 7 days"
    state = _detect_runtime_state(prompt)
    return {
        "metric": metric,
        "state": state,
        "partition": _detect_runtime_partition(prompt),
        "start": start,
        "end": end,
        "group_by": _detect_group_by(prompt),
        "threshold_seconds": duration[0] if duration else None,
        "time_window_label": time_window_label,
    }


def _detect_runtime_metric(lower: str, duration: tuple[int, str] | None) -> str:
    """Handle the internal detect runtime metric helper path for this module.

    Inputs:
        Receives lower, duration for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._detect_runtime_metric.
    """
    if duration is not None and re.search(r"\b(?:count|how\s+many)\b", lower):
        return "count_longer_than"
    if re.search(r"\bruntime\s+summary\b|\bmin\s*/\s*max\s*/\s*average\b", lower):
        return "runtime_summary"
    if re.search(r"\b(?:min|minimum|shortest)\b", lower):
        return "min_elapsed"
    if re.search(r"\b(?:max|maximum|longest)\b", lower):
        return "max_elapsed"
    if re.search(r"\b(?:sum|total)\s+(?:run\s*time|runtime|elapsed|duration)\b", lower):
        return "sum_elapsed"
    if re.search(r"\b(?:count|how\s+many)\b", lower):
        return "count"
    return "average_elapsed"


def _detect_runtime_state(prompt: str) -> str | None:
    """Handle the internal detect runtime state helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._detect_runtime_state.
    """
    lower = prompt.lower()
    if _runtime_all_states(prompt):
        return None
    states = _detect_job_states(prompt)
    return _first_state(states)


def _runtime_all_states(prompt: str) -> bool:
    """Handle the internal runtime all states helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._runtime_all_states.
    """
    return _runtime_all_states_raw(prompt) is not None


def _runtime_all_states_raw(prompt: str) -> str | None:
    """Handle the internal runtime all states raw helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._runtime_all_states_raw.
    """
    patterns = [
        r"\bdo\s+not\s+filter\s+by\s+completed\b",
        r"\bdon't\s+filter\s+by\s+completed\b",
        r"\bdo\s+not\s+restrict\s+to\s+completed\b",
        r"\bdon't\s+restrict\s+to\s+completed\b",
        r"\bnot\s+just\s+completed\b",
        r"\ball\s+jobs?\b",
        r"\bany\s+state\b",
        r"\ball\s+job\s+states\b",
        r"\bacross\s+all\s+states\b",
        r"\bregardless\s+of\s+state\b",
        r"\bget\s+all\s+jobs?\b",
        r"\binclude\s+all\s+job\s+states\b",
        r"\binclude\s+failed\s+jobs?\s+too\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def _detect_negated_state_filters(prompt: str) -> list[str]:
    """Handle the internal detect negated state filters helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._detect_negated_state_filters.
    """
    if re.search(
        r"\b(?:do\s+not|don't)\s+(?:filter\s+by|restrict\s+to)\s+completed\b|\bnot\s+just\s+completed\b",
        prompt,
        re.IGNORECASE,
    ):
        return ["COMPLETED"]
    return []


def _detect_runtime_partition(prompt: str) -> str | None:
    """Handle the internal detect runtime partition helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._detect_runtime_partition.
    """
    patterns = [
        r"\b(?:on|in)\s+([A-Za-z0-9._-]+)\s+partition\b",
        r"\b([A-Za-z0-9._-]+)\s+partition\b",
        r"\b(?:on|in)\s+([A-Za-z0-9._-]+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match is None:
            continue
        candidate = match.group(1).strip()
        if candidate.lower() in {
            "a",
            "an",
            "the",
            "all",
            "completed",
            "failed",
            "cancelled",
            "canceled",
            "timed",
            "jobs",
            "job",
            "last",
            "by",
            "as",
            "json",
            "csv",
            "only",
            "today",
            "yesterday",
            "partition",
            "partitions",
            "slurm",
            "cluster",
            "accounting",
            "sacct",
        }:
            continue
        return candidate
    explicit = _detect_partition(prompt)
    if explicit:
        return explicit
    return None


def _default_accounting_start() -> str:
    """Handle the internal default accounting start helper path for this module.

    Inputs:
        Uses module or instance state; no caller-supplied data parameters are required.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._default_accounting_start.
    """
    return (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")


def _time_window_label(prompt: str, start: str | None, end: str | None) -> str | None:
    """Handle the internal time window label helper path for this module.

    Inputs:
        Receives prompt, start, end for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._time_window_label.
    """
    lower = prompt.lower()
    if "last 7 days" in lower:
        return "Last 7 days"
    if "last 24 hours" in lower:
        return "Last 24 hours"
    if "since yesterday" in lower:
        return "Since yesterday"
    if "yesterday" in lower:
        return "Yesterday"
    if "today" in lower:
        return "Today"
    if "this week" in lower:
        return "This week"
    if start or end:
        return "Requested time window"
    return None


def _is_broad_cluster_health(lower: str) -> bool:
    """Handle the internal is broad cluster health helper path for this module.

    Inputs:
        Receives lower for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._is_broad_cluster_health.
    """
    return (
        ("cluster" in lower or "slurm cluster" in lower)
        and re.search(r"\b(?:status|health|healthy|unhealthy|wrong|summary|summarize|operational|busy)\b", lower) is not None
    ) or re.search(r"\bwhat\s+is\s+wrong\s+with\s+(?:the\s+)?cluster\b", lower) is not None


def _mentions_scheduler_health(lower: str) -> bool:
    """Handle the internal mentions scheduler health helper path for this module.

    Inputs:
        Receives lower for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._mentions_scheduler_health.
    """
    return "scheduler" in lower and re.search(r"\b(?:health|healthy|status)\b", lower) is not None


def _mentions_slurmdbd_health(lower: str) -> bool:
    """Handle the internal mentions slurmdbd health helper path for this module.

    Inputs:
        Receives lower for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._mentions_slurmdbd_health.
    """
    return "slurmdbd" in lower


def _mentions_accounting_health(lower: str) -> bool:
    """Handle the internal mentions accounting health helper path for this module.

    Inputs:
        Receives lower for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._mentions_accounting_health.
    """
    return "accounting" in lower and re.search(r"\b(?:health|healthy|status|available|availability)\b", lower) is not None


def _mentions_gpu(lower: str) -> bool:
    """Handle the internal mentions gpu helper path for this module.

    Inputs:
        Receives lower for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._mentions_gpu.
    """
    return re.search(r"\b(?:gpu|gpus|gres)\b", lower) is not None


def _mentions_partitions(lower: str) -> bool:
    """Handle the internal mentions partitions helper path for this module.

    Inputs:
        Receives lower for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._mentions_partitions.
    """
    return re.search(r"\b(?:partition|partitions)\b", lower) is not None


def _mentions_nodes(lower: str) -> bool:
    """Handle the internal mentions nodes helper path for this module.

    Inputs:
        Receives lower for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._mentions_nodes.
    """
    return re.search(r"\b(?:node|nodes)\b", lower) is not None


def _mentions_node_detail(lower: str) -> bool:
    """Handle the internal mentions node detail helper path for this module.

    Inputs:
        Receives lower for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._mentions_node_detail.
    """
    return re.search(r"\bnode\s+details?\b|\bdetails?\s+for\s+[-A-Za-z0-9_.]*node", lower) is not None


def _mentions_problematic_nodes(lower: str) -> bool:
    """Handle the internal mentions problematic nodes helper path for this module.

    Inputs:
        Receives lower for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._mentions_problematic_nodes.
    """
    return re.search(r"\b(?:problematic|unhealthy|unavailable|down\s+or\s+drained|down|drained|no_respond|not\s+responding)\s+nodes?\b", lower) is not None


def _mentions_unhealthy_things(lower: str) -> bool:
    """Handle the internal mentions unhealthy things helper path for this module.

    Inputs:
        Receives lower for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._mentions_unhealthy_things.
    """
    return re.search(r"\b(?:anything\s+unhealthy|unhealthy\s+things|what\s+is\s+wrong)\b", lower) is not None


def _mentions_accounting_jobs(lower: str) -> bool:
    """Handle the internal mentions accounting jobs helper path for this module.

    Inputs:
        Receives lower for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._mentions_accounting_jobs.
    """
    return any(
        token in lower
        for token in (
            "sacct",
            "accounting",
            "failed",
            "completed",
            "cancelled",
            "canceled",
            "timed out",
            "timeout",
            "yesterday",
            "recent",
            "elapsed time",
        )
    )


def _mentions_jobs(lower: str) -> bool:
    """Handle the internal mentions jobs helper path for this module.

    Inputs:
        Receives lower for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._mentions_jobs.
    """
    return re.search(r"\b(?:job|jobs|queue)\b", lower) is not None


def _mentions_queue(lower: str) -> bool:
    """Handle the internal mentions queue helper path for this module.

    Inputs:
        Receives lower for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._mentions_queue.
    """
    return "queue" in lower or "squeue" in lower


def _mentions_queue_status(lower: str) -> bool:
    """Handle the internal mentions queue status helper path for this module.

    Inputs:
        Receives lower for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._mentions_queue_status.
    """
    return re.search(r"\b(?:(?:summarize|summary\s+of|show)\s+queue|queue\s+(?:status|summary|health|pressure)|scheduler\s+health)\b", lower) is not None


def _query_type_for_requests(requests: list[SlurmRequest]) -> SlurmQueryType:
    """Handle the internal query type for requests helper path for this module.

    Inputs:
        Receives requests for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._query_type_for_requests.
    """
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
    if kind in {"accounting_jobs", "accounting_aggregate"}:
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


def select_slurm_data_source(frame: SlurmSemanticFrame) -> SlurmDataSource:
    """Select slurm data source for the surrounding runtime workflow.

    Inputs:
        Receives frame for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics.select_slurm_data_source.
    """
    if frame.query_type == "unsupported_mutation":
        return "unknown"
    if len(frame.requests) > 1 or frame.query_type == "compound":
        return "derived"
    if not frame.requests:
        return "unknown"
    request = frame.requests[0]
    if request.kind == "accounting_aggregate":
        return "sacct"
    if request.kind == "accounting_jobs":
        return "sacct"
    if request.kind in {"job_count", "job_listing"}:
        return "sacct" if request.filters.get("source") == "sacct" else "squeue"
    if request.kind == "job_detail":
        return "scontrol"
    if request.kind in {"node_status", "problematic_nodes", "partition_status", "gpu_availability", "resource_summary"}:
        return "sinfo"
    if request.kind in {"slurmdbd_health", "accounting_health"}:
        return "sacctmgr"
    if request.kind in {"cluster_health", "scheduler_health", "queue_status"}:
        return "derived" if request.kind == "cluster_health" else "squeue"
    return "unknown"


def _first_filter_value(requests: list[SlurmRequest], key: str) -> Any | None:
    """Handle the internal first filter value helper path for this module.

    Inputs:
        Receives requests, key for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._first_filter_value.
    """
    for request in requests:
        if key in request.filters:
            return request.filters[key]
    return None


def _clean_filters(filters: dict[str, Any]) -> dict[str, Any]:
    """Handle the internal clean filters helper path for this module.

    Inputs:
        Receives filters for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_semantics._clean_filters.
    """
    return {key: value for key, value in filters.items() if value is not None}
