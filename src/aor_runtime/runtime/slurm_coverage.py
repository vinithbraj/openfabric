from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aor_runtime.runtime.slurm_semantics import SlurmRequest, SlurmSemanticConstraint, SlurmSemanticFrame


@dataclass
class SlurmCoverageResult:
    passed: bool
    covered_requests: list[str] = field(default_factory=list)
    missing_requests: list[SlurmRequest] = field(default_factory=list)
    covered_constraints: list[str] = field(default_factory=list)
    missing_constraints: list[SlurmSemanticConstraint] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "covered_requests": list(self.covered_requests),
            "missing_requests": [request.to_dict() for request in self.missing_requests],
            "covered_constraints": list(self.covered_constraints),
            "missing_constraints": [constraint.to_dict() for constraint in self.missing_constraints],
            "reason": self.reason,
        }


def validate_slurm_coverage(frame: SlurmSemanticFrame, intent_or_plan: Any) -> SlurmCoverageResult:
    intents = _flatten_intents(intent_or_plan)
    covered_requests: list[str] = []
    missing_requests: list[SlurmRequest] = []
    for request in frame.requests:
        if _request_is_covered(request, intents):
            covered_requests.append(request.id)
        else:
            missing_requests.append(request)

    covered_constraints: list[str] = []
    missing_constraints: list[SlurmSemanticConstraint] = []
    for constraint in frame.constraints:
        if _constraint_is_covered(constraint, intents):
            covered_constraints.append(constraint.id)
        else:
            missing_constraints.append(constraint)

    passed = not missing_requests and not missing_constraints
    reason = "coverage passed" if passed else _coverage_reason(missing_requests, missing_constraints)
    return SlurmCoverageResult(
        passed=passed,
        covered_requests=covered_requests,
        missing_requests=missing_requests,
        covered_constraints=covered_constraints,
        missing_constraints=missing_constraints,
        reason=reason,
    )


def _flatten_intents(intent_or_plan: Any) -> list[Any]:
    if intent_or_plan is None:
        return []
    if isinstance(intent_or_plan, list):
        flattened: list[Any] = []
        for item in intent_or_plan:
            flattened.extend(_flatten_intents(item))
        return flattened
    if type(intent_or_plan).__name__ == "SlurmCompoundIntent":
        return _flatten_intents(getattr(intent_or_plan, "intents", []) or [])
    return [intent_or_plan]


def _request_is_covered(request: SlurmRequest, intents: list[Any]) -> bool:
    if request.kind == "cluster_health":
        groups = {_metric_group(intent) for intent in intents if _metric_group(intent)}
        return "cluster_summary" in groups or {"queue_summary", "node_summary"}.issubset(groups)
    return any(_single_intent_covers_request(request, intent) for intent in intents)


def _single_intent_covers_request(request: SlurmRequest, intent: Any) -> bool:
    name = type(intent).__name__
    filters = request.filters or {}

    if request.kind == "unsupported_mutation":
        return name == "SlurmUnsupportedMutationIntent"

    if request.kind == "cluster_health":
        return _metric_group(intent) == "cluster_summary"

    if request.kind == "queue_status":
        return name == "SlurmQueueIntent" or _metric_group(intent) in {"queue_summary", "scheduler_health", "cluster_summary"}

    if request.kind == "job_count":
        if name != "SlurmJobCountIntent":
            return False
        return _job_filters_match(filters, intent)

    if request.kind == "job_listing":
        if name != "SlurmQueueIntent":
            return False
        return _job_filters_match(filters, intent)

    if request.kind == "job_detail":
        return name == "SlurmJobDetailIntent" and _attr(intent, "job_id") == filters.get("job_id")

    if request.kind == "accounting_jobs":
        if name not in {"SlurmAccountingIntent", "SlurmJobCountIntent"}:
            return False
        if name == "SlurmJobCountIntent" and _attr(intent, "source") != "sacct":
            return False
        return _job_filters_match(filters, intent) and _time_filters_match(filters, intent) and _duration_filters_match(filters, intent)

    if request.kind == "accounting_aggregate":
        if name != "SlurmAccountingAggregateIntent":
            return False
        return (
            _job_filters_match(filters, intent)
            and _time_filters_match(filters, intent)
            and _aggregate_filters_match(filters, intent)
        )

    if request.kind == "node_status":
        if name != "SlurmNodeStatusIntent":
            return False
        return _node_filters_match(filters, intent)

    if request.kind == "problematic_nodes":
        return (
            (name == "SlurmNodeStatusIntent" and _attr(intent, "state_group") == "problematic")
            or _metric_group(intent) == "problematic_nodes"
        )

    if request.kind == "partition_status":
        return name == "SlurmPartitionSummaryIntent" or _metric_group(intent) in {"partition_summary", "cluster_summary"}

    if request.kind == "gpu_availability":
        return _metric_group(intent) == "gpu_summary" or (name == "SlurmNodeStatusIntent" and bool(_attr(intent, "gpu_only")))

    if request.kind == "scheduler_health":
        return _metric_group(intent) in {"scheduler_health", "queue_summary", "cluster_summary"}

    if request.kind in {"slurmdbd_health", "accounting_health"}:
        return name == "SlurmDBDHealthIntent" or _metric_group(intent) in {"slurmdbd_health", "accounting_health"}

    if request.kind == "resource_summary":
        return _metric_group(intent) in {"cluster_summary", "gpu_summary", "node_summary"}

    return False


def _constraint_is_covered(constraint: SlurmSemanticConstraint, intents: list[Any]) -> bool:
    if constraint.kind == "source":
        return any(_source_for_intent(intent) == constraint.value for intent in intents)
    if constraint.kind == "metric":
        return any(_attr(intent, "metric") == constraint.value for intent in intents)
    if constraint.kind == "output_mode":
        return any(str(_attr(intent, "output_mode") or "").lower() == str(constraint.value).lower() for intent in intents)
    if constraint.kind == "job_state":
        return any(str(_attr(intent, "state") or "").upper() == str(constraint.value).upper() for intent in intents)
    if constraint.kind == "job_state_all":
        return any((bool(_attr(intent, "include_all_states")) or _attr(intent, "state") in (None, "")) and _attr(intent, "state") in (None, "") for intent in intents)
    if constraint.kind == "job_state_negation":
        return any((bool(_attr(intent, "include_all_states")) or _attr(intent, "state") in (None, "")) and _attr(intent, "state") in (None, "") for intent in intents)
    if constraint.kind == "job_state_default":
        return any(
            bool(_attr(intent, "default_state_applied"))
            and not bool(_attr(intent, "include_all_states"))
            and str(_attr(intent, "state") or "").upper() == str(constraint.value).upper()
            for intent in intents
        )
    if constraint.kind == "job_user":
        return any(_attr(intent, "user") == constraint.value for intent in intents)
    if constraint.kind in {"job_partition", "partition_filter"}:
        return any(_attr(intent, "partition") == constraint.value for intent in intents)
    if constraint.kind == "job_id":
        return any(_attr(intent, "job_id") == constraint.value for intent in intents)
    if constraint.kind == "node_state":
        return any(_attr(intent, "state_group") == constraint.value or _attr(intent, "state") == constraint.value for intent in intents)
    if constraint.kind == "node_name":
        return any(_attr(intent, "node") == constraint.value for intent in intents)
    if constraint.kind == "time_window":
        return any(_attr(intent, "start") or _attr(intent, "end") for intent in intents)
    if constraint.kind == "duration_comparison":
        return any(
            _attr(intent, "min_elapsed_seconds") == constraint.value
            or _attr(intent, "max_elapsed_seconds") == constraint.value
            or _attr(intent, "threshold_seconds") == constraint.value
            for intent in intents
        )
    if constraint.kind == "group_by":
        return any(_attr(intent, "group_by") == constraint.value or _metric_group(intent) in {"queue_summary", "accounting_summary"} for intent in intents)
    if constraint.kind == "limit":
        return any(_attr(intent, "limit") == constraint.value for intent in intents)
    if constraint.kind in {"gpu_required", "gres_filter"}:
        return any(_metric_group(intent) == "gpu_summary" or bool(_attr(intent, "gpu_only")) for intent in intents)
    return True


def _job_filters_match(filters: dict[str, Any], intent: Any) -> bool:
    for field in ("user", "partition", "state", "group_by"):
        if field in filters and _attr(intent, field) != filters[field]:
            return False
    return True


def _node_filters_match(filters: dict[str, Any], intent: Any) -> bool:
    for field in ("node", "partition", "state", "state_group"):
        if field == "state_group" and field in filters:
            expected = filters[field]
            if _attr(intent, "state_group") == expected or _attr(intent, "state") == expected:
                continue
            return False
        if field in filters and _attr(intent, field) != filters[field]:
            return False
    return True


def _time_filters_match(filters: dict[str, Any], intent: Any) -> bool:
    for field in ("start", "end"):
        if filters.get(field) and _attr(intent, field) != filters[field]:
            return False
    return True


def _duration_filters_match(filters: dict[str, Any], intent: Any) -> bool:
    if filters.get("min_elapsed_seconds") and _attr(intent, "min_elapsed_seconds") != filters["min_elapsed_seconds"]:
        return False
    if filters.get("max_elapsed_seconds") and _attr(intent, "max_elapsed_seconds") != filters["max_elapsed_seconds"]:
        return False
    return True


def _aggregate_filters_match(filters: dict[str, Any], intent: Any) -> bool:
    for field in ("metric", "threshold_seconds"):
        if field in filters and _attr(intent, field) != filters[field]:
            return False
    if bool(filters.get("include_all_states")):
        if not bool(_attr(intent, "include_all_states")) or _attr(intent, "state") not in (None, ""):
            return False
    if bool(filters.get("default_state_applied")):
        if not bool(_attr(intent, "default_state_applied")) or str(_attr(intent, "state") or "").upper() != "COMPLETED":
            return False
    return True


def _metric_group(intent: Any) -> str | None:
    return _attr(intent, "metric_group")


def _attr(intent: Any, field: str) -> Any:
    return getattr(intent, field, None)


def _source_for_intent(intent: Any) -> str:
    name = type(intent).__name__
    if name == "SlurmAccountingAggregateIntent":
        return "sacct"
    if name == "SlurmAccountingIntent":
        return "sacct"
    if name == "SlurmJobCountIntent":
        return str(_attr(intent, "source") or "squeue")
    if name == "SlurmQueueIntent":
        return "squeue"
    if name in {"SlurmNodeStatusIntent", "SlurmPartitionSummaryIntent", "SlurmMetricsIntent"}:
        return "sinfo"
    if name in {"SlurmJobDetailIntent", "SlurmNodeDetailIntent"}:
        return "scontrol"
    if name == "SlurmDBDHealthIntent":
        return "sacctmgr"
    return "unknown"


def _coverage_reason(missing_requests: list[SlurmRequest], missing_constraints: list[SlurmSemanticConstraint]) -> str:
    pieces: list[str] = []
    if missing_requests:
        pieces.append("missing requests: " + ", ".join(request.id for request in missing_requests))
    if missing_constraints:
        pieces.append("missing constraints: " + ", ".join(constraint.id for constraint in missing_constraints))
    return "; ".join(pieces) or "coverage failed"
