from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any

from aor_runtime.config import Settings
from aor_runtime.core.contracts import StepLog
from aor_runtime.runtime.slurm_aggregations import aggregate_slurm_accounting_jobs
from aor_runtime.tools.slurm import slurm_accounting


@dataclass(frozen=True)
class SemanticObligation:
    domain: str
    field: str
    value: str
    strategy: str
    source: str


@dataclass(frozen=True)
class FilterStrategySpec:
    tool: str
    reliable_pushdowns: tuple[str, ...] = ()
    tentative_pushdowns: tuple[str, ...] = ()
    local_filterable_fields: tuple[str, ...] = ()


@dataclass
class SemanticCanonicalizationResult:
    repairs: list[str] = field(default_factory=list)
    obligations: list[SemanticObligation] = field(default_factory=list)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "obligations": [asdict(obligation) for obligation in self.obligations],
            "repairs": list(self.repairs),
        }


@dataclass
class SemanticFallbackResult:
    log: StepLog
    applied: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


SLURM_ACCOUNTING_AGGREGATE_STRATEGY = FilterStrategySpec(
    tool="slurm.accounting_aggregate",
    reliable_pushdowns=("partition", "user", "start", "end"),
    tentative_pushdowns=("state",),
    local_filterable_fields=("state",),
)


def apply_semantic_obligations_to_actions(goal: str, actions: list[dict[str, Any]]) -> SemanticCanonicalizationResult:
    result = SemanticCanonicalizationResult(obligations=extract_semantic_obligations(goal))
    if not result.obligations:
        return result

    desired_state = _obligation_value(result.obligations, domain="slurm", field="state")
    all_states = _obligation_value(result.obligations, domain="slurm", field="state_mode") == "all"
    desired_partition = _obligation_value(result.obligations, domain="slurm", field="partition")

    for action in actions:
        tool = str(action.get("tool") or "")
        if tool not in {"slurm.accounting", "slurm.accounting_aggregate"}:
            continue
        inputs = action.setdefault("inputs", {})
        if all_states:
            if inputs.get("state") is not None or bool(inputs.get("include_all_states")) is not True:
                inputs["state"] = None
                if tool == "slurm.accounting_aggregate":
                    inputs["include_all_states"] = True
                    inputs["default_state_applied"] = False
                result.repairs.append("Applied all-states SLURM obligation to accounting action.")
            continue
        if desired_state:
            current = _canonical_job_state(inputs.get("state"))
            if current != desired_state or bool(inputs.get("include_all_states")):
                inputs["state"] = desired_state
                if tool == "slurm.accounting_aggregate":
                    inputs["include_all_states"] = False
                    inputs["default_state_applied"] = False
                result.repairs.append(f"Applied SLURM state obligation: state={desired_state}.")
        if desired_partition and not str(inputs.get("partition") or "").strip():
            inputs["partition"] = desired_partition
            result.repairs.append(f"Applied SLURM partition obligation: partition={desired_partition}.")
    return result


def extract_semantic_obligations(goal: str) -> list[SemanticObligation]:
    text = str(goal or "")
    obligations: list[SemanticObligation] = []
    if _mentions_slurm(text):
        state_mode = _extract_slurm_state_mode(text)
        if state_mode == "all":
            obligations.append(
                SemanticObligation(
                    domain="slurm",
                    field="state_mode",
                    value="all",
                    strategy="reliable_pushdown",
                    source="goal",
                )
            )
        else:
            state = _extract_slurm_job_state(text)
            if state:
                obligations.append(
                    SemanticObligation(
                        domain="slurm",
                        field="state",
                        value=state,
                        strategy="tentative_pushdown_with_local_filter_fallback",
                        source="goal",
                    )
                )
        partition = _extract_slurm_partition(text)
        if partition:
            obligations.append(
                SemanticObligation(
                    domain="slurm",
                    field="partition",
                    value=partition,
                    strategy="reliable_pushdown",
                    source="goal",
                )
            )
    return obligations


def maybe_apply_semantic_fallback(settings: Settings, *, goal: str, log: StepLog) -> SemanticFallbackResult:
    if not log.success or log.step.action != "slurm.accounting_aggregate":
        return SemanticFallbackResult(log=log)
    obligations = extract_semantic_obligations(goal)
    desired_state = _obligation_value(obligations, domain="slurm", field="state")
    if not desired_state:
        return SemanticFallbackResult(log=log)
    if not _slurm_aggregate_needs_state_fallback(log, desired_state):
        return SemanticFallbackResult(log=log)

    try:
        fallback = _rerun_slurm_accounting_aggregate_with_local_state_filter(
            settings,
            log=log,
            desired_state=desired_state,
        )
    except Exception as exc:  # noqa: BLE001
        return SemanticFallbackResult(
            log=log,
            metadata={
                "attempted": True,
                "tool": "slurm.accounting_aggregate",
                "field": "state",
                "value": desired_state,
                "error": str(exc),
            },
        )
    metadata = dict(fallback.get("semantic_fallback") or {})
    return SemanticFallbackResult(
        log=log.model_copy(update={"result": fallback}),
        applied=True,
        metadata=metadata,
    )


def _rerun_slurm_accounting_aggregate_with_local_state_filter(
    settings: Settings,
    *,
    log: StepLog,
    desired_state: str,
) -> dict[str, Any]:
    args = dict(log.step.args or {})
    broad = slurm_accounting(
        settings,
        user=args.get("user"),
        state=None,
        partition=args.get("partition"),
        start=args.get("start"),
        end=args.get("end"),
        min_elapsed_seconds=None,
        max_elapsed_seconds=None,
        group_by=None,
        limit=args.get("limit"),
        gateway_node=args.get("gateway_node"),
    )
    jobs = [
        dict(job)
        for job in list(broad.get("jobs") or [])
        if _canonical_job_state(job.get("state")) == desired_state
    ]
    intent = SimpleNamespace(
        user=args.get("user"),
        state=desired_state,
        include_all_states=False,
        excluded_states=list(args.get("excluded_states") or []),
        default_state_applied=bool(args.get("default_state_applied") or False),
        partition=args.get("partition"),
        start=args.get("start"),
        end=args.get("end"),
        metric=str(args.get("metric") or "average_elapsed"),
        group_by=args.get("group_by"),
        threshold_seconds=args.get("threshold_seconds"),
        time_window_label=args.get("time_window_label"),
    )
    aggregate = aggregate_slurm_accounting_jobs(jobs, intent)
    warnings = list(aggregate.get("warnings") or [])
    broad_truncated = bool(broad.get("truncated") or False)
    if broad_truncated:
        warnings.append(
            "Semantic fallback used a truncated broad accounting result; aggregate is over returned rows only."
        )
    aggregate.update(
        {
            "value_seconds": aggregate.get("value_seconds"),
            "value_human": aggregate.get("value_human"),
            "count": aggregate.get("count"),
            "count_longer_than": aggregate.get("count_longer_than"),
            "threshold_human": aggregate.get("threshold_human"),
            "filters": {
                "user": args.get("user"),
                "state": desired_state,
                "partition": args.get("partition"),
                "start": args.get("start"),
                "end": args.get("end"),
                "min_elapsed_seconds": None,
                "max_elapsed_seconds": None,
            },
            "source": "sacct",
            "include_all_states": False,
            "excluded_states": list(args.get("excluded_states") or []),
            "default_state_applied": bool(args.get("default_state_applied") or False),
            "state": desired_state,
            "total_count": int(broad.get("total_count") or 0) if broad_truncated else len(jobs),
            "returned_count": len(jobs),
            "limit": broad.get("limit"),
            "truncated": broad_truncated,
            "warnings": warnings,
            "semantic_fallback": {
                "applied": True,
                "tool": "slurm.accounting_aggregate",
                "field": "state",
                "value": desired_state,
                "strategy": "broad_fetch_local_filter",
                "source_total_count": broad.get("total_count"),
                "source_returned_count": broad.get("returned_count"),
                "source_truncated": broad_truncated,
            },
        }
    )
    return aggregate


def _slurm_aggregate_needs_state_fallback(log: StepLog, desired_state: str) -> bool:
    result = dict(log.result or {})
    args = dict(log.step.args or {})
    observed_state = _canonical_job_state(result.get("state"))
    arg_state = _canonical_job_state(args.get("state"))
    if bool(result.get("include_all_states")):
        return True
    if observed_state != desired_state:
        return True
    if arg_state == desired_state and int(result.get("job_count") or 0) == 0:
        return True
    return False


def _obligation_value(obligations: list[SemanticObligation], *, domain: str, field: str) -> str | None:
    for obligation in obligations:
        if obligation.domain == domain and obligation.field == field:
            return obligation.value
    return None


def _mentions_slurm(goal: str) -> bool:
    return bool(
        re.search(
            r"\b(?:slurm|sacct|squeue|jobs?|partition|partitions|cluster|queue|nodes?)\b",
            str(goal or ""),
            re.IGNORECASE,
        )
    )


def _extract_slurm_state_mode(goal: str) -> str | None:
    text = str(goal or "").lower()
    if re.search(r"\b(?:all|any)\s+jobs?\b", text):
        return "all"
    if re.search(r"\b(?:do\s+not|don't|dont)\s+filter\s+by\s+completed\b", text):
        return "all"
    if re.search(r"\bwithout\s+(?:a\s+)?completed(?:-only)?\s+filter\b", text):
        return "all"
    return None


def _extract_slurm_job_state(goal: str) -> str | None:
    text = str(goal or "").lower()
    patterns = [
        ("COMPLETED", r"\b(?:completed|complete|finished|successful|succeeded)\s+jobs?\b|\bjobs?\s+(?:that\s+)?(?:completed|finished|succeeded)\b|\bonly\s+completed\b"),
        ("FAILED", r"\bfailed\s+jobs?\b|\bjobs?\s+(?:that\s+)?failed\b"),
        ("RUNNING", r"\brunning\s+jobs?\b|\bjobs?\s+(?:that\s+are\s+)?running\b"),
        ("PENDING", r"\bpending\s+jobs?\b|\bqueued\s+jobs?\b|\bjobs?\s+(?:that\s+are\s+)?pending\b"),
        ("CANCELLED", r"\bcancell?ed\s+jobs?\b|\bjobs?\s+(?:that\s+were\s+)?cancell?ed\b"),
        ("TIMEOUT", r"\btimeout\s+jobs?\b|\btimed\s+out\s+jobs?\b|\bjobs?\s+that\s+timed\s+out\b"),
    ]
    for state, pattern in patterns:
        if re.search(pattern, text):
            return state
    return None


def _extract_slurm_partition(goal: str) -> str | None:
    text = str(goal or "")
    for pattern in (
        r"\b(?:in|on|for)\s+(?:the\s+)?([A-Za-z0-9._-]+)\s+partition\b",
        r"\bpartition\s+([A-Za-z0-9._-]+)\b",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _canonical_job_state(value: Any) -> str:
    text = str(value or "").strip().upper().replace("-", "_")
    aliases = {
        "CANCELLED": "CANCELLED",
        "CANCELED": "CANCELLED",
        "CD": "COMPLETED",
        "CG": "COMPLETING",
        "COMPLETED": "COMPLETED",
        "COMPLETING": "COMPLETING",
        "FAILED": "FAILED",
        "F": "FAILED",
        "PD": "PENDING",
        "PENDING": "PENDING",
        "R": "RUNNING",
        "RUNNING": "RUNNING",
        "TO": "TIMEOUT",
        "TIMEOUT": "TIMEOUT",
    }
    return aliases.get(text, text)
