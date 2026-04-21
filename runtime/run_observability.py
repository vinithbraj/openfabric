import copy
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any


RUN_OBSERVABILITY_SCHEMA_VERSION = "phase1"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _duration_between(start: Any, end: Any) -> float | None:
    start_ts = _parse_timestamp(start)
    end_ts = _parse_timestamp(end)
    if start_ts is None or end_ts is None:
        return None
    return round(max(0.0, (end_ts - start_ts).total_seconds() * 1000), 2)


def _normalized_status(value: Any) -> str:
    text = str(value or "").strip()
    return text or "unknown"


def _normalized_verdict(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    verdict = str(payload.get("verdict") or "").strip()
    if verdict:
        return verdict
    if payload.get("valid") is True:
        return "valid"
    if payload.get("valid") is False:
        return "invalid"
    return ""


def _routing_actions(record: dict[str, Any]) -> list[str]:
    history = record.get("routing_history")
    if isinstance(history, list):
        actions = [
            str(item.get("action") or "").strip()
            for item in history
            if isinstance(item, dict) and str(item.get("action") or "").strip()
        ]
        if actions:
            return actions
    routing = record.get("routing")
    if isinstance(routing, dict):
        action = str(routing.get("action") or "").strip()
        if action:
            return [action]
    return []


def _flatten_steps(
    steps: list[Any],
    *,
    attempt: int,
    parent_step_id: str = "",
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for index, raw_step in enumerate(steps, start=1):
        if not isinstance(raw_step, dict):
            continue
        step_id = str(raw_step.get("id") or raw_step.get("step_id") or f"step{index}").strip() or f"step{index}"
        validation = raw_step.get("validation") if isinstance(raw_step.get("validation"), dict) else {}
        routing = raw_step.get("routing") if isinstance(raw_step.get("routing"), dict) else {}
        flattened.append(
            {
                "attempt": attempt,
                "step_id": step_id,
                "parent_step_id": parent_step_id or None,
                "task": str(raw_step.get("task") or "").strip(),
                "target_agent": str(raw_step.get("target_agent") or "").strip(),
                "status": _normalized_status(raw_step.get("status")),
                "duration_ms": round(_safe_float(raw_step.get("duration_ms")), 2),
                "event": str(raw_step.get("event") or "").strip(),
                "error": str(raw_step.get("error") or "").strip(),
                "validation_verdict": _normalized_verdict(validation),
                "validation_reason": str(validation.get("reason") or "").strip() if isinstance(validation, dict) else "",
                "routing_action": str(routing.get("action") or "").strip() if isinstance(routing, dict) else "",
                "routing_reason": str(routing.get("reason") or "").strip() if isinstance(routing, dict) else "",
                "has_nested_steps": isinstance(raw_step.get("steps"), list) and bool(raw_step.get("steps")),
                "replan_count": len([item for item in raw_step.get("replan_history", []) if isinstance(item, dict)])
                if isinstance(raw_step.get("replan_history"), list)
                else (1 if isinstance(raw_step.get("replan"), dict) else 0),
                "clarification_required": isinstance(raw_step.get("clarification"), dict),
                "raw": copy.deepcopy(raw_step),
            }
        )
        nested_steps = raw_step.get("steps")
        if isinstance(nested_steps, list) and nested_steps:
            flattened.extend(_flatten_steps(nested_steps, attempt=attempt, parent_step_id=step_id))
    return flattened


def build_run_observability(
    session: dict[str, Any],
    *,
    timeline: list[dict[str, Any]] | None = None,
    graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(session, dict):
        raise ValueError("Persisted run session must be a dict.")

    attempts = [item for item in session.get("attempts", []) if isinstance(item, dict)]
    flattened_steps: list[dict[str, Any]] = []
    workflow_validation_counter: Counter[str] = Counter()
    step_validation_counter: Counter[str] = Counter()
    routing_counter: Counter[str] = Counter()
    failure_records: list[dict[str, Any]] = []
    attempt_metrics: list[dict[str, Any]] = []
    agent_metrics: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "agent": "",
            "step_count": 0,
            "completed_step_count": 0,
            "failed_step_count": 0,
            "total_duration_ms": 0.0,
            "max_duration_ms": 0.0,
        }
    )

    for index, attempt in enumerate(attempts, start=1):
        attempt_number = _safe_int(attempt.get("attempt")) or index
        option = attempt.get("option") if isinstance(attempt.get("option"), dict) else {}
        attempt_steps = _flatten_steps(attempt.get("steps", []), attempt=attempt_number)
        flattened_steps.extend(attempt_steps)

        attempt_verdict = _normalized_verdict(attempt.get("validation"))
        if attempt_verdict:
            workflow_validation_counter[attempt_verdict] += 1
        attempt_actions = _routing_actions(attempt)
        for action in attempt_actions:
            routing_counter[action] += 1

        total_duration_ms = round(sum(item.get("duration_ms", 0.0) for item in attempt_steps), 2)
        attempt_error = str(attempt.get("error") or "").strip()
        attempt_record = {
            "attempt": attempt_number,
            "option_id": str(option.get("id") or "").strip(),
            "option_label": str(option.get("label") or "").strip(),
            "status": _normalized_status(attempt.get("status")),
            "validation_verdict": attempt_verdict,
            "routing_action": attempt_actions[-1] if attempt_actions else "",
            "step_count": len(attempt_steps),
            "failed_step_count": len(
                [item for item in attempt_steps if item.get("status") in {"failed", "needs_clarification"} or item.get("error")]
            ),
            "total_duration_ms": total_duration_ms,
            "error": attempt_error,
        }
        attempt_metrics.append(attempt_record)

        if attempt_error or attempt_verdict in {"invalid", "uncertain"}:
            failure_records.append(
                {
                    "scope": "workflow",
                    "attempt": attempt_number,
                    "status": attempt_record["status"],
                    "reason": str((attempt.get("validation") or {}).get("reason") or attempt_error).strip(),
                    "error": attempt_error,
                    "option_id": attempt_record["option_id"],
                    "option_label": attempt_record["option_label"],
                    "validation_verdict": attempt_verdict,
                    "routing_action": attempt_record["routing_action"],
                }
            )

    step_status_counter: Counter[str] = Counter()
    for step in flattened_steps:
        status = _normalized_status(step.get("status"))
        step_status_counter[status] += 1

        verdict = str(step.get("validation_verdict") or "").strip()
        if verdict:
            step_validation_counter[verdict] += 1

        for action in _routing_actions(step.get("raw") if isinstance(step.get("raw"), dict) else {}):
            routing_counter[action] += 1

        agent_name = str(step.get("target_agent") or "").strip() or "unknown"
        metrics = agent_metrics[agent_name]
        metrics["agent"] = agent_name
        metrics["step_count"] += 1
        if status == "completed":
            metrics["completed_step_count"] += 1
        if status in {"failed", "needs_clarification"} or step.get("error"):
            metrics["failed_step_count"] += 1
        metrics["total_duration_ms"] = round(metrics["total_duration_ms"] + _safe_float(step.get("duration_ms")), 2)
        metrics["max_duration_ms"] = round(max(metrics["max_duration_ms"], _safe_float(step.get("duration_ms"))), 2)

        if status in {"failed", "needs_clarification"} or step.get("error"):
            failure_records.append(
                {
                    "scope": "step",
                    "attempt": step.get("attempt"),
                    "step_id": step.get("step_id"),
                    "task": step.get("task"),
                    "target_agent": agent_name,
                    "status": status,
                    "reason": str(step.get("validation_reason") or step.get("routing_reason") or step.get("error") or "").strip(),
                    "error": str(step.get("error") or "").strip(),
                    "duration_ms": step.get("duration_ms"),
                }
            )

    sorted_agents = sorted(
        (
            {
                **metrics,
                "avg_duration_ms": round(metrics["total_duration_ms"] / metrics["step_count"], 2)
                if metrics["step_count"]
                else 0.0,
            }
            for metrics in agent_metrics.values()
        ),
        key=lambda item: (-item["total_duration_ms"], item["agent"]),
    )

    slowest_steps = sorted(
        [
            {
                "attempt": item.get("attempt"),
                "step_id": item.get("step_id"),
                "task": item.get("task"),
                "target_agent": item.get("target_agent"),
                "status": item.get("status"),
                "event": item.get("event"),
                "duration_ms": item.get("duration_ms"),
            }
            for item in flattened_steps
        ],
        key=lambda item: (-_safe_float(item.get("duration_ms")), str(item.get("step_id") or "")),
    )[:10]

    total_step_duration_ms = round(sum(item.get("duration_ms", 0.0) for item in flattened_steps), 2)
    wall_clock_duration_ms = _duration_between(session.get("created_at"), session.get("updated_at"))
    graph_stats = graph.get("statistics") if isinstance(graph, dict) and isinstance(graph.get("statistics"), dict) else {}
    timeline_entries = [item for item in (timeline or []) if isinstance(item, dict)]
    timeline_stage_counter = Counter(str(item.get("stage") or "").strip() for item in timeline_entries if str(item.get("stage") or "").strip())
    timeline_observations = []
    for index, item in enumerate(timeline_entries):
        previous = timeline_entries[index - 1] if index > 0 else None
        timeline_observations.append(
            {
                "stage": str(item.get("stage") or "").strip(),
                "status": _normalized_status(item.get("status")),
                "updated_at": item.get("updated_at"),
                "delta_ms_from_previous": _duration_between(
                    previous.get("updated_at") if isinstance(previous, dict) else None,
                    item.get("updated_at"),
                ),
            }
        )

    error_count = len(failure_records)
    step_count = len(flattened_steps)
    observability = {
        "schema_version": RUN_OBSERVABILITY_SCHEMA_VERSION,
        "run_id": str(session.get("run_id") or ""),
        "status": _normalized_status(session.get("status")),
        "created_at": session.get("created_at"),
        "updated_at": session.get("updated_at"),
        "terminal_event": session.get("terminal_event"),
        "wall_clock_duration_ms": wall_clock_duration_ms,
        "timings": {
            "wall_clock_duration_ms": wall_clock_duration_ms,
            "step_total_duration_ms": total_step_duration_ms,
            "step_avg_duration_ms": round(total_step_duration_ms / step_count, 2) if step_count else 0.0,
            "max_step_duration_ms": slowest_steps[0]["duration_ms"] if slowest_steps else 0.0,
        },
        "counts": {
            "attempt_count": len(attempts),
            "step_count": step_count,
            "error_count": error_count,
            "replan_count": sum(_safe_int(item.get("replan_count")) for item in flattened_steps),
            "clarification_count": len([item for item in flattened_steps if item.get("clarification_required")]),
            "validation_request_count": sum(workflow_validation_counter.values()) + sum(step_validation_counter.values()),
            "workflow_validation_count": sum(workflow_validation_counter.values()),
            "step_validation_count": sum(step_validation_counter.values()),
        },
        "step_status_counts": dict(step_status_counter),
        "validation_counts": {
            "workflow": dict(workflow_validation_counter),
            "step": dict(step_validation_counter),
        },
        "routing_action_counts": dict(routing_counter),
        "node_kind_counts": {
            key: graph_stats.get(key)
            for key in (
                "node_count",
                "edge_count",
                "attempt_count",
                "step_count",
                "validator_count",
                "reducer_count",
                "router_count",
                "replan_count",
                "clarification_count",
            )
            if graph_stats.get(key) is not None
        },
        "agent_metrics": sorted_agents,
        "attempt_metrics": attempt_metrics,
        "slowest_steps": slowest_steps,
        "failures": failure_records[:20],
        "timeline": {
            "entry_count": len(timeline_entries),
            "stage_counts": dict(timeline_stage_counter),
            "stages": timeline_observations,
        },
        "audit": {
            "has_errors": error_count > 0 or _normalized_status(session.get("status")) == "failed",
            "agents": [item["agent"] for item in sorted_agents if item.get("agent")],
            "max_step_duration_ms": slowest_steps[0]["duration_ms"] if slowest_steps else 0.0,
            "resumable": _normalized_status(session.get("status")) == "running"
            and not (
                isinstance(session.get("terminal_event"), str)
                and session.get("terminal_event")
                and isinstance(session.get("terminal_payload"), dict)
            ),
            "replayable": isinstance(session.get("terminal_event"), str)
            and bool(str(session.get("terminal_event")).strip())
            and isinstance(session.get("terminal_payload"), dict),
        },
    }
    return observability
