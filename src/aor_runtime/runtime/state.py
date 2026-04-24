from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from typing_extensions import TypedDict


class RuntimeState(TypedDict, total=False):
    session_id: str
    run_id: str
    spec_name: str
    spec_path: str
    goal: str
    input: dict[str, Any]
    compiled_spec: dict[str, Any]
    dry_run: bool
    awaiting_confirmation: bool
    confirmation_kind: str | None
    confirmation_step: dict[str, Any] | None
    confirmation_message: str | None
    policies_used: list[str]
    high_level_plan: list[str] | None
    step_outputs: dict[str, Any]
    plan: dict[str, Any]
    plan_summary: str | None
    plan_canonicalized: bool
    plan_repairs: list[str]
    history: list[dict[str, Any]]
    attempt_history: list[dict[str, Any]]
    status: str
    done: bool
    retries: int
    attempt: int
    current_step_index: int
    trigger: str
    current_node: str
    next_action: str
    validation: dict[str, Any] | None
    validation_checks: list[dict[str, Any]]
    final_output: dict[str, Any]
    failure_context: dict[str, Any] | None
    metrics: dict[str, Any]
    error: str | None
    started_at: str
    updated_at: str


def initial_runtime_state(
    *,
    session_id: str,
    spec_name: str,
    spec_path: str,
    input_payload: dict[str, Any],
    compiled_spec: dict[str, Any],
    trigger: str = "manual",
) -> RuntimeState:
    goal = str(input_payload.get("task") or input_payload.get("prompt") or input_payload.get("input") or "")
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "session_id": session_id,
        "run_id": session_id,
        "spec_name": spec_name,
        "spec_path": spec_path,
        "goal": goal,
        "input": input_payload,
        "compiled_spec": compiled_spec,
        "dry_run": False,
        "awaiting_confirmation": False,
        "confirmation_kind": None,
        "confirmation_step": None,
        "confirmation_message": None,
        "policies_used": [],
        "high_level_plan": None,
        "step_outputs": {},
        "plan": {},
        "plan_summary": None,
        "plan_canonicalized": False,
        "plan_repairs": [],
        "history": [],
        "attempt_history": [],
        "status": "planning",
        "done": False,
        "retries": 0,
        "attempt": 0,
        "current_step_index": 0,
        "trigger": trigger,
        "current_node": "",
        "next_action": "planner",
        "validation": None,
        "validation_checks": [],
        "final_output": {"content": "", "artifacts": [], "metadata": {}},
        "failure_context": None,
        "metrics": {"llm_calls": 0, "latency_ms": 0.0, "steps_executed": 0, "retries": 0},
        "error": None,
        "started_at": timestamp,
        "updated_at": timestamp,
    }
