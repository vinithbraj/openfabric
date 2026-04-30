"""OpenFABRIC Runtime Module: aor_runtime.runtime.state

Purpose:
    Define runtime state containers used by sessions and execution progress.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from typing_extensions import TypedDict


class RuntimeState(TypedDict, total=False):
    """Represent runtime state within the OpenFABRIC runtime. It extends TypedDict.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by RuntimeState.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.state.RuntimeState and related tests.
    """
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
    planning_metadata: dict[str, Any]
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
    """Initial runtime state for the surrounding runtime workflow.

    Inputs:
        Receives session_id, spec_name, spec_path, input_payload, compiled_spec, trigger for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.state.initial_runtime_state.
    """
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
        "planning_metadata": {},
        "failure_context": None,
        "metrics": {
            "llm_calls": 0,
            "llm_intent_calls": 0,
            "raw_planner_llm_calls": 0,
            "llm_prompt_tokens": 0,
            "llm_completion_tokens": 0,
            "llm_total_tokens": 0,
            "latency_ms": 0.0,
            "steps_executed": 0,
            "retries": 0,
        },
        "error": None,
        "started_at": timestamp,
        "updated_at": timestamp,
    }
