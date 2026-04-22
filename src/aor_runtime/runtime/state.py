from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from typing_extensions import TypedDict


class RuntimeState(TypedDict, total=False):
    run_id: str
    spec_name: str
    goal: str
    input: dict[str, Any]
    plan: dict[str, Any]
    history: list[dict[str, Any]]
    status: str
    retries: int
    current_node: str
    next_node: str
    validation: dict[str, Any] | None
    final_output: dict[str, Any]
    failure_context: dict[str, Any] | None
    error: str | None
    started_at: str
    updated_at: str


def initial_runtime_state(*, run_id: str, spec_name: str, input_payload: dict[str, Any]) -> RuntimeState:
    goal = str(input_payload.get("task") or input_payload.get("prompt") or input_payload.get("input") or "")
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "run_id": run_id,
        "spec_name": spec_name,
        "goal": goal,
        "input": input_payload,
        "plan": {},
        "history": [],
        "status": "planning",
        "retries": 0,
        "current_node": "",
        "next_node": "",
        "validation": None,
        "final_output": {"content": "", "artifacts": [], "metadata": {}},
        "failure_context": None,
        "error": None,
        "started_at": timestamp,
        "updated_at": timestamp,
    }
