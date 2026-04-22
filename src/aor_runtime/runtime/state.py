from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Any

from typing_extensions import Annotated, TypedDict


class RuntimeState(TypedDict, total=False):
    run_id: str
    spec_name: str
    task: str
    input: dict[str, Any]
    current_node: str
    next_node: str
    status: str
    history: Annotated[list[dict[str, Any]], operator.add]
    intermediate_outputs: dict[str, Any]
    metadata: dict[str, Any]
    last_result: dict[str, Any]
    error: dict[str, Any] | None
    started_at: str
    updated_at: str


def initial_runtime_state(*, run_id: str, spec_name: str, input_payload: dict[str, Any]) -> RuntimeState:
    task = str(input_payload.get("task") or input_payload.get("prompt") or input_payload.get("input") or "")
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "run_id": run_id,
        "spec_name": spec_name,
        "task": task,
        "input": input_payload,
        "current_node": "",
        "next_node": "",
        "status": "running",
        "history": [],
        "intermediate_outputs": {},
        "metadata": {},
        "last_result": {},
        "error": None,
        "started_at": timestamp,
        "updated_at": timestamp,
    }
