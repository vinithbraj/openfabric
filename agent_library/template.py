from __future__ import annotations

from copy import deepcopy
from typing import Any

from .contracts import build_agent_api, build_agent_descriptor


def agent_api(**kwargs: Any) -> dict[str, Any]:
    return build_agent_api(**kwargs)


def agent_descriptor(**kwargs: Any) -> dict[str, Any]:
    return build_agent_descriptor(**kwargs)


def emit(event: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"emits": [{"event": event, "payload": payload}]}


def emit_many(*events: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    emits = []
    for event_name, payload in events:
        if not isinstance(event_name, str) or not event_name.strip():
            continue
        if not isinstance(payload, dict):
            continue
        emits.append({"event": event_name, "payload": deepcopy(payload)})
    return {"emits": emits}


def emit_sequence(events: list[dict[str, Any]]) -> dict[str, Any]:
    emits = []
    for item in events:
        if not isinstance(item, dict):
            continue
        event_name = item.get("event")
        payload = item.get("payload")
        if not isinstance(event_name, str) or not event_name.strip():
            continue
        if not isinstance(payload, dict):
            continue
        emits.append({"event": event_name, "payload": deepcopy(payload)})
    return {"emits": emits}


def noop() -> dict[str, Any]:
    return {"emits": []}


def task_result(
    detail: str,
    *,
    status: str | None = None,
    error: Any = None,
    result: Any = None,
    replan_hint: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"detail": detail}
    if isinstance(status, str) and status.strip():
        payload["status"] = status.strip()
    if error not in (None, "", [], {}):
        payload["error"] = error
    if result is not None:
        payload["result"] = result
    if isinstance(replan_hint, dict) and replan_hint:
        payload["replan_hint"] = deepcopy(replan_hint)
    for key, value in extra.items():
        if value not in (None, "", [], {}):
            payload[key] = deepcopy(value)
    return emit("task.result", payload)


def failure_result(
    detail: str,
    *,
    error: Any = None,
    result: Any = None,
    status: str = "failed",
    **extra: Any,
) -> dict[str, Any]:
    return task_result(detail, status=status, error=error or detail, result=result, **extra)


def needs_decomposition(
    detail: str,
    *,
    suggested_capabilities: list[str] | None = None,
    failure_class: str = "needs_decomposition",
    result: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    replan_hint = {
        "reason": detail,
        "failure_class": failure_class,
    }
    if isinstance(suggested_capabilities, list):
        safe = [str(item).strip() for item in suggested_capabilities if isinstance(item, str) and str(item).strip()]
        if safe:
            replan_hint["suggested_capabilities"] = safe
    return task_result(
        detail,
        status="needs_decomposition",
        error=detail,
        result=result,
        replan_hint=replan_hint,
        **extra,
    )


def final_answer(answer: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"answer": answer}
    for key, value in extra.items():
        if value not in (None, "", [], {}):
            payload[key] = deepcopy(value)
    return emit("answer.final", payload)
