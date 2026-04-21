import copy
from typing import Any

from .graph import build_workflow_graph


RUN_INSPECTION_SCHEMA_VERSION = "phase1"


def _compact_text(value: Any, limit: int = 240) -> Any:
    if not isinstance(value, str):
        return value
    compact = value.strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _compact_value(value: Any, *, text_limit: int = 240, row_limit: int = 4) -> Any:
    if isinstance(value, str):
        return _compact_text(value, text_limit)
    if isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            return [_compact_value(item, text_limit=text_limit, row_limit=row_limit) for item in value[:row_limit]]
        return [_compact_value(item, text_limit=text_limit, row_limit=row_limit) for item in value[:row_limit]]
    if isinstance(value, dict):
        compact = {}
        for key in list(value.keys())[:12]:
            compact[key] = _compact_value(value.get(key), text_limit=text_limit, row_limit=row_limit)
        return compact
    return value


def _timeline_stage(timeline: list[dict[str, Any]] | None) -> str | None:
    if not isinstance(timeline, list) or not timeline:
        return None
    last = timeline[-1]
    if not isinstance(last, dict):
        return None
    stage = last.get("stage")
    return stage.strip() if isinstance(stage, str) and stage.strip() else None


def _selected_attempt(attempts: list[dict[str, Any]], selected_attempt_index: Any) -> dict[str, Any] | None:
    if isinstance(selected_attempt_index, int) and 1 <= selected_attempt_index <= len(attempts):
        selected = attempts[selected_attempt_index - 1]
        return selected if isinstance(selected, dict) else None
    return None


def _active_step_id(attempts: list[dict[str, Any]]) -> str | None:
    for attempt in reversed(attempts):
        if not isinstance(attempt, dict):
            continue
        workflow_state = attempt.get("workflow_state")
        if not isinstance(workflow_state, dict):
            continue
        inflight_step = workflow_state.get("inflight_step")
        if isinstance(inflight_step, dict):
            step_id = inflight_step.get("id")
            if isinstance(step_id, str) and step_id.strip():
                return step_id.strip()
    return None


def build_persisted_workflow_graph(session: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(session, dict):
        raise ValueError("Persisted run session must be a dict.")

    terminal_payload = session.get("terminal_payload")
    if isinstance(terminal_payload, dict):
        terminal_graph = terminal_payload.get("graph")
        if isinstance(terminal_graph, dict):
            return copy.deepcopy(terminal_graph)

    attempts = [item for item in session.get("attempts", []) if isinstance(item, dict)]
    selected_attempt = _selected_attempt(attempts, session.get("selected_attempt_index"))
    selected_option = None
    if isinstance(selected_attempt, dict) and isinstance(selected_attempt.get("option"), dict):
        selected_option = selected_attempt.get("option")
    elif isinstance(terminal_payload, dict) and isinstance(terminal_payload.get("selected_option"), dict):
        selected_option = terminal_payload.get("selected_option")

    payload = session.get("payload") if isinstance(session.get("payload"), dict) else {}
    result = None
    if isinstance(terminal_payload, dict):
        result = terminal_payload.get("result")
    if result in (None, "", [], {}) and isinstance(selected_attempt, dict):
        result = selected_attempt.get("result")
    if result in (None, "", [], {}) and attempts:
        result = attempts[-1].get("result")

    return build_workflow_graph(
        task=str(session.get("task") or payload.get("task") or ""),
        task_shape=str(session.get("task_shape") or payload.get("task_shape") or ""),
        status=str(session.get("status") or "unknown"),
        attempts=attempts,
        selected_option=selected_option if isinstance(selected_option, dict) else None,
        result=result,
        presentation=payload.get("presentation") if isinstance(payload.get("presentation"), dict) else None,
        run_id=str(session.get("run_id") or ""),
    )


def build_run_summary(
    session: dict[str, Any],
    *,
    timeline: list[dict[str, Any]] | None = None,
    graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(session, dict):
        raise ValueError("Persisted run session must be a dict.")
    attempts = [item for item in session.get("attempts", []) if isinstance(item, dict)]
    selected_attempt = _selected_attempt(attempts, session.get("selected_attempt_index"))
    selected_option_id = None
    if isinstance(selected_attempt, dict) and isinstance(selected_attempt.get("option"), dict):
        option_id = selected_attempt["option"].get("id")
        if isinstance(option_id, str) and option_id.strip():
            selected_option_id = option_id.strip()
    graph_stats = graph.get("statistics") if isinstance(graph, dict) and isinstance(graph.get("statistics"), dict) else {}
    terminal_payload = session.get("terminal_payload")
    terminal_event = session.get("terminal_event")
    resumable = str(session.get("status") or "").strip() == "running" and not (
        isinstance(terminal_event, str) and terminal_event.strip() and isinstance(terminal_payload, dict)
    )
    replayable = isinstance(terminal_event, str) and terminal_event.strip() and isinstance(terminal_payload, dict)

    return {
        "schema_version": RUN_INSPECTION_SCHEMA_VERSION,
        "run_id": str(session.get("run_id") or ""),
        "task": str(session.get("task") or ""),
        "task_shape": str(session.get("task_shape") or ""),
        "status": str(session.get("status") or ""),
        "created_at": session.get("created_at"),
        "updated_at": session.get("updated_at"),
        "attempt_count": len(attempts),
        "completed_attempt_count": len(
            [item for item in attempts if isinstance(item, dict) and str(item.get("status") or "") == "completed"]
        ),
        "current_attempt_index": session.get("current_attempt_index"),
        "selected_attempt_index": session.get("selected_attempt_index"),
        "selected_option_id": selected_option_id,
        "active_step_id": _active_step_id(attempts),
        "replan_count": session.get("replan_count", 0),
        "uncertain_count": session.get("uncertain_count", 0),
        "terminal_event": terminal_event,
        "timeline_entries": len(timeline or []),
        "last_stage": _timeline_stage(timeline),
        "graph_available": isinstance(graph, dict),
        "graph_node_count": graph_stats.get("node_count"),
        "graph_edge_count": graph_stats.get("edge_count"),
        "replayable": replayable,
        "resumable": resumable,
    }


def _state_view(session: dict[str, Any], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    payload = session.get("payload") if isinstance(session.get("payload"), dict) else {}
    return {
        "task": str(session.get("task") or ""),
        "task_shape": str(session.get("task_shape") or ""),
        "status": str(session.get("status") or ""),
        "limits": copy.deepcopy(session.get("limits")) if isinstance(session.get("limits"), dict) else {},
        "replan_count": session.get("replan_count", 0),
        "uncertain_count": session.get("uncertain_count", 0),
        "current_attempt_index": session.get("current_attempt_index"),
        "selected_attempt_index": session.get("selected_attempt_index"),
        "deferred_clarification": _compact_value(session.get("deferred_clarification"), text_limit=180, row_limit=2),
        "presentation": _compact_value(payload.get("presentation"), text_limit=180, row_limit=2),
        "attempts": _compact_value(attempts, text_limit=200, row_limit=3),
    }


def build_run_inspection(
    session: dict[str, Any],
    *,
    timeline: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(session, dict):
        raise ValueError("Persisted run session must be a dict.")
    attempts = [item for item in session.get("attempts", []) if isinstance(item, dict)]
    graph = build_persisted_workflow_graph(session)
    summary = build_run_summary(session, timeline=timeline, graph=graph)
    return {
        "schema_version": RUN_INSPECTION_SCHEMA_VERSION,
        "run_id": summary["run_id"],
        "summary": summary,
        "state": _state_view(session, attempts),
        "graph": graph,
        "graph_mermaid": render_workflow_graph_mermaid(graph),
        "timeline": copy.deepcopy(timeline) if isinstance(timeline, list) else [],
    }


def _escape_mermaid_text(value: Any) -> str:
    text = str(value or "").strip()
    text = text.replace('"', "'")
    text = text.replace("{", "(").replace("}", ")")
    text = text.replace("[", "(").replace("]", ")")
    return text.replace("\n", "<br/>")


def _mermaid_label(node: dict[str, Any]) -> str:
    kind = str(node.get("kind") or "node").strip()
    title = str(
        node.get("label")
        or node.get("task")
        or node.get("agent_name")
        or node.get("node_id")
        or kind
    ).strip()
    status = str(node.get("status") or "").strip()
    parts = [title]
    if kind and kind not in {"step", "attempt", "workflow"}:
        parts.append(kind)
    if status:
        parts.append(f"[{status}]")
    return _escape_mermaid_text("\n".join(parts))


def _mermaid_class(kind: str) -> str:
    return {
        "workflow": "workflow",
        "attempt": "attempt",
        "step": "step",
        "group_step": "step",
        "validator": "validator",
        "reducer": "reducer",
        "router": "router",
        "replan": "replan",
        "clarification": "clarification",
    }.get(kind, "default")


def render_workflow_graph_mermaid(graph: dict[str, Any]) -> str:
    if not isinstance(graph, dict):
        return "flowchart TD"
    nodes = [item for item in graph.get("nodes", []) if isinstance(item, dict)]
    edges = [item for item in graph.get("edges", []) if isinstance(item, dict)]
    alias_map = {
        str(node.get("node_id") or f"node_{index}"): f"N{index}"
        for index, node in enumerate(nodes, start=1)
    }

    lines = ["flowchart TD"]
    for node in nodes:
        node_id = str(node.get("node_id") or "")
        alias = alias_map.get(node_id)
        if not alias:
            continue
        lines.append(f'  {alias}["{_mermaid_label(node)}"]')

    for edge in edges:
        source = alias_map.get(str(edge.get("source") or ""))
        target = alias_map.get(str(edge.get("target") or ""))
        if not source or not target:
            continue
        relation = _escape_mermaid_text(edge.get("relation") or "")
        connector = f" -->|{relation}| " if relation else " --> "
        lines.append(f"  {source}{connector}{target}")

    lines.extend(
        [
            "  classDef default fill:#f8fafc,stroke:#94a3b8,color:#0f172a;",
            "  classDef workflow fill:#dbeafe,stroke:#2563eb,color:#1e3a8a;",
            "  classDef attempt fill:#e0f2fe,stroke:#0284c7,color:#0f172a;",
            "  classDef step fill:#ecfccb,stroke:#65a30d,color:#365314;",
            "  classDef validator fill:#fef3c7,stroke:#d97706,color:#78350f;",
            "  classDef reducer fill:#dcfce7,stroke:#16a34a,color:#14532d;",
            "  classDef router fill:#f5d0fe,stroke:#c026d3,color:#701a75;",
            "  classDef replan fill:#fde68a,stroke:#ca8a04,color:#78350f;",
            "  classDef clarification fill:#fecaca,stroke:#dc2626,color:#7f1d1d;",
        ]
    )
    for node in nodes:
        node_id = str(node.get("node_id") or "")
        alias = alias_map.get(node_id)
        if not alias:
            continue
        lines.append(f"  class {alias} {_mermaid_class(str(node.get('kind') or 'default'))}")
    return "\n".join(lines)
