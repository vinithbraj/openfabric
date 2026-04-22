import hashlib
import json
from typing import Any

from agent_library.contracts import api_emit_events, api_trigger_event


GRAPH_SCHEMA_VERSION = "phase3"


def _stable_suffix(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _compact_text(value: Any, limit: int = 400) -> Any:
    if not isinstance(value, str):
        return value
    compact = value.strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _compact_rows(rows: Any, limit: int = 4) -> Any:
    if not isinstance(rows, list):
        return rows
    return [_compact_value(row, text_limit=180, row_limit=3) for row in rows[:limit]]


def _compact_value(value: Any, *, text_limit: int = 400, row_limit: int = 4) -> Any:
    if isinstance(value, str):
        return _compact_text(value, text_limit)
    if isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            return _compact_rows(value, row_limit)
        return [_compact_value(item, text_limit=text_limit, row_limit=row_limit) for item in value[:row_limit]]
    if isinstance(value, dict):
        compact = {}
        keys = list(value.keys())[:10]
        for key in keys:
            item = value.get(key)
            if key == "rows":
                compact[key] = _compact_rows(item, row_limit)
                if isinstance(item, list) and len(item) > row_limit:
                    compact["rows_note"] = f"Showing first {row_limit} rows out of {len(item)}."
            else:
                compact[key] = _compact_value(item, text_limit=text_limit, row_limit=row_limit)
        return compact
    return value


def _edge(edge_id: str, source: str, target: str, relation: str, **extra: Any) -> dict:
    payload = {
        "edge_id": edge_id,
        "source": source,
        "target": target,
        "relation": relation,
    }
    for key, value in extra.items():
        if value not in (None, "", [], {}):
            payload[key] = value
    return payload


def infer_agent_graph_role(
    agent_name: str,
    subscribes_to: list[str] | None,
    emits: list[str] | None,
    metadata: dict | None = None,
) -> str:
    metadata = metadata or {}
    api_specs = metadata.get("apis", metadata.get("methods", []))
    subscribe_set = {item for item in (subscribes_to or []) if isinstance(item, str)}
    emit_set = {item for item in (emits or []) if isinstance(item, str)}
    if isinstance(api_specs, list):
        for api in api_specs:
            if not isinstance(api, dict):
                continue
            trigger_event = api_trigger_event(api)
            if trigger_event:
                subscribe_set.add(trigger_event)
            emit_set.update(api_emit_events(api))
    lowered = agent_name.lower()

    if "validation.request" in subscribe_set or "validator" in lowered:
        return "validator"
    if "data.reduce" in subscribe_set or "data.reduced" in emit_set or "reducer" in lowered:
        return "reducer"
    if "answer.final" in emit_set or "synthesizer" in lowered:
        return "synthesizer"
    if "notifier" in lowered or "notify.result" in emit_set:
        return "notifier"
    if "filesystem" in lowered or "file.content" in emit_set:
        return "filesystem"
    if "planner.replan.result" in emit_set or "task.plan" in emit_set or "planner" in lowered:
        return "router"
    capability_domains = metadata.get("capability_domains")
    if isinstance(capability_domains, list):
        domains = {item for item in capability_domains if isinstance(item, str)}
        if "response_synthesis" in domains:
            return "synthesizer"
        if "planning" in domains or "routing" in domains:
            return "router"
    return "executor"


def _api_spec(method: dict[str, Any]) -> dict[str, Any]:
    trigger_event = api_trigger_event(method)
    emitted_events = api_emit_events(method)
    api = {
        "name": str(method.get("name") or "").strip(),
        "event": trigger_event,
        "trigger_event": trigger_event,
    }
    if emitted_events:
        api["emits"] = emitted_events
    summary = method.get("summary")
    when = method.get("when")
    if isinstance(summary, str) and summary.strip():
        api["summary"] = summary.strip()
    if isinstance(when, str) and when.strip():
        api["when"] = when.strip()
    for key in ("intent_tags", "examples", "risk_level", "anti_patterns", "side_effect_level"):
        value = method.get(key)
        if isinstance(value, list):
            safe_values = [item for item in value if isinstance(item, str) and item.strip()]
            if safe_values:
                api[key] = safe_values
        elif isinstance(value, str) and value.strip():
            api[key] = value.strip()
    if isinstance(method.get("deterministic"), bool):
        api["deterministic"] = method.get("deterministic")
    for key in ("request_contract", "result_contract"):
        value = method.get(key)
        if isinstance(value, str) and value.strip():
            api[key] = value.strip()
    for key in ("request_envelope_fields", "result_envelope_fields"):
        value = method.get(key)
        if isinstance(value, list):
            safe_values = [item for item in value if isinstance(item, str) and item.strip()]
            if safe_values:
                api[key] = safe_values
    for key in ("input_schema", "output_schema"):
        value = method.get(key)
        if isinstance(value, dict) and value:
            api[key] = _compact_value(value, text_limit=180, row_limit=3)
    return api


def build_agent_graph_node(
    agent_name: str,
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    subscribes_to = config.get("subscribes_to", [])
    emits = config.get("emits", [])
    runtime_cfg = config.get("runtime", {})
    methods = metadata.get("apis", metadata.get("methods", config.get("methods", [])))
    if isinstance(methods, list):
        for item in methods:
            if not isinstance(item, dict):
                continue
            trigger_event = api_trigger_event(item)
            if trigger_event and trigger_event not in subscribes_to:
                subscribes_to = [*subscribes_to, trigger_event]
            for emitted_event in api_emit_events(item):
                if emitted_event not in emits:
                    emits = [*emits, emitted_event]
    role = infer_agent_graph_role(agent_name, subscribes_to, emits, metadata)

    graph_node = {
        "node_id": f"agent:{agent_name}",
        "kind": "agent",
        "role": role,
        "agent_name": agent_name,
        "label": agent_name,
        "description": str(metadata.get("description", config.get("description", "")) or "").strip(),
        "interfaces": {
            "receives": [item for item in subscribes_to if isinstance(item, str)],
            "emits": [item for item in emits if isinstance(item, str)],
        },
        "capabilities": {
            "domains": [item for item in metadata.get("capability_domains", []) if isinstance(item, str)],
            "verbs": [item for item in metadata.get("action_verbs", []) if isinstance(item, str)],
            "apis": [_api_spec(item) for item in methods if isinstance(item, dict)],
        },
        "runtime": {
            "adapter": runtime_cfg.get("adapter"),
            "endpoint": runtime_cfg.get("endpoint"),
        },
    }

    contract = {}
    contract_version = metadata.get("contract_version")
    if isinstance(contract_version, str) and contract_version.strip():
        contract["version"] = contract_version.strip()
    for key in ("request_contract", "result_contract"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            contract[key] = value.strip()
    for key in ("request_envelope_fields", "result_envelope_fields"):
        value = metadata.get(key)
        if isinstance(value, list) and value:
            contract[key] = [item for item in value if isinstance(item, str) and item.strip()]
    request_schema = metadata.get("request_schema")
    if isinstance(request_schema, dict) and request_schema:
        contract["request_schema"] = _compact_value(request_schema, text_limit=180, row_limit=3)
    result_schema = metadata.get("result_schema")
    if isinstance(result_schema, dict) and result_schema:
        contract["result_schema"] = _compact_value(result_schema, text_limit=180, row_limit=3)
    if contract:
        graph_node["contract"] = contract

    for key in (
        "execution_model",
        "deterministic_catalog_version",
        "deterministic_catalog_reference",
        "side_effect_policy",
        "safety_enforced_by_agent",
        "template_agent",
        "database_name",
        "cluster_name",
    ):
        value = metadata.get(key)
        if value not in (None, "", [], {}):
            graph_node[key] = value
    return graph_node


def build_capability_graph(agent_catalog: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_events: set[str] = set()

    for item in agent_catalog:
        if not isinstance(item, dict):
            continue
        graph_node = item.get("graph_node")
        if isinstance(graph_node, dict):
            nodes.append(_compact_value(graph_node, text_limit=240, row_limit=3))
        agent_node_id = f"agent:{item.get('name')}"
        for event_name in item.get("subscribes_to", []):
            if not isinstance(event_name, str) or not event_name.strip():
                continue
            event_id = f"event:{event_name.strip()}"
            if event_id not in seen_events:
                seen_events.add(event_id)
                nodes.append(
                    {
                        "node_id": event_id,
                        "kind": "event",
                        "role": "event",
                        "label": event_name.strip(),
                    }
                )
            edges.append(
                _edge(
                    f"{event_id}->{agent_node_id}:consumes",
                    event_id,
                    agent_node_id,
                    "consumes",
                )
            )
        for event_name in item.get("emits", []):
            if not isinstance(event_name, str) or not event_name.strip():
                continue
            event_id = f"event:{event_name.strip()}"
            if event_id not in seen_events:
                seen_events.add(event_id)
                nodes.append(
                    {
                        "node_id": event_id,
                        "kind": "event",
                        "role": "event",
                        "label": event_name.strip(),
                    }
                )
            edges.append(
                _edge(
                    f"{agent_node_id}->{event_id}:emits",
                    agent_node_id,
                    event_id,
                    "emits",
                )
            )

    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "kind": "capability_topology",
        "nodes": nodes,
        "edges": edges,
        "statistics": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "agent_count": len([item for item in nodes if item.get("kind") == "agent"]),
            "event_count": len([item for item in nodes if item.get("kind") == "event"]),
        },
    }


def _extract_local_reduction_command(step: dict[str, Any]) -> str:
    payload = step.get("payload")
    if isinstance(payload, dict):
        command = payload.get("local_reduction_command")
        if isinstance(command, str) and command.strip():
            return command.strip()
        nested_result = payload.get("result")
        if isinstance(nested_result, dict):
            command = nested_result.get("local_reduction_command")
            if isinstance(command, str) and command.strip():
                return command.strip()
    evidence = step.get("evidence")
    if isinstance(evidence, dict):
        payload = evidence.get("payload")
        if isinstance(payload, dict):
            command = payload.get("local_reduction_command")
            if isinstance(command, str) and command.strip():
                return command.strip()
    return ""


def _extract_reduction_request(step: dict[str, Any]) -> dict[str, Any]:
    payload = step.get("payload")
    if isinstance(payload, dict):
        request = payload.get("reduction_request")
        if isinstance(request, dict):
            return request
    evidence = step.get("evidence")
    if isinstance(evidence, dict):
        payload = evidence.get("payload")
        if isinstance(payload, dict):
            request = payload.get("reduction_request")
            if isinstance(request, dict):
                return request
    return {}


def _extract_reduced_output(step: dict[str, Any]) -> Any:
    payload = step.get("payload")
    if isinstance(payload, dict):
        for key in ("reduced_result", "refined_answer"):
            value = payload.get(key)
            if value not in (None, "", [], {}):
                return value
        nested_result = payload.get("result")
        if isinstance(nested_result, dict):
            for key in ("reduced_result", "refined_answer"):
                value = nested_result.get(key)
                if value not in (None, "", [], {}):
                    return value
    evidence = step.get("evidence")
    if isinstance(evidence, dict):
        summary = evidence.get("summary_text")
        if summary not in (None, "", [], {}):
            return summary
    return step.get("result")


def _append_validation_node(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    source_node_id: str,
    validation: dict[str, Any] | None,
    scope_id: str,
) -> str:
    if not isinstance(validation, dict) or not validation:
        return source_node_id
    validator_node_id = f"{source_node_id}:validator"
    nodes.append(
        {
            "node_id": validator_node_id,
            "kind": "validator",
            "role": "validator",
            "label": f"validate {scope_id}",
            "status": "passed" if validation.get("valid") else validation.get("verdict", "unknown"),
            "validation": _compact_value(validation, text_limit=200, row_limit=3),
        }
    )
    edges.append(
        _edge(
            f"{source_node_id}->{validator_node_id}:validated_by",
            source_node_id,
            validator_node_id,
            "validated_by",
        )
    )
    return validator_node_id


def _append_reducer_node(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    source_node_id: str,
    step: dict[str, Any],
) -> str:
    reduction_command = _extract_local_reduction_command(step)
    reduction_request = _extract_reduction_request(step)
    payload = step.get("payload") if isinstance(step.get("payload"), dict) else {}
    reduction = payload.get("reduction") if isinstance(payload, dict) else None
    reduction_strategy = payload.get("reduction_strategy") if isinstance(payload, dict) else None
    if not reduction_command and not reduction and not reduction_strategy and not reduction_request:
        return source_node_id
    reducer_node_id = f"{source_node_id}:reducer"
    nodes.append(
        {
            "node_id": reducer_node_id,
            "kind": "reducer",
            "role": "reducer",
            "label": f"reduce {step.get('id')}",
            "status": step.get("status", "unknown"),
            "instruction": {
                "operation": "planned_reduction" if reduction_request else "local_reduction",
                "command": _compact_text(reduction_command, 240) if reduction_command else None,
                "request": _compact_value(reduction_request, text_limit=180, row_limit=3) if reduction_request else None,
            },
            "strategy": reduction_strategy,
            "reduction": _compact_value(reduction, text_limit=180, row_limit=3) if isinstance(reduction, dict) else None,
            "result": _compact_value(_extract_reduced_output(step), text_limit=200, row_limit=3),
        }
    )
    edges.append(
        _edge(
            f"{source_node_id}->{reducer_node_id}:reduced_by",
            source_node_id,
            reducer_node_id,
            "reduced_by",
        )
    )
    return reducer_node_id


def _history_records(item: dict[str, Any], key: str) -> list[dict[str, Any]]:
    history = item.get(f"{key}_history")
    if isinstance(history, list) and history:
        return [entry for entry in history if isinstance(entry, dict)]
    single = item.get(key)
    if isinstance(single, dict):
        return [single]
    return []


def _append_router_nodes(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    source_node_id: str,
    item: dict[str, Any],
    scope_id: str,
) -> tuple[str, list[str]]:
    previous_node_id = source_node_id
    router_node_ids: list[str] = []
    for index, routing in enumerate(_history_records(item, "routing"), start=1):
        router_node_id = f"{source_node_id}:router" if index == 1 else f"{source_node_id}:router:{index}"
        nodes.append(
            {
                "node_id": router_node_id,
                "kind": "router",
                "role": "router",
                "scope": routing.get("scope"),
                "stage": routing.get("stage"),
                "label": f"{routing.get('action') or 'route'} {scope_id}",
                "status": routing.get("action") or "unknown",
                "decision": _compact_value(routing, text_limit=200, row_limit=3),
            }
        )
        edges.append(
            _edge(
                f"{previous_node_id}->{router_node_id}:routed_by",
                previous_node_id,
                router_node_id,
                "routed_by",
            )
        )
        previous_node_id = router_node_id
        router_node_ids.append(router_node_id)
    return previous_node_id, router_node_ids


def _append_replan_nodes(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    source_node_id: str,
    item: dict[str, Any],
    scope_id: str,
) -> tuple[str, list[str]]:
    previous_node_id = source_node_id
    replan_node_ids: list[str] = []
    for index, replan in enumerate(_history_records(item, "replan"), start=1):
        replan_node_id = f"{source_node_id}:replan" if index == 1 else f"{source_node_id}:replan:{index}"
        nodes.append(
            {
                "node_id": replan_node_id,
                "kind": "replan",
                "role": "router",
                "scope": replan.get("scope"),
                "label": f"replan {scope_id}",
                "status": replan.get("status", "requested"),
                "reason": _compact_text(replan.get("reason"), 200),
                "request": _compact_value(replan.get("request"), text_limit=180, row_limit=3),
                "result": _compact_value(replan.get("result"), text_limit=180, row_limit=3),
                "steps": _compact_value(replan.get("steps"), text_limit=180, row_limit=3),
                "replace_step_id": replan.get("replace_step_id"),
                "derived_option_id": replan.get("derived_option_id"),
            }
        )
        edges.append(
            _edge(
                f"{previous_node_id}->{replan_node_id}:replans_with",
                previous_node_id,
                replan_node_id,
                "replans_with",
            )
        )
        previous_node_id = replan_node_id
        replan_node_ids.append(replan_node_id)
    return previous_node_id, replan_node_ids


def _append_clarification_node(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    source_node_id: str,
    clarification: dict[str, Any] | None,
    scope_id: str,
) -> str:
    if not isinstance(clarification, dict) or not clarification:
        return source_node_id
    clarification_node_id = f"{source_node_id}:clarification"
    nodes.append(
        {
            "node_id": clarification_node_id,
            "kind": "clarification",
            "role": "router",
            "label": f"clarify {scope_id}",
            "status": "required",
            "detail": _compact_text(clarification.get("detail"), 220),
            "question": _compact_text(clarification.get("question"), 220),
            "missing_information": clarification.get("missing_information"),
        }
    )
    edges.append(
        _edge(
            f"{source_node_id}->{clarification_node_id}:requires_clarification",
            source_node_id,
            clarification_node_id,
            "requires_clarification",
        )
    )
    return clarification_node_id


def _step_node_id(scope_node_id: str, step: dict[str, Any]) -> str:
    step_id = str(step.get("id") or step.get("step_id") or f"step-{_stable_suffix(step)}")
    return f"{scope_node_id}:step:{step_id}"


def _append_step_nodes(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    parent_node_id: str,
    scope_node_id: str,
    step: dict[str, Any],
) -> None:
    step_id = str(step.get("id") or step.get("step_id") or f"step-{_stable_suffix(step)}")
    node_id = _step_node_id(scope_node_id, step)
    nested_steps = step.get("steps")
    kind = "group_step" if isinstance(nested_steps, list) and nested_steps else "step"
    instruction = step.get("instruction") if isinstance(step.get("instruction"), dict) else None

    node = {
        "node_id": node_id,
        "kind": kind,
        "role": "executor",
        "label": step_id,
        "task": step.get("task", ""),
        "agent_name": step.get("target_agent", ""),
        "status": step.get("status", "unknown"),
        "execution": {
            "event": step.get("event", ""),
            "duration_ms": step.get("duration_ms"),
            "raw_output": _compact_value(step.get("payload"), text_limit=220, row_limit=3),
            "result": _compact_value(step.get("result"), text_limit=220, row_limit=3),
            "evidence": _compact_value(step.get("evidence"), text_limit=220, row_limit=3),
        },
    }
    if instruction:
        node["instruction"] = _compact_value(instruction, text_limit=220, row_limit=3)
    depends_on = step.get("depends_on")
    if isinstance(depends_on, list) and depends_on:
        node["depends_on"] = [item for item in depends_on if isinstance(item, str)]
    when = step.get("when")
    if isinstance(when, dict) and when:
        node["when"] = _compact_value(when, text_limit=180, row_limit=3)

    nodes.append(node)
    edges.append(
        _edge(
            f"{parent_node_id}->{node_id}:contains",
            parent_node_id,
            node_id,
            "contains",
        )
    )

    if isinstance(depends_on, list):
        for dependency in depends_on:
            if isinstance(dependency, str) and dependency.strip():
                edges.append(
                    _edge(
                        f"{scope_node_id}:step:{dependency}->{node_id}:depends_on",
                        f"{scope_node_id}:step:{dependency}",
                        node_id,
                        "depends_on",
                    )
                )

    current_node_id = _append_reducer_node(nodes, edges, node_id, step)
    current_node_id = _append_validation_node(nodes, edges, current_node_id, step.get("validation"), step_id)
    current_node_id, _ = _append_router_nodes(nodes, edges, current_node_id, step, step_id)
    current_node_id, replan_node_ids = _append_replan_nodes(nodes, edges, current_node_id, step, step_id)
    _append_clarification_node(nodes, edges, current_node_id, step.get("clarification"), step_id)

    if isinstance(nested_steps, list):
        for child_step in nested_steps:
            if isinstance(child_step, dict):
                _append_step_nodes(nodes, edges, node_id, scope_node_id, child_step)
        if replan_node_ids:
            for child_step in nested_steps:
                if isinstance(child_step, dict):
                    child_node_id = _step_node_id(scope_node_id, child_step)
                    edges.append(
                        _edge(
                            f"{replan_node_ids[-1]}->{child_node_id}:expands_to",
                            replan_node_ids[-1],
                            child_node_id,
                            "expands_to",
                        )
                    )


def build_workflow_graph(
    *,
    task: str,
    task_shape: str,
    status: str,
    attempts: list[dict[str, Any]],
    selected_option: dict[str, Any] | None,
    result: Any,
    presentation: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    workflow_suffix = _stable_suffix(task, task_shape, status, len(attempts))
    workflow_node_id = f"workflow:{workflow_suffix}"
    nodes: list[dict[str, Any]] = [
        {
            "node_id": workflow_node_id,
            "kind": "workflow",
            "role": "workflow",
            "label": task.strip() or "workflow",
            "task": task,
            "task_shape": task_shape,
            "status": status,
            "presentation": _compact_value(presentation or {}, text_limit=160, row_limit=3),
            "result": _compact_value(result, text_limit=220, row_limit=3),
            "run_id": run_id or None,
        }
    ]
    edges: list[dict[str, Any]] = []
    selected_option_id = ""
    if isinstance(selected_option, dict):
        selected_option_id = str(selected_option.get("id") or "").strip()

    for attempt_index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, dict):
            continue
        option = attempt.get("option", {}) if isinstance(attempt.get("option"), dict) else {}
        option_id = str(option.get("id") or f"option{attempt_index}").strip()
        attempt_node_id = f"{workflow_node_id}:attempt:{attempt_index}"
        nodes.append(
            {
                "node_id": attempt_node_id,
                "kind": "attempt",
                "role": "router",
                "label": option.get("label") or option_id,
                "attempt": attempt_index,
                "status": attempt.get("status", "unknown"),
                "selected": option_id == selected_option_id,
                "option": _compact_value(option, text_limit=180, row_limit=3),
                "result": _compact_value(attempt.get("result"), text_limit=220, row_limit=3),
                "error": _compact_text(attempt.get("error"), 220),
            }
        )
        edges.append(
            _edge(
                f"{workflow_node_id}->{attempt_node_id}:attempt",
                workflow_node_id,
                attempt_node_id,
                "attempt",
            )
        )
        previous_attempt_node_id = f"{workflow_node_id}:attempt:{attempt_index - 1}" if attempt_index > 1 else ""
        if previous_attempt_node_id:
            edges.append(
                _edge(
                    f"{previous_attempt_node_id}->{attempt_node_id}:next_attempt",
                    previous_attempt_node_id,
                    attempt_node_id,
                    "next_attempt",
                )
            )

        current_node_id = _append_validation_node(nodes, edges, attempt_node_id, attempt.get("validation"), f"workflow_attempt_{attempt_index}")
        current_node_id, _ = _append_router_nodes(nodes, edges, current_node_id, attempt, f"workflow_attempt_{attempt_index}")
        current_node_id, replan_node_ids = _append_replan_nodes(nodes, edges, current_node_id, attempt, f"workflow_attempt_{attempt_index}")
        _append_clarification_node(nodes, edges, current_node_id, attempt.get("clarification"), f"workflow_attempt_{attempt_index}")

        derived_from_attempt = option.get("derived_from_attempt")
        if isinstance(derived_from_attempt, int) and derived_from_attempt > 0:
            source_attempt_node_id = f"{workflow_node_id}:attempt:{derived_from_attempt}"
            source_replan_node_id = f"{source_attempt_node_id}:validator:router:replan"
            if any(node.get("node_id") == source_replan_node_id for node in nodes):
                edges.append(
                    _edge(
                        f"{source_replan_node_id}->{attempt_node_id}:activates",
                        source_replan_node_id,
                        attempt_node_id,
                        "activates",
                    )
                )
            else:
                edges.append(
                    _edge(
                        f"{source_attempt_node_id}->{attempt_node_id}:derived_from",
                        source_attempt_node_id,
                        attempt_node_id,
                        "derived_from",
                    )
                )
        for step in attempt.get("steps", []):
            if isinstance(step, dict):
                _append_step_nodes(nodes, edges, attempt_node_id, attempt_node_id, step)

    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "kind": "workflow_execution",
        "root_node_id": workflow_node_id,
        "run_id": run_id or None,
        "task": task,
        "task_shape": task_shape,
        "status": status,
        "selected_option_id": selected_option_id or None,
        "nodes": nodes,
        "edges": edges,
        "statistics": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "attempt_count": len([item for item in nodes if item.get("kind") == "attempt"]),
            "step_count": len([item for item in nodes if item.get("kind") in {"step", "group_step"}]),
            "validator_count": len([item for item in nodes if item.get("kind") == "validator"]),
            "reducer_count": len([item for item in nodes if item.get("kind") == "reducer"]),
            "router_count": len([item for item in nodes if item.get("kind") == "router"]),
            "replan_count": len([item for item in nodes if item.get("kind") == "replan"]),
            "clarification_count": len([item for item in nodes if item.get("kind") == "clarification"]),
        },
    }
