"""Open WebUI-friendly event formatting."""

from __future__ import annotations

import json
from typing import Any

from agent_runtime.observability.events import PipelineEvent

_SCALAR_FIRST_KEYS = [
    "request_id",
    "prompt_type",
    "prompt_preview",
    "task_id",
    "node_id",
    "dag_id",
    "capability_id",
    "operation_id",
    "status",
    "final_status",
    "display_type",
    "data_type",
    "llm_fits",
    "llm_confidence",
    "llm_error_kind",
    "llm_primary_failure_mode",
    "error_class",
    "requires_confirmation",
    "allowed",
    "fit_count",
    "gap_count",
    "task_count",
    "node_count",
    "edge_count",
    "result_count",
    "content_length",
]

_INLINE_CODE_KEYS = {
    "allowed",
    "capability_id",
    "consumer_argument_name",
    "consumer_task_id",
    "dag_id",
    "data_type",
    "display_type",
    "error_class",
    "event_type",
    "failed_stage",
    "final_status",
    "llm_error_kind",
    "llm_fits",
    "llm_primary_failure_mode",
    "node_id",
    "object_type",
    "operation_id",
    "output_key",
    "path",
    "producer_output_key",
    "producer_task_id",
    "prompt_type",
    "request_id",
    "requires_confirmation",
    "risk_level",
    "semantic_verb",
    "source_node_id",
    "stage",
    "status",
    "suggested_domain",
    "suggested_object_type",
    "target_node_id",
    "task_id",
}

_PROSE_KEYS = {
    "message",
    "prompt_preview",
    "reason",
    "safe_summary",
    "summary",
    "unresolved_reason",
}

_SENTENCE_LIST_KEYS = {
    "assumptions",
    "blocked_reasons",
    "dataflow_warnings",
    "dependency_warnings",
    "deterministic_rejections",
    "missing_required_arguments",
    "missing_user_intents",
    "output_expectation_warnings",
    "reasons",
    "suspicious_nodes",
    "unresolved_dataflows",
    "unresolved_references",
    "warnings",
}


def _pretty_stage(stage: str) -> str:
    return str(stage or "").replace("_", " ").strip().title() or "Pipeline"


def _pretty_key(key: str) -> str:
    special = {
        "capability_id": "Capability",
        "consumer_argument_name": "Consumer Argument",
        "consumer_task_id": "Consumer Task",
        "content_length": "Content Length",
        "dag_id": "DAG",
        "data_type": "Data Type",
        "display_type": "Display Type",
        "edge_count": "Edge Count",
        "error_class": "Error Class",
        "error_type": "Error Type",
        "final_status": "Final Status",
        "fit_count": "Fit Count",
        "gap_count": "Gap Count",
        "llm_confidence": "LLM Confidence",
        "llm_error_kind": "LLM Error Kind",
        "llm_error_message": "LLM Error Message",
        "llm_fits": "LLM Fits",
        "llm_primary_failure_mode": "LLM Primary Failure Mode",
        "node_count": "Node Count",
        "node_id": "Node",
        "normalized_likely_domains": "Normalized Likely Domains",
        "normalized_manifest_domain": "Normalized Capability Domain",
        "normalized_manifest_object_types": "Normalized Capability Object Types",
        "normalized_task_domain": "Normalized Task Domain",
        "normalized_task_object_type": "Normalized Task Object Type",
        "operation_id": "Operation",
        "producer_output_key": "Producer Output Key",
        "producer_task_id": "Producer Task",
        "prompt_preview": "Prompt Preview",
        "prompt_type": "Prompt Type",
        "request_id": "Request",
        "requires_confirmation": "Requires Confirmation",
        "result_count": "Result Count",
        "safe_preview_count": "Safe Preview Count",
        "shape_types": "Shape Types",
        "source_node_id": "Source Node",
        "suggested_domain": "Suggested Domain",
        "suggested_object_type": "Suggested Object Type",
        "target_node_id": "Target Node",
        "task_count": "Task Count",
        "task_id": "Task",
    }
    if key in special:
        return special[key]
    return str(key or "").replace("_", " ").strip().title() or "Detail"


def _ordered_detail_keys(details: dict[str, Any]) -> list[str]:
    priority = {key: index for index, key in enumerate(_SCALAR_FIRST_KEYS)}
    return sorted(details.keys(), key=lambda key: (priority.get(key, len(priority)), key))


def _format_code(value: Any) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif value is None:
        text = "null"
    else:
        text = str(value)
    text = text.replace("`", "'")
    return f"`{text}`"


def _format_scalar(key: str, value: Any) -> str:
    if isinstance(value, bool | int | float):
        return _format_code(value)
    text = str(value)
    if key in _PROSE_KEYS:
        return text
    if key in _INLINE_CODE_KEYS or key.endswith("_id") or key.endswith("_type"):
        return _format_code(text)
    if text.startswith("node::") or "." in text and " " not in text:
        return _format_code(text)
    return text


def _render_sentence_list(key: str, values: list[Any]) -> list[str]:
    if not values:
        return []
    lines = [f"**{_pretty_key(key)}**"]
    for value in values:
        if value is None:
            continue
        lines.append(f"- {value}")
    return lines


def _render_code_list(key: str, values: list[Any]) -> list[str]:
    if not values:
        return []
    lines = [f"**{_pretty_key(key)}**"]
    for value in values:
        if value is None:
            continue
        lines.append(f"- {_format_code(value)}")
    return lines


def _render_tasks(values: list[dict[str, Any]]) -> list[str]:
    if not values:
        return []
    lines = ["**Tasks**"]
    for task in values:
        task_id = _format_code(task.get("task_id") or task.get("id") or "task")
        description = str(task.get("description") or "No description provided.")
        depends_on = task.get("depends_on") or task.get("dependencies") or []
        lines.append(f"- {task_id}")
        lines.append(f"  - Details: {_format_code(description)}")
        if depends_on:
            lines.append(
                f"  - Depends On: {', '.join(_format_code(dep) for dep in depends_on)}"
            )
    return lines


def _render_candidates(values: list[dict[str, Any]]) -> list[str]:
    if not values:
        return []
    lines = ["**Candidates**"]
    for candidate in values:
        capability_id = _format_code(candidate.get("capability_id") or "unknown")
        operation_id = _format_code(candidate.get("operation_id") or "unknown")
        confidence = candidate.get("confidence")
        reason = str(candidate.get("reason") or "").strip()
        line = f"- {capability_id} via {operation_id}"
        if confidence is not None:
            line += f" - confidence {_format_code(confidence)}"
        if reason:
            line += f". {reason}"
        lines.append(line)
    return lines


def _render_nodes(values: list[dict[str, Any]]) -> list[str]:
    if not values:
        return []
    lines = ["**Nodes**"]
    for node in values:
        node_id = _format_code(node.get("node_id") or "node")
        capability_id = _format_code(node.get("capability_id") or "unknown")
        operation_id = _format_code(node.get("operation_id") or "unknown")
        depends_on = node.get("depends_on") or []
        line = f"- {node_id} runs {capability_id} via {operation_id}"
        if depends_on:
            line += f" (depends on: {', '.join(_format_code(dep) for dep in depends_on)})"
        lines.append(line)
    return lines


def _render_sections(values: list[dict[str, Any]]) -> list[str]:
    if not values:
        return []
    lines = ["**Sections**"]
    for section in values:
        title = str(section.get("title") or "Untitled section")
        display_type = section.get("display_type")
        source_node_id = section.get("source_node_id")
        line = f"- {title}"
        if display_type:
            line += f" - {_format_code(display_type)}"
        if source_node_id:
            line += f" from {_format_code(source_node_id)}"
        lines.append(line)
    return lines


def _render_dict_block(key: str, value: dict[str, Any]) -> list[str]:
    if not value:
        return []
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
    return [
        f"**{_pretty_key(key)}**",
        "```json",
        payload,
        "```",
    ]


def _render_list_block(key: str, value: list[Any]) -> list[str]:
    if not value:
        return []
    if all(isinstance(item, dict) for item in value):
        dict_values = [item for item in value if isinstance(item, dict)]
        if key == "tasks":
            return _render_tasks(dict_values)
        if key == "candidates":
            return _render_candidates(dict_values)
        if key == "nodes":
            return _render_nodes(dict_values)
        if key == "sections":
            return _render_sections(dict_values)
        payload = json.dumps(dict_values, indent=2, sort_keys=True, ensure_ascii=True)
        return [
            f"**{_pretty_key(key)}**",
            "```json",
            payload,
            "```",
        ]
    if all(not isinstance(item, dict | list) for item in value):
        if key in _SENTENCE_LIST_KEYS:
            return _render_sentence_list(key, value)
        return _render_code_list(key, value)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, default=str)
    return [
        f"**{_pretty_key(key)}**",
        "```json",
        payload,
        "```",
    ]


def _render_detail_blocks(details: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    scalar_lines: list[str] = []
    sections: list[list[str]] = []
    for key in _ordered_detail_keys(details):
        value = details[key]
        if value is None or value == "" or value == [] or value == {}:
            continue
        if isinstance(value, list):
            section = _render_list_block(key, value)
            if section:
                sections.append(section)
            continue
        if isinstance(value, dict):
            section = _render_dict_block(key, value)
            if section:
                sections.append(section)
            continue
        if key in _PROSE_KEYS:
            sections.append([f"**{_pretty_key(key)}**", str(value)])
            continue
        scalar_lines.append(f"- {_pretty_key(key)}: {_format_scalar(key, value)}")
    return scalar_lines, sections


def _blockquote(lines: list[str]) -> list[str]:
    """Render markdown lines inside a blockquote for a left-side rail."""

    quoted: list[str] = []
    for line in lines:
        if line:
            quoted.append(f"> {line}")
        else:
            quoted.append(">")
    return quoted


def format_event_for_openwebui(event: PipelineEvent) -> str:
    """Render one pipeline event as polished markdown for Open WebUI streaming."""

    lines = [f"**{event.title}**", "", event.summary]
    scalar_lines, sections = _render_detail_blocks(event.details)
    if scalar_lines:
        lines.extend(["", *scalar_lines])
    for section in sections:
        lines.extend(["", *section])
    body = _blockquote(lines)
    return "\n".join([f"### {_pretty_stage(event.stage)}", "---", *body]).strip() + "\n\n"
