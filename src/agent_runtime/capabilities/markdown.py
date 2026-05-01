"""Markdown rendering capability for stored structured data."""

from __future__ import annotations

import json
from typing import Any

from agent_runtime.capabilities.base import BaseCapability
from agent_runtime.capabilities.schemas import CapabilityManifest
from agent_runtime.core.errors import ValidationError
from agent_runtime.core.types import DataRef, ExecutionResult

RENDER_TYPES = {"table", "summary", "code_block", "list"}


class MarkdownRenderCapability(BaseCapability):
    """Render stored structured data into Markdown."""

    manifest = CapabilityManifest(
        capability_id="markdown.render",
        domain="presentation",
        operation_id="render",
        name="Render Markdown",
        description="Render structured content into Markdown text.",
        semantic_verbs=["render", "summarize"],
        object_types=["report", "markdown", "document"],
        argument_schema={
            "input_ref": {"type": "string"},
            "render_type": {"type": "string"},
            "parameters": {"type": "object"},
        },
        required_arguments=["input_ref", "render_type", "parameters"],
        optional_arguments=[],
        output_schema={"markdown": {"type": "string"}},
        execution_backend="internal",
        backend_operation="markdown.render",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[
            {
                "arguments": {
                    "input_ref": "data-demo",
                    "render_type": "table",
                    "parameters": {"title": "Summary"},
                }
            }
        ],
        safety_notes=["Presentation-only capability."],
    )

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ExecutionResult:
        """Render stored data to Markdown using a fixed rendering vocabulary."""

        validated = self.validate_arguments(dict(arguments or {}))
        result_store = context.get("result_store")
        if result_store is None:
            raise ValidationError("result_store is required for markdown rendering.")

        input_value = validated["input_ref"]
        if isinstance(input_value, DataRef):
            input_data = result_store.get(input_value.ref_id)
        elif isinstance(input_value, str):
            input_data = result_store.get(input_value)
        else:
            input_data = input_value
        render_type = str(validated["render_type"])
        parameters = dict(validated["parameters"] or {})
        if render_type not in RENDER_TYPES:
            raise ValidationError(f"unsupported markdown render_type: {render_type}")

        if render_type == "table":
            markdown = _render_table(input_data, parameters)
        elif render_type == "summary":
            markdown = _render_summary(input_data, parameters)
        elif render_type == "code_block":
            markdown = _render_code_block(input_data, parameters)
        else:
            markdown = _render_list(input_data, parameters)

        return ExecutionResult(
            node_id=str(context.get("node_id") or ""),
            status="success",
            data_preview={"markdown": markdown},
            metadata={"render_type": render_type},
        )


def _render_title(parameters: dict[str, Any]) -> str:
    """Render an optional Markdown heading."""

    title = str(parameters.get("title") or "").strip()
    return f"## {title}\n\n" if title else ""


def _rows_from_input(input_data: Any) -> list[dict[str, Any]]:
    """Extract row dictionaries from stored input when possible."""

    if isinstance(input_data, list) and all(isinstance(item, dict) for item in input_data):
        return [dict(item) for item in input_data]
    if isinstance(input_data, dict):
        if isinstance(input_data.get("rows"), list):
            rows = input_data["rows"]
        elif isinstance(input_data.get("entries"), list):
            rows = input_data["entries"]
        else:
            rows = None
        if rows is not None and all(isinstance(item, dict) for item in rows):
            return [dict(item) for item in rows]
    raise ValidationError("table render requires row-oriented input.")


def _render_table(input_data: Any, parameters: dict[str, Any]) -> str:
    """Render row data as a Markdown table."""

    rows = _rows_from_input(input_data)
    title = _render_title(parameters)
    if not rows:
        return title + "_No rows available._"

    headers = list(rows[0].keys())
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(str(row.get(header, "")) for header in headers) + " |"
        for row in rows
    ]
    return title + "\n".join([header_line, separator, *body])


def _render_summary(input_data: Any, parameters: dict[str, Any]) -> str:
    """Render a compact Markdown summary of stored data."""

    title = _render_title(parameters)
    if isinstance(input_data, dict):
        lines = [f"- **{key}**: {value}" for key, value in input_data.items()]
    elif isinstance(input_data, list):
        lines = [f"- Item count: {len(input_data)}"]
    else:
        lines = [str(input_data)]
    return title + "\n".join(lines)


def _render_code_block(input_data: Any, parameters: dict[str, Any]) -> str:
    """Render input data inside a fenced code block."""

    title = _render_title(parameters)
    language = str(parameters.get("language") or "text")
    if isinstance(input_data, str):
        body = input_data
    else:
        body = json.dumps(input_data, indent=2, default=str)
    return title + f"```{language}\n{body}\n```"


def _render_list(input_data: Any, parameters: dict[str, Any]) -> str:
    """Render stored data as a Markdown bullet list."""

    title = _render_title(parameters)
    items: list[str]
    if isinstance(input_data, list):
        items = [str(item) for item in input_data]
    elif isinstance(input_data, dict):
        items = [f"{key}: {value}" for key, value in input_data.items()]
    else:
        items = [str(input_data)]
    return title + "\n".join(f"- {item}" for item in items)
