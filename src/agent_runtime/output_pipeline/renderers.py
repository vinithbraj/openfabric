"""Deterministic renderers for composed output."""

from __future__ import annotations

import json
from typing import Any

from agent_runtime.core.types import DisplayPlan, RenderedOutput


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    """Extract row-shaped data from a payload when possible."""

    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return [dict(item) for item in payload]
    if isinstance(payload, dict):
        if isinstance(payload.get("entries"), list):
            return [dict(item) for item in payload["entries"] if isinstance(item, dict)]
        if isinstance(payload.get("rows"), list):
            return [dict(item) for item in payload["rows"] if isinstance(item, dict)]
        if isinstance(payload.get("matches"), list):
            return [{"path": item} for item in payload["matches"]]
    return []


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render row dictionaries as a Markdown table."""

    if not rows:
        return "_No rows available._"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row.get(header, "")) for header in headers) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _plain_text_from_payload(payload: Any) -> str:
    """Render payload content as concise plain text."""

    if isinstance(payload, dict):
        if "content_preview" in payload:
            return str(payload["content_preview"])
        if "markdown" in payload:
            return str(payload["markdown"])
        if "matches" in payload:
            return "\n".join(str(item) for item in payload["matches"])
        if "entries" in payload:
            return "\n".join(str(item.get("path") or item.get("name")) for item in payload["entries"])
        if "rows" in payload:
            return json.dumps(payload["rows"], indent=2, default=str)
        return "\n".join(f"{key}: {value}" for key, value in payload.items())
    if isinstance(payload, list):
        return "\n".join(str(item) for item in payload)
    return str(payload)


def _code_block_from_payload(payload: Any, language: str = "text") -> str:
    """Render payload inside a fenced code block."""

    if isinstance(payload, dict) and "content_preview" in payload:
        body = str(payload["content_preview"])
    elif isinstance(payload, dict) and "markdown" in payload:
        body = str(payload["markdown"])
    elif isinstance(payload, str):
        body = payload
    else:
        body = json.dumps(payload, indent=2, default=str)
    return f"```{language}\n{body}\n```"


def render_plain_text(title: str | None, payload: Any) -> str:
    """Render payload as plain text."""

    prefix = f"{title}\n\n" if title else ""
    return prefix + _plain_text_from_payload(payload)


def render_markdown(title: str | None, payload: Any) -> str:
    """Render payload as Markdown."""

    prefix = f"## {title}\n\n" if title else ""
    if isinstance(payload, dict) and "markdown" in payload:
        return prefix + str(payload["markdown"])
    rows = _rows_from_payload(payload)
    if rows:
        return prefix + _markdown_table(rows)
    return prefix + _plain_text_from_payload(payload)


def render_table(title: str | None, payload: Any) -> str:
    """Render payload explicitly as a Markdown table."""

    prefix = f"## {title}\n\n" if title else ""
    return prefix + _markdown_table(_rows_from_payload(payload))


def render_json(title: str | None, payload: Any) -> str:
    """Render payload as formatted JSON."""

    prefix = f"## {title}\n\n" if title else ""
    return prefix + json.dumps(payload, indent=2, default=str)


def render_code_block(title: str | None, payload: Any, parameters: dict[str, Any] | None = None) -> str:
    """Render payload as a fenced code block."""

    prefix = f"## {title}\n\n" if title else ""
    language = str((parameters or {}).get("language") or "text")
    return prefix + _code_block_from_payload(payload, language)


def render_multi_section(
    title: str | None,
    sections: list[dict[str, Any]],
    source_lookup: dict[str, dict[str, Any]],
    render_section,
) -> str:
    """Render a multi-section display plan by delegating each section."""

    blocks: list[str] = []
    if title:
        blocks.append(f"# {title}")
    for section in sections:
        blocks.append(render_section(section, source_lookup))
    return "\n\n".join(block for block in blocks if block.strip())


def render_display_plan(
    display_plan: DisplayPlan,
    source_lookup: dict[str, dict[str, Any]],
    resolve_payload,
) -> RenderedOutput:
    """Render a display plan against resolved source payloads."""

    def render_one(display_type: str, title: str | None, payload: Any, parameters: dict[str, Any] | None = None) -> str:
        if display_type == "plain_text":
            return render_plain_text(title, payload)
        if display_type == "markdown":
            return render_markdown(title, payload)
        if display_type == "table":
            return render_table(title, payload)
        if display_type == "json":
            return render_json(title, payload)
        if display_type == "code_block":
            return render_code_block(title, payload, parameters)
        return render_markdown(title, payload)

    def render_section(section: dict[str, Any], lookup: dict[str, dict[str, Any]]) -> str:
        section_type = str(section.get("display_type") or display_plan.display_type)
        payload = resolve_payload(section, lookup)
        title = section.get("title")
        parameters = section.get("parameters") if isinstance(section.get("parameters"), dict) else {}
        return render_one(section_type, title, payload, parameters)

    if display_plan.display_type == "multi_section":
        content = render_multi_section(display_plan.title, display_plan.sections, source_lookup, render_section)
    else:
        if display_plan.sections:
            section = display_plan.sections[0]
            payload = resolve_payload(section, source_lookup)
            parameters = section.get("parameters") if isinstance(section.get("parameters"), dict) else {}
            title = section.get("title") or display_plan.title
        else:
            first_source = next(iter(source_lookup.values()), {"payload": {"message": "No results available."}})
            payload = first_source["payload"]
            parameters = {}
            title = display_plan.title
        content = render_one(display_plan.display_type, title, payload, parameters)

    return RenderedOutput(content=content, display_plan=display_plan, metadata={})


class MarkdownRenderer:
    """Compatibility renderer wrapper for older code paths."""

    def render(self, display_plan: DisplayPlan, source_lookup: dict[str, dict[str, Any]], resolve_payload) -> RenderedOutput:
        """Render a display plan using the new deterministic renderer set."""

        return render_display_plan(display_plan, source_lookup, resolve_payload)
