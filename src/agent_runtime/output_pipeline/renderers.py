"""Deterministic renderers built on typed result shapes."""

from __future__ import annotations

import json
from typing import Any

from agent_runtime.core.types import DisplayPlan, RenderedOutput
from agent_runtime.output_pipeline.result_shapes import (
    AggregateResult,
    CapabilityListResult,
    DirectoryListingResult,
    ErrorResult,
    FileContentResult,
    MultiSectionResult,
    ProcessListResult,
    RecordListResult,
    ResultShape,
    ScalarResult,
    TableResult,
    TextResult,
)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    """Render row dictionaries as a Markdown table."""

    if not rows:
        return "_No rows available._"
    headers = columns or list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row.get(header, "")) for header in headers) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _code_block(body: str, language: str = "text") -> str:
    """Wrap plain text in a fenced code block."""

    return f"```{language}\n{body}\n```"


def _with_title(title: str | None, body: str, level: str = "##") -> str:
    """Prefix body text with an optional heading."""

    if not title:
        return body
    return f"{level} {title}\n\n{body}"


def render_text_result(
    result: TextResult,
    *,
    title: str | None = None,
    display_type: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> str:
    """Render a normalized text result."""

    resolved_title = title or result.title
    if display_type == "code_block":
        language = str((parameters or {}).get("language") or "text")
        return _with_title(resolved_title, _code_block(result.text, language))
    return _with_title(resolved_title, result.text)


def render_table_result(result: TableResult, *, title: str | None = None) -> str:
    """Render a normalized table result."""

    return _with_title(title or result.title, _markdown_table(result.rows))


def render_record_list_result(
    result: RecordListResult,
    *,
    title: str | None = None,
    columns: list[str] | None = None,
) -> str:
    """Render a normalized record-list result."""

    return _with_title(title or result.title, _markdown_table(result.records, columns))


def render_scalar_result(result: ScalarResult, *, title: str | None = None) -> str:
    """Render a normalized scalar result."""

    label = result.label or title or result.title or "Value"
    unit_suffix = f" {result.unit}" if result.unit else ""
    body = f"{label}: {result.value}{unit_suffix}"
    return _with_title(title or result.title, body) if title or result.title else body


def render_aggregate_result(result: AggregateResult, *, title: str | None = None) -> str:
    """Render a normalized aggregate result as a scalar-style summary."""

    label = result.label or title or result.title or "Aggregate"
    unit_suffix = f" {result.unit}" if result.unit else ""
    value_line = f"{label}: {result.value}{unit_suffix}"
    details: list[str] = []
    if result.field:
        details.append(f"operation: {result.operation}")
        details.append(f"field: {result.field}")
    elif result.operation:
        details.append(f"operation: {result.operation}")
    details.append(f"rows considered: {result.row_count}")
    details.append(f"values used: {result.used_count}")
    if result.skipped_count > 0:
        details.append(f"skipped values: {result.skipped_count}")
    body = "\n".join([value_line, *details])
    return _with_title(title or result.title, body) if title or result.title else body


def render_capability_list_result(result: CapabilityListResult, *, title: str | None = None) -> str:
    """Render capabilities grouped by domain."""

    blocks: list[str] = []
    for domain in sorted(result.grouped_capabilities):
        block_lines = [f"### {domain}"]
        capabilities = result.grouped_capabilities[domain]
        if capabilities and isinstance(capabilities[0], dict):
            rows = [dict(item) for item in capabilities if isinstance(item, dict)]
            block_lines.append(_markdown_table(rows))
        else:
            block_lines.extend(f"- {item}" for item in capabilities)
        blocks.append("\n".join(block_lines))
    body = "\n\n".join(blocks) if blocks else "_No capabilities available._"
    return _with_title(title or result.title, body)


def render_error_result(result: ErrorResult, *, title: str | None = None) -> str:
    """Render a normalized error result safely."""

    return _with_title(title or result.title, result.message)


def render_multi_section_result(result: MultiSectionResult) -> str:
    """Render a multi-section result by rendering each child shape."""

    blocks: list[str] = []
    if result.title:
        blocks.append(f"# {result.title}")
    for section in result.sections:
        blocks.append(render_result_shape(section))
    return "\n\n".join(block for block in blocks if block.strip())


def render_result_shape(
    result: ResultShape,
    *,
    title: str | None = None,
    display_type: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> str:
    """Render one normalized result shape deterministically."""

    if isinstance(result, MultiSectionResult):
        return render_multi_section_result(result)
    if isinstance(result, AggregateResult):
        return render_aggregate_result(result, title=title)
    if isinstance(result, ScalarResult):
        return render_scalar_result(result, title=title)
    if isinstance(result, CapabilityListResult):
        return render_capability_list_result(result, title=title)
    if isinstance(result, DirectoryListingResult):
        return render_record_list_result(
            RecordListResult(
                node_id=result.node_id,
                capability_id=result.capability_id,
                operation_id=result.operation_id,
                title=result.title,
                records=result.entries,
            ),
            title=title or result.title,
            columns=["name", "path", "type", "size", "modified_time"],
        )
    if isinstance(result, ProcessListResult):
        return render_record_list_result(
            RecordListResult(
                node_id=result.node_id,
                capability_id=result.capability_id,
                operation_id=result.operation_id,
                title=result.title,
                records=result.processes,
            ),
            title=title or result.title,
        )
    if isinstance(result, FileContentResult):
        return render_text_result(
            TextResult(
                node_id=result.node_id,
                capability_id=result.capability_id,
                operation_id=result.operation_id,
                title=result.title,
                text=result.content_preview,
            ),
            title=title or result.title,
            display_type=display_type or "code_block",
            parameters=parameters,
        )
    if isinstance(result, TableResult):
        return render_table_result(result, title=title)
    if isinstance(result, RecordListResult):
        return render_record_list_result(result, title=title)
    if isinstance(result, ErrorResult):
        return render_error_result(result, title=title)
    if isinstance(result, TextResult):
        return render_text_result(result, title=title, display_type=display_type, parameters=parameters)
    return render_text_result(
        TextResult(node_id=result.node_id, capability_id=result.capability_id, operation_id=result.operation_id, title=result.title, text=str(result)),
        title=title,
        display_type=display_type,
        parameters=parameters,
    )


def render_display_plan(
    display_plan: DisplayPlan,
    source_lookup: dict[str, dict[str, Any]],
) -> RenderedOutput:
    """Render a display plan against normalized source records."""

    def render_section(section: dict[str, Any]) -> str:
        source_node_id = section.get("source_node_id")
        source_data_ref = section.get("source_data_ref")
        if source_node_id is not None:
            record = source_lookup[f"node:{source_node_id}"]
        elif source_data_ref is not None:
            record = source_lookup[f"data:{source_data_ref}"]
        else:
            raise ValueError("Display plan section must reference source_node_id or source_data_ref.")
        shape = record["shape"]
        title = section.get("title") or getattr(shape, "title", None)
        parameters = section.get("parameters") if isinstance(section.get("parameters"), dict) else {}
        display_type = str(section.get("display_type") or display_plan.display_type)
        return render_result_shape(shape, title=title, display_type=display_type, parameters=parameters)

    if display_plan.display_type == "multi_section":
        section_shape = MultiSectionResult(
            node_id="multi-section",
            title=display_plan.title,
            sections=[
                source_lookup[
                    (
                        f"node:{section['source_node_id']}"
                        if section.get("source_node_id") is not None
                        else f"data:{section['source_data_ref']}"
                    )
                ]["shape"]
                for section in display_plan.sections
            ],
        )
        content = render_multi_section_result(section_shape)
        if display_plan.sections:
            rendered_sections = [render_section(section) for section in display_plan.sections]
            content = "\n\n".join(
                [f"# {display_plan.title}"] + rendered_sections if display_plan.title else rendered_sections
            )
    else:
        if display_plan.sections:
            content = render_section(display_plan.sections[0])
        else:
            first_source = next(iter(source_lookup.values()), None)
            if first_source is None:
                content = "No results available."
            else:
                content = render_result_shape(
                    first_source["shape"],
                    title=display_plan.title,
                    display_type=display_plan.display_type,
                    parameters={},
                )

    return RenderedOutput(content=content, display_plan=display_plan, metadata={})


class MarkdownRenderer:
    """Compatibility renderer wrapper for older code paths."""

    def render(self, display_plan: DisplayPlan, source_lookup: dict[str, dict[str, Any]], resolve_payload=None) -> RenderedOutput:
        """Render a display plan using typed result shapes."""

        _ = resolve_payload
        return render_display_plan(display_plan, source_lookup)
