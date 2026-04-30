"""OpenFABRIC Runtime Module: aor_runtime.tools.text_format

Purpose:
    Implement deterministic local data formatting.

Responsibilities:
    Format rows, dictionaries, shell tables, CSV, text, and Markdown without sending raw data to the LLM.

Data flow / Interfaces:
    Receives resolved tool outputs and returns formatted content plus metadata.

Boundaries:
    Must preserve large-output policy by cooperating with auto-artifact and presentation thresholds.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Literal

from pydantic import Field

from aor_runtime.core.contracts import ToolSpec
from aor_runtime.runtime.output_envelope import parse_shell_table
from aor_runtime.tools.base import BaseTool, ToolArgsModel, ToolResultModel


OutputFormat = Literal["txt", "csv", "json", "markdown"]


def format_data(
    source: Any,
    output_format: OutputFormat = "txt",
    *,
    query_used: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Format data for the surrounding runtime workflow.

    Inputs:
        Receives source, output_format, query_used, output_path for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.text_format.format_data.
    """
    rows = _coerce_rows(source)
    columns = _columns_for_rows(rows)
    normalized_format = str(output_format or "txt").strip().lower()
    if normalized_format not in {"txt", "csv", "json", "markdown"}:
        normalized_format = "txt"

    if normalized_format == "csv":
        content = _rows_to_csv(rows, columns) if rows else _source_to_txt(source)
    elif normalized_format == "json":
        content = json.dumps(rows if rows else _coerce_json_source(source), ensure_ascii=False, indent=2, default=str)
    elif normalized_format == "markdown":
        content = _rows_to_markdown(rows, columns) if rows else _source_to_markdown(source)
    else:
        content = _rows_to_txt(rows, columns) if rows else _source_to_txt(source)

    return {
        "content": content,
        "format": normalized_format,
        "row_count": len(rows) if rows else _source_count(source),
        "columns": columns,
        "query_used": str(query_used or ""),
        "output_path": str(output_path or ""),
    }


class TextFormatTool(BaseTool):
    """Represent text format tool within the OpenFABRIC runtime. It extends BaseTool.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by TextFormatTool.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.text_format.TextFormatTool and related tests.
    """
    class ToolArgs(ToolArgsModel):
        """Represent tool args within the OpenFABRIC runtime. It extends ToolArgsModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolArgs.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.text_format.ToolArgs and related tests.
        """
        source: Any
        format: OutputFormat = "txt"
        query_used: str | None = None
        output_path: str | None = None

    class ToolResult(ToolResultModel):
        """Represent tool result within the OpenFABRIC runtime. It extends ToolResultModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolResult.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.text_format.ToolResult and related tests.
        """
        content: str
        format: str
        row_count: int
        columns: list[str] = Field(default_factory=list)
        query_used: str = ""
        output_path: str = ""

    def __init__(self) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through TextFormatTool.__init__ calls and related tests.
        """
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="text.format",
            description="Deterministically format structured rows, lists, or scalar values as txt, csv, json, or markdown.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "source": {},
                    "format": {"type": "string", "enum": ["txt", "csv", "json", "markdown"]},
                    "query_used": {"type": ["string", "null"]},
                    "output_path": {"type": ["string", "null"]},
                },
                "required": ["source"],
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        """Run for TextFormatTool instances.

        Inputs:
            Receives arguments for this TextFormatTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through TextFormatTool.run calls and related tests.
        """
        return self.ToolResult.model_validate(
            format_data(
                arguments.source,
                arguments.format,
                query_used=arguments.query_used,
                output_path=arguments.output_path,
            )
        )


def _coerce_rows(source: Any) -> list[dict[str, Any]]:
    """Handle the internal coerce rows helper path for this module.

    Inputs:
        Receives source for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.text_format._coerce_rows.
    """
    catalog = _extract_catalog(source)
    if catalog is not None:
        return _catalog_to_rows(catalog)
    if isinstance(source, str):
        return parse_shell_table(source)
    if isinstance(source, dict) and isinstance(source.get("rows"), list):
        return _coerce_rows(source["rows"])
    if isinstance(source, dict):
        if _is_simple_mapping(source):
            return [{"field": str(key), "value": value} for key, value in source.items()]
        return []
    if isinstance(source, list):
        if not source:
            return []
        if all(isinstance(item, dict) for item in source):
            return [dict(item) for item in source]
        return [{"value": item} for item in source]
    return []


def _columns_for_rows(rows: list[dict[str, Any]]) -> list[str]:
    """Handle the internal columns for rows helper path for this module.

    Inputs:
        Receives rows for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.text_format._columns_for_rows.
    """
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            column = str(key)
            if column in seen:
                continue
            columns.append(column)
            seen.add(column)
    return columns


def _rows_to_txt(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Handle the internal rows to txt helper path for this module.

    Inputs:
        Receives rows, columns for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.text_format._rows_to_txt.
    """
    if not rows:
        return "No rows returned."
    if len(columns) == 1:
        key = columns[0]
        return "\n".join(_stringify(row.get(key, "")) for row in rows)
    lines = ["\t".join(columns)]
    lines.extend("\t".join(_stringify(row.get(column, "")) for column in columns) for row in rows)
    return "\n".join(lines)


def _rows_to_csv(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Handle the internal rows to csv helper path for this module.

    Inputs:
        Receives rows, columns for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.text_format._rows_to_csv.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue()


def _rows_to_markdown(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Handle the internal rows to markdown helper path for this module.

    Inputs:
        Receives rows, columns for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.text_format._rows_to_markdown.
    """
    if not columns:
        return "No rows returned."
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_markdown_cell(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def _source_to_txt(source: Any) -> str:
    """Handle the internal source to txt helper path for this module.

    Inputs:
        Receives source for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.text_format._source_to_txt.
    """
    catalog = _extract_catalog(source)
    if catalog is not None:
        return _catalog_to_txt(catalog)
    if source is None:
        return ""
    if isinstance(source, str):
        return source
    if isinstance(source, (int, float, bool)):
        return _stringify(source)
    if isinstance(source, dict):
        rows = _coerce_rows(source) or _mapping_to_rows(source)
        return _rows_to_txt(rows, _columns_for_rows(rows))
    if isinstance(source, list):
        rows = _coerce_rows(source)
        return _rows_to_txt(rows, _columns_for_rows(rows)) if rows else "No rows returned."
    return _stringify(source)


def _source_to_markdown(source: Any) -> str:
    """Handle the internal source to markdown helper path for this module.

    Inputs:
        Receives source for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.text_format._source_to_markdown.
    """
    catalog = _extract_catalog(source)
    if catalog is not None:
        rows = _catalog_to_rows(catalog)
        return _rows_to_markdown(rows, _columns_for_rows(rows))
    if isinstance(source, dict):
        rows = _coerce_rows(source) or _mapping_to_rows(source)
        if rows:
            return _rows_to_markdown(rows, _columns_for_rows(rows))
    if isinstance(source, list):
        rows = _coerce_rows(source)
        return _rows_to_markdown(rows, _columns_for_rows(rows)) if rows else "No rows returned."
    return _source_to_txt(source)


def _coerce_json_source(source: Any) -> Any:
    """Handle the internal coerce json source helper path for this module.

    Inputs:
        Receives source for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.text_format._coerce_json_source.
    """
    if isinstance(source, dict) and "rows" in source:
        return source.get("rows")
    catalog = _extract_catalog(source)
    if catalog is not None:
        return _catalog_to_rows(catalog)
    return source


def _source_count(source: Any) -> int:
    """Handle the internal source count helper path for this module.

    Inputs:
        Receives source for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.text_format._source_count.
    """
    if isinstance(source, (list, tuple, set)):
        return len(source)
    if isinstance(source, dict) and isinstance(source.get("rows"), list):
        return len(source["rows"])
    return 1 if source is not None else 0


def _stringify(value: Any) -> str:
    """Handle the internal stringify helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.text_format._stringify.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _escape_markdown_cell(value: Any) -> str:
    """Handle the internal escape markdown cell helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.text_format._escape_markdown_cell.
    """
    return _stringify(value).replace("|", "\\|").replace("\n", " ")


def _extract_catalog(source: Any) -> dict[str, Any] | None:
    """Handle the internal extract catalog helper path for this module.

    Inputs:
        Receives source for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.text_format._extract_catalog.
    """
    if not isinstance(source, dict):
        return None
    catalog = source.get("catalog") if isinstance(source.get("catalog"), dict) else source
    if isinstance(catalog, dict) and isinstance(catalog.get("tables"), list) and catalog.get("database"):
        return catalog
    return None


def _catalog_to_rows(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Handle the internal catalog to rows helper path for this module.

    Inputs:
        Receives catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.text_format._catalog_to_rows.
    """
    rows: list[dict[str, Any]] = []
    for table in list(catalog.get("tables") or []):
        if not isinstance(table, dict):
            continue
        columns = [str(column.get("name") or column.get("column_name") or "") for column in list(table.get("columns") or []) if isinstance(column, dict)]
        rows.append(
            {
                "database": catalog.get("database", ""),
                "schema": table.get("schema") or table.get("schema_name") or "",
                "table": table.get("table") or table.get("table_name") or "",
                "columns": ", ".join(column for column in columns if column),
            }
        )
    return rows


def _catalog_to_txt(catalog: dict[str, Any]) -> str:
    """Handle the internal catalog to txt helper path for this module.

    Inputs:
        Receives catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.text_format._catalog_to_txt.
    """
    rows = _catalog_to_rows(catalog)
    if not rows:
        return ""
    return "\n".join(
        f"{row['schema']}.{row['table']}: {row['columns']}".strip()
        for row in rows
    )


def _is_simple_mapping(source: dict[str, Any]) -> bool:
    """Handle the internal is simple mapping helper path for this module.

    Inputs:
        Receives source for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.text_format._is_simple_mapping.
    """
    return all(not isinstance(value, (dict, list, tuple, set)) for value in source.values())


def _mapping_to_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Handle the internal mapping to rows helper path for this module.

    Inputs:
        Receives source for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.text_format._mapping_to_rows.
    """
    return [{"field": str(key), "value": _compact_structured_value(value)} for key, value in source.items()]


def _compact_structured_value(value: Any) -> str:
    """Handle the internal compact structured value helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.text_format._compact_structured_value.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if not value:
            return "empty"
        simple_items: list[str] = []
        for index, (key, nested) in enumerate(value.items()):
            if index >= 8:
                simple_items.append("...")
                break
            if isinstance(nested, (dict, list, tuple, set)):
                simple_items.append(f"{key}={_compact_structured_value(nested)}")
            else:
                simple_items.append(f"{key}={_stringify(nested)}")
        return ", ".join(simple_items)
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if not items:
            return "0 items"
        if all(not isinstance(item, (dict, list, tuple, set)) for item in items):
            preview = ", ".join(_stringify(item) for item in items[:8])
            return f"{preview}, ..." if len(items) > 8 else preview
        return f"{len(items)} items"
    return _stringify(value)
