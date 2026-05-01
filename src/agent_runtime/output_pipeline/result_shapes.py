"""Typed result-shape normalization for deterministic rendering."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.core.types import ExecutionResult


class _BaseResultShape(BaseModel):
    """Shared shape metadata."""

    model_config = ConfigDict(extra="forbid")

    shape_type: str
    node_id: str
    capability_id: str | None = None
    operation_id: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextResult(_BaseResultShape):
    shape_type: Literal["text"] = "text"
    text: str


class TableResult(_BaseResultShape):
    shape_type: Literal["table"] = "table"
    rows: list[dict[str, Any]] = Field(default_factory=list)


class RecordListResult(_BaseResultShape):
    shape_type: Literal["record_list"] = "record_list"
    records: list[dict[str, Any]] = Field(default_factory=list)


class ScalarResult(_BaseResultShape):
    shape_type: Literal["scalar"] = "scalar"
    value: Any = None
    label: str | None = None
    unit: str | None = None


class AggregateResult(_BaseResultShape):
    shape_type: Literal["aggregate"] = "aggregate"
    operation: str
    field: str | None = None
    value: Any = None
    unit: str | None = None
    row_count: int = 0
    used_count: int = 0
    skipped_count: int = 0
    label: str | None = None


class FileContentResult(_BaseResultShape):
    shape_type: Literal["file_content"] = "file_content"
    path: str | None = None
    content_preview: str
    truncated: bool = False


class DirectoryListingResult(_BaseResultShape):
    shape_type: Literal["directory_listing"] = "directory_listing"
    path: str | None = None
    entries: list[dict[str, Any]] = Field(default_factory=list)
    truncated: bool = False


class ProcessListResult(_BaseResultShape):
    shape_type: Literal["process_list"] = "process_list"
    processes: list[dict[str, Any]] = Field(default_factory=list)
    pattern: str | None = None
    truncated: bool = False


class CapabilityListResult(_BaseResultShape):
    shape_type: Literal["capability_list"] = "capability_list"
    grouped_capabilities: dict[str, list[Any]] = Field(default_factory=dict)
    capabilities: list[dict[str, Any]] = Field(default_factory=list)
    capability_count: int = 0


class ErrorResult(_BaseResultShape):
    shape_type: Literal["error"] = "error"
    message: str


class MultiSectionResult(_BaseResultShape):
    shape_type: Literal["multi_section"] = "multi_section"
    sections: list["_RenderableResultShape"] = Field(default_factory=list)


_RenderableResultShape: TypeAlias = (
    TextResult
    | TableResult
    | RecordListResult
    | ScalarResult
    | AggregateResult
    | FileContentResult
    | DirectoryListingResult
    | ProcessListResult
    | CapabilityListResult
    | ErrorResult
)

ResultShape: TypeAlias = _RenderableResultShape | MultiSectionResult


def _sanitize_error_message(message: str | None) -> str:
    """Collapse raw traceback-like text into a safe user-facing message."""

    text = str(message or "").strip()
    if not text:
        return "Execution failed."
    if "Traceback (most recent call last)" in text:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            if ":" in line and not line.startswith("Traceback"):
                return line
        return "Execution failed."
    return text.splitlines()[0].strip()


def _payload_for_normalization(result: ExecutionResult, result_store=None) -> Any:
    """Resolve the most useful payload available for normalization."""

    if result_store is not None and result.data_ref is not None:
        try:
            return result_store.get(result.data_ref.ref_id)
        except Exception:
            pass
    if result.data_preview is not None:
        return result.data_preview
    return None


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    """Extract row-like dictionaries from supported payload families."""

    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return [dict(item) for item in payload]
    if not isinstance(payload, dict):
        return []
    for key in ("rows", "entries", "processes", "listeners"):
        value = payload.get(key)
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return [dict(item) for item in value]
    matches = payload.get("matches")
    if isinstance(matches, list):
        return [{"path": item} for item in matches]
    return []


def normalize_execution_result(result: ExecutionResult, result_store=None) -> ResultShape:
    """Normalize one execution result into a semantic result shape."""

    capability_id = str(result.metadata.get("capability_id") or "").strip() or None
    operation_id = str(result.metadata.get("operation_id") or "").strip() or None
    payload = _payload_for_normalization(result, result_store)
    title = None
    if isinstance(payload, dict):
        title = str(payload.get("title") or "").strip() or None

    if result.status != "success":
        return ErrorResult(
            node_id=result.node_id,
            capability_id=capability_id,
            operation_id=operation_id,
            title=title,
            message=_sanitize_error_message(result.error),
        )

    if capability_id == "filesystem.list_directory" and isinstance(payload, dict):
        return DirectoryListingResult(
            node_id=result.node_id,
            capability_id=capability_id,
            operation_id=operation_id,
            title=title,
            path=payload.get("path"),
            entries=_rows_from_payload(payload),
            truncated=bool(payload.get("truncated", False)),
        )

    if capability_id == "filesystem.search_files":
        return RecordListResult(
            node_id=result.node_id,
            capability_id=capability_id,
            operation_id=operation_id,
            title=title,
            records=_rows_from_payload(payload),
        )

    if capability_id == "filesystem.read_file" and isinstance(payload, dict):
        return FileContentResult(
            node_id=result.node_id,
            capability_id=capability_id,
            operation_id=operation_id,
            title=title,
            path=payload.get("path"),
            content_preview=str(payload.get("content_preview") or ""),
            truncated=bool(payload.get("truncated", False)),
        )

    if capability_id == "filesystem.write_file" and isinstance(payload, dict):
        path = str(payload.get("path") or "").strip()
        absolute_path = str(payload.get("absolute_path") or "").strip()
        display_path = absolute_path or path
        file_format = str(payload.get("format") or "").strip()
        bytes_written = payload.get("bytes_written")
        message = str(payload.get("message") or "").strip()
        if not message:
            message = f"Saved file to `{display_path}`"
            if bytes_written is not None:
                message += f" ({bytes_written} bytes"
                if file_format:
                    message += f", {file_format}"
                message += ")."
            else:
                message += "."
        return TextResult(
            node_id=result.node_id,
            capability_id=capability_id,
            operation_id=operation_id,
            title=title or "Saved File",
            text=message,
            metadata={
                "path": path,
                "absolute_path": absolute_path or None,
                "display_path": display_path or None,
                "format": file_format or None,
                "bytes_written": bytes_written,
                "created": bool(payload.get("created", False)),
                "overwritten": bool(payload.get("overwritten", False)),
            },
        )

    if capability_id == "data.aggregate" and isinstance(payload, dict):
        return AggregateResult(
            node_id=result.node_id,
            capability_id=capability_id,
            operation_id=operation_id,
            title=title,
            operation=str(payload.get("operation") or result.metadata.get("operation") or "aggregate"),
            field=payload.get("field"),
            value=payload.get("value"),
            unit=payload.get("unit"),
            row_count=int(payload.get("row_count", 0) or 0),
            used_count=int(payload.get("used_count", 0) or 0),
            skipped_count=int(payload.get("skipped_count", 0) or 0),
            label=payload.get("label"),
        )

    if capability_id in {"data.head", "data.project"}:
        return RecordListResult(
            node_id=result.node_id,
            capability_id=capability_id,
            operation_id=operation_id,
            title=title,
            records=_rows_from_payload(payload),
        )

    if capability_id == "runtime.describe_capabilities" and isinstance(payload, dict):
        return CapabilityListResult(
            node_id=result.node_id,
            capability_id=capability_id,
            operation_id=operation_id,
            title=title,
            grouped_capabilities=dict(payload.get("grouped_capabilities") or {}),
            capabilities=_rows_from_payload(payload),
            capability_count=int(payload.get("capability_count", 0) or 0),
        )

    if capability_id == "shell.list_processes" and isinstance(payload, dict):
        return ProcessListResult(
            node_id=result.node_id,
            capability_id=capability_id,
            operation_id=operation_id,
            title=title,
            processes=_rows_from_payload(payload),
            pattern=payload.get("pattern"),
            truncated=bool(payload.get("truncated", False)),
        )

    if capability_id == "shell.git_status" and isinstance(payload, dict):
        lines = payload.get("status_lines")
        text = "\n".join(str(item) for item in lines) if isinstance(lines, list) else str(payload)
        return TextResult(
            node_id=result.node_id,
            capability_id=capability_id,
            operation_id=operation_id,
            title=title,
            text=text,
        )

    if capability_id in {
        "system.memory_status",
        "system.disk_usage",
        "system.cpu_load",
        "system.environment_summary",
    } and isinstance(payload, dict):
        return TableResult(
            node_id=result.node_id,
            capability_id=capability_id,
            operation_id=operation_id,
            title=title,
            rows=_rows_from_payload(payload),
        )

    if capability_id == "system.uptime" and isinstance(payload, dict):
        return ScalarResult(
            node_id=result.node_id,
            capability_id=capability_id,
            operation_id=operation_id,
            title=title,
            value=payload.get("human") or payload.get("seconds"),
            label="Uptime",
            unit=None if payload.get("human") else "seconds",
        )

    if capability_id == "markdown.render" and isinstance(payload, dict):
        return TextResult(
            node_id=result.node_id,
            capability_id=capability_id,
            operation_id=operation_id,
            title=title,
            text=str(payload.get("markdown") or ""),
        )

    rows = _rows_from_payload(payload)
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return TableResult(
            node_id=result.node_id,
            capability_id=capability_id,
            operation_id=operation_id,
            title=title,
            rows=rows,
        )
    if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
        return RecordListResult(
            node_id=result.node_id,
            capability_id=capability_id,
            operation_id=operation_id,
            title=title,
            records=rows,
        )
    if isinstance(payload, dict) and isinstance(payload.get("matches"), list):
        return RecordListResult(
            node_id=result.node_id,
            capability_id=capability_id,
            operation_id=operation_id,
            title=title,
            records=rows,
        )
    if isinstance(payload, dict) and "value" in payload:
        if "operation" in payload:
            return AggregateResult(
                node_id=result.node_id,
                capability_id=capability_id,
                operation_id=operation_id,
                title=title,
                operation=str(payload.get("operation") or "aggregate"),
                field=payload.get("field"),
                value=payload.get("value"),
                unit=payload.get("unit"),
                row_count=int(payload.get("row_count", 0) or 0),
                used_count=int(payload.get("used_count", 0) or 0),
                skipped_count=int(payload.get("skipped_count", 0) or 0),
                label=payload.get("label"),
            )
        return ScalarResult(
            node_id=result.node_id,
            capability_id=capability_id,
            operation_id=operation_id,
            title=title,
            value=payload.get("value"),
            label=payload.get("label"),
            unit=payload.get("unit"),
        )
    if isinstance(payload, dict) and "content_preview" in payload:
        return FileContentResult(
            node_id=result.node_id,
            capability_id=capability_id,
            operation_id=operation_id,
            title=title,
            path=payload.get("path"),
            content_preview=str(payload.get("content_preview") or ""),
            truncated=bool(payload.get("truncated", False)),
        )
    if isinstance(payload, dict) and "message" in payload:
        return TextResult(
            node_id=result.node_id,
            capability_id=capability_id,
            operation_id=operation_id,
            title=title,
            text=str(payload.get("message") or ""),
        )

    return TextResult(
        node_id=result.node_id,
        capability_id=capability_id,
        operation_id=operation_id,
        title=title,
        text=str(payload if payload is not None else result.error or ""),
    )


__all__ = [
    "AggregateResult",
    "CapabilityListResult",
    "DirectoryListingResult",
    "ErrorResult",
    "FileContentResult",
    "MultiSectionResult",
    "ProcessListResult",
    "RecordListResult",
    "ResultShape",
    "ScalarResult",
    "TableResult",
    "TextResult",
    "normalize_execution_result",
]


MultiSectionResult.model_rebuild()
