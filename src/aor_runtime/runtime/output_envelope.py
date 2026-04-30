"""OpenFABRIC Runtime Module: aor_runtime.runtime.output_envelope

Purpose:
    Normalize heterogeneous tool results into common output envelopes.

Responsibilities:
    Convert SQL rows, SLURM groups/jobs, filesystem matches, shell tables, files, scalars, and text into comparable shapes.

Data flow / Interfaces:
    Feeds auto-artifacts, presentation, and output-shape validation with presentation/source counts and truncation metadata.

Boundaries:
    Keeps raw tool payload structures from leaking directly into final rendering decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from aor_runtime.runtime.output_shape import TOOL_RESULT_SHAPES


EnvelopeKind = Literal["scalar", "table", "file", "text", "json", "status", "unknown"]


@dataclass(frozen=True)
class OutputEnvelope:
    """Represent output envelope within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by OutputEnvelope.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.output_envelope.OutputEnvelope and related tests.
    """
    kind: EnvelopeKind = "unknown"
    rows: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""
    scalar: Any = None
    file: str = ""
    presentation_count: int = 0
    source_count: int | None = None
    truncated: bool = False
    source_tool: str = ""
    source_field: str = ""


def normalize_tool_output(tool: str, result: dict[str, Any] | Any) -> OutputEnvelope:
    """Normalize tool output for the surrounding runtime workflow.

    Inputs:
        Receives tool, result for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_envelope.normalize_tool_output.
    """
    if not isinstance(result, dict):
        text = "" if result is None else str(result)
        return OutputEnvelope(kind="text" if text else "unknown", text=text, presentation_count=1 if text else 0, source_tool=tool)

    if tool == "shell.exec":
        stdout = str(result.get("stdout") or "")
        rows = parse_shell_table(stdout)
        if rows:
            return OutputEnvelope(
                kind="table",
                rows=rows,
                text=stdout,
                presentation_count=len(rows),
                source_count=None,
                truncated=bool(result.get("truncated")),
                source_tool=tool,
                source_field="stdout",
            )
        return OutputEnvelope(
            kind="text" if stdout else "status",
            text=stdout,
            scalar=result.get("returncode"),
            presentation_count=1 if stdout else 0,
            truncated=bool(result.get("truncated")),
            source_tool=tool,
            source_field="stdout",
        )

    shape = TOOL_RESULT_SHAPES.get(tool)
    if shape is not None:
        for field_name in shape.collection_fields:
            value = result.get(field_name)
            rows = _coerce_collection_rows(tool, field_name, value)
            if rows:
                presentation_count, source_count = _collection_counts(tool, result, field_name, rows)
                return OutputEnvelope(
                    kind="table",
                    rows=rows,
                    presentation_count=presentation_count,
                    source_count=source_count,
                    truncated=bool(result.get("truncated")) or presentation_count > len(rows),
                    source_tool=tool,
                    source_field=field_name,
                )
        for field_name in shape.file_fields:
            value = str(result.get(field_name) or "")
            if value:
                return OutputEnvelope(kind="file", file=value, presentation_count=1, source_tool=tool, source_field=field_name)
        for field_name in shape.scalar_fields:
            if field_name in result:
                return OutputEnvelope(
                    kind="scalar",
                    scalar=result.get(field_name),
                    presentation_count=1,
                    source_tool=tool,
                    source_field=field_name,
                )
        for field_name in shape.text_fields:
            value = str(result.get(field_name) or "")
            if value:
                return OutputEnvelope(kind="text", text=value, presentation_count=1, source_tool=tool, source_field=field_name)

    rows = _coerce_rows(result)
    if rows:
        return OutputEnvelope(kind="table", rows=rows, presentation_count=len(rows), source_tool=tool, source_field="result")
    return OutputEnvelope(kind="json", text=str(result), presentation_count=1 if result else 0, source_tool=tool, source_field="result")


def parse_shell_table(text: str) -> list[dict[str, str]]:
    """Parse shell table for the surrounding runtime workflow.

    Inputs:
        Receives text for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_envelope.parse_shell_table.
    """
    lines = [line.rstrip() for line in str(text or "").splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    headers = _headers_for_shell_table(lines[0])
    if len(headers) < 2:
        return []
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        values = re.split(r"\s+", line.strip(), maxsplit=len(headers) - 1)
        if len(values) < min(2, len(headers)):
            continue
        while len(values) < len(headers):
            values.append("")
        rows.append({headers[index]: values[index] for index in range(len(headers))})
    if len(rows) < 1:
        return []
    return rows


def _headers_for_shell_table(header_line: str) -> list[str]:
    """Handle the internal headers for shell table helper path for this module.

    Inputs:
        Receives header_line for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_envelope._headers_for_shell_table.
    """
    normalized = re.sub(r"\s+", " ", str(header_line or "").strip())
    lowered = normalized.lower()
    if lowered.startswith("user pid %cpu %mem"):
        return ["USER", "PID", "%CPU", "%MEM", "VSZ", "RSS", "TTY", "STAT", "START", "TIME", "COMMAND"]
    if lowered.startswith("filesystem ") and " mounted on" in lowered:
        return ["Filesystem", "Size", "Used", "Avail", "Use%", "Mounted on"]
    if lowered.startswith("netid state recv-q send-q"):
        return ["Netid", "State", "Recv-Q", "Send-Q", "Local Address:Port", "Peer Address:Port", "Process"]
    if lowered.startswith("command pid user"):
        return re.split(r"\s+", normalized)
    headers = re.split(r"\s+", normalized)
    if len(headers) < 2:
        return []
    if not _looks_like_header(headers):
        return []
    return headers


def _looks_like_header(headers: list[str]) -> bool:
    """Handle the internal looks like header helper path for this module.

    Inputs:
        Receives headers for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_envelope._looks_like_header.
    """
    known = {
        "pid",
        "user",
        "command",
        "state",
        "name",
        "size",
        "used",
        "avail",
        "filesystem",
        "tty",
        "stat",
        "time",
    }
    normalized = {header.strip().lower().strip("%") for header in headers}
    return bool(normalized & known)


def _coerce_rows(value: Any) -> list[dict[str, Any]]:
    """Handle the internal coerce rows helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_envelope._coerce_rows.
    """
    if isinstance(value, list):
        if all(isinstance(item, dict) for item in value):
            return [dict(item) for item in value]
        return [{"value": item} for item in value]
    if isinstance(value, dict):
        return [dict(value)]
    return []


def _coerce_collection_rows(tool: str, field_name: str, value: Any) -> list[dict[str, Any]]:
    """Handle the internal coerce collection rows helper path for this module.

    Inputs:
        Receives tool, field_name, value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_envelope._coerce_collection_rows.
    """
    if tool in {"slurm.queue", "slurm.accounting"} and field_name == "grouped" and isinstance(value, dict):
        return [{"group": key, "count": count} for key, count in sorted(value.items(), key=lambda item: str(item[0]))]
    return _coerce_rows(value)


def _collection_counts(
    tool: str,
    result: dict[str, Any],
    field_name: str,
    rows: list[dict[str, Any]],
) -> tuple[int, int | None]:
    """Handle the internal collection counts helper path for this module.

    Inputs:
        Receives tool, result, field_name, rows for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_envelope._collection_counts.
    """
    fallback = len(rows)
    if tool == "slurm.accounting_aggregate" and field_name == "groups":
        return fallback, _int_field(result, "job_count", "total_count", "count")
    if tool in {"slurm.queue", "slurm.accounting"} and field_name == "grouped":
        return fallback, _int_field(result, "total_count", "count", "returned_count")
    if tool == "slurm.nodes" and field_name == "nodes":
        return fallback, _int_field(result, "partition_row_count", "count", "total_count")
    if tool in {"slurm.queue", "slurm.accounting"} and field_name == "jobs":
        count = _int_field(result, "total_count", "count", "returned_count")
        return max(count or fallback, fallback), None
    if tool == "sql.query" and field_name == "rows":
        count = _int_field(result, "row_count")
        return max(count or fallback, fallback), None
    return fallback, None


def _int_field(result: dict[str, Any], *keys: str) -> int | None:
    """Handle the internal int field helper path for this module.

    Inputs:
        Receives result for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_envelope._int_field.
    """
    for key in keys:
        value = result.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None
