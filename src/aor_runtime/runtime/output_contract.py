from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel


class OutputContract(BaseModel):
    mode: Literal["text", "csv", "json", "count"]
    path_style: Literal["name", "relative", "absolute"] | None = None
    json_shape: Literal["matches", "rows", "count", "value"] | None = None
    include_extra_text: bool = False


def normalize_output(value: Any, contract: OutputContract) -> Any:
    if contract.mode == "count":
        count_value = _coerce_count(value)
        if contract.json_shape == "count":
            return {"count": count_value}
        return count_value

    if contract.mode == "csv":
        if contract.json_shape == "rows":
            return _coerce_rows(value)
        values = _coerce_sequence(value, path_style=contract.path_style)
        return values

    if contract.mode == "text":
        if contract.json_shape == "rows":
            return _coerce_rows(value)
        if isinstance(value, str):
            if contract.path_style is None:
                return value
            lines = [line for line in value.splitlines() if line.strip()]
            if len(lines) <= 1:
                return _apply_path_style(value.strip(), contract.path_style) if value.strip() else value
            return [_apply_path_style(line, contract.path_style) for line in lines]
        return _coerce_sequence(value, path_style=contract.path_style)

    normalized_json = _normalize_json_value(value, contract)
    if contract.json_shape == "matches":
        return {"matches": _coerce_sequence(normalized_json, path_style=contract.path_style)}
    if contract.json_shape == "rows":
        return {"rows": _coerce_rows(normalized_json)}
    if contract.json_shape == "count":
        return {"count": _coerce_count(normalized_json)}
    if contract.json_shape == "value":
        return {"value": normalized_json}
    return normalized_json


def render_output(value: Any, contract: OutputContract) -> str:
    if contract.mode == "count":
        if isinstance(value, dict) and "count" in value:
            return str(value["count"])
        return str(value)
    if contract.mode == "csv":
        if contract.json_shape == "rows":
            return _rows_to_csv(_coerce_rows(value))
        if isinstance(value, str):
            return value.strip()
        return ",".join(str(item) for item in _coerce_sequence(value, path_style=contract.path_style))
    if contract.mode == "text":
        if contract.json_shape == "rows":
            return _rows_to_text(_coerce_rows(value))
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return "\n".join(str(item) for item in value)
        return str(value or "").strip()
    return _json_dumps_safe(value)


def build_output_contract(
    *,
    mode: str,
    path_style: str | None = None,
    json_shape: str | None = None,
    include_extra_text: bool = False,
) -> dict[str, Any]:
    contract = OutputContract(
        mode=mode,
        path_style=path_style,
        json_shape=json_shape,
        include_extra_text=include_extra_text,
    )
    return contract.model_dump(exclude_none=True)


def _normalize_json_value(value: Any, contract: OutputContract) -> Any:
    if contract.json_shape == "rows":
        return _coerce_rows(value)
    if contract.json_shape == "matches":
        return _coerce_sequence(value, path_style=contract.path_style)
    if isinstance(value, str):
        stripped = value.strip()
        if _looks_like_json_value(stripped):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
    return value


def _coerce_rows(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and "rows" in value and isinstance(value["rows"], list):
        return value["rows"]
    if isinstance(value, dict):
        for key in ("jobs", "nodes", "partitions"):
            if key in value and isinstance(value[key], list):
                return value[key]
    return [value]


def _coerce_count(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            return int(stripped)
        lines = [line for line in stripped.splitlines() if line.strip()]
        return len(lines)
    if isinstance(value, dict):
        if "count" in value:
            return _coerce_count(value["count"])
        if "matches" in value:
            return len(_coerce_sequence(value["matches"]))
        if "rows" in value and isinstance(value["rows"], list):
            return len(value["rows"])
        for key in ("jobs", "nodes", "partitions"):
            if key in value and isinstance(value[key], list):
                return len(value[key])
        return len(value)
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return 0


def _coerce_sequence(value: Any, *, path_style: str | None = None) -> list[str]:
    if isinstance(value, str):
        items = [line for line in value.splitlines() if line.strip()]
        if not items and value.strip():
            items = [value.strip()]
    elif isinstance(value, list):
        if _is_single_column_rows(value):
            key = next(iter(value[0]))
            items = [str(row.get(key, "")) for row in value]
        else:
            items = [str(item) for item in value]
    elif isinstance(value, (tuple, set)):
        items = [str(item) for item in value]
    elif isinstance(value, dict) and "matches" in value:
        return _coerce_sequence(value["matches"], path_style=path_style)
    elif isinstance(value, dict) and "rows" in value:
        return _coerce_sequence(value["rows"], path_style=path_style)
    elif isinstance(value, dict):
        for key in ("jobs", "nodes", "partitions"):
            if key in value and isinstance(value[key], list):
                return _coerce_sequence(value[key], path_style=path_style)
        items = [str(value)]
    elif value is None:
        items = []
    else:
        items = [str(value)]
    if path_style:
        return [_apply_path_style(item, path_style) for item in items]
    return items


def _apply_path_style(value: str, path_style: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        return normalized
    if path_style == "name":
        return Path(normalized).name
    if path_style == "relative":
        return normalized[2:] if normalized.startswith("./") else normalized
    return normalized


def _is_single_column_rows(value: list[Any]) -> bool:
    if not value or not all(isinstance(item, dict) and len(item) == 1 for item in value):
        return False
    first_key = next(iter(value[0]))
    return all(next(iter(item)) == first_key for item in value)


def _rows_to_csv(rows: list[Any]) -> str:
    if not rows:
        return ""
    normalized_rows = [row for row in rows if isinstance(row, dict)]
    if not normalized_rows:
        return ",".join(str(item) for item in rows)
    headers = list(normalized_rows[0].keys())
    if len(headers) == 1:
        header = headers[0]
        return ",".join(str(row.get(header, "")) for row in normalized_rows)
    lines = [",".join(headers)]
    for row in normalized_rows:
        lines.append(",".join(str(row.get(header, "")) for header in headers))
    return "\n".join(lines)


def _rows_to_text(rows: list[Any]) -> str:
    if not rows:
        return ""
    normalized_rows = [row for row in rows if isinstance(row, dict)]
    if not normalized_rows:
        return "\n".join(str(item) for item in rows)
    headers = list(normalized_rows[0].keys())
    if len(headers) == 1:
        header = headers[0]
        return "\n".join(str(row.get(header, "")) for row in normalized_rows)
    widths = {
        header: max(len(str(header)), *(len(str(row.get(header, ""))) for row in normalized_rows))
        for header in headers
    }
    header_line = " | ".join(str(header).ljust(widths[header]) for header in headers)
    separator = "-+-".join("-" * widths[header] for header in headers)
    lines = [header_line, separator]
    for row in normalized_rows:
        lines.append(" | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))
    return "\n".join(lines)


def _looks_like_json_value(value: str) -> bool:
    return (value.startswith("{") and value.endswith("}")) or (value.startswith("[") and value.endswith("]"))


def _json_dumps_safe(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)
