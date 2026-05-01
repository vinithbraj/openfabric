"""Deterministic internal data primitives for structured execution results."""

from __future__ import annotations

import re
from statistics import mean
from typing import Any

from agent_runtime.capabilities.base import BaseCapability
from agent_runtime.capabilities.schemas import CapabilityManifest
from agent_runtime.core.errors import ValidationError
from agent_runtime.core.types import DataRef, ExecutionResult, InputRef

_ALLOWED_AGGREGATE_OPERATIONS = {"sum", "count", "min", "max", "avg"}
_SAFE_FIELD_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]+$")


def _resolve_input_data(input_value: Any, result_store) -> Any:
    """Resolve input_ref-like values into concrete structured data."""

    if isinstance(input_value, InputRef):
        return result_store.resolve_input_ref(input_value)
    if isinstance(input_value, DataRef):
        return result_store.get(input_value.ref_id)
    if isinstance(input_value, str):
        return result_store.get(input_value)
    return input_value


def _safe_field_name(value: Any, field_name: str = "field") -> str:
    """Validate a field-like identifier without allowing expressions."""

    text = str(value or "").strip()
    if not text:
        raise ValidationError(f"{field_name} must be a non-empty string.")
    if not _SAFE_FIELD_PATTERN.match(text):
        raise ValidationError(f"{field_name} contains unsupported expression text.")
    return text


def _extract_records(input_data: Any) -> list[dict[str, Any]]:
    """Extract structured records from supported payload shapes."""

    if isinstance(input_data, list):
        if all(isinstance(item, dict) for item in input_data):
            return [dict(item) for item in input_data]
        raise ValidationError("input_ref must resolve to structured row objects.")
    if not isinstance(input_data, dict):
        raise ValidationError("input_ref must resolve to a list or object containing structured records.")

    if isinstance(input_data.get("entries"), list):
        rows = input_data["entries"]
        if all(isinstance(item, dict) for item in rows):
            return [dict(item) for item in rows]
    if isinstance(input_data.get("rows"), list):
        rows = input_data["rows"]
        if all(isinstance(item, dict) for item in rows):
            return [dict(item) for item in rows]
    if isinstance(input_data.get("processes"), list):
        rows = input_data["processes"]
        if all(isinstance(item, dict) for item in rows):
            return [dict(item) for item in rows]
    if isinstance(input_data.get("listeners"), list):
        rows = input_data["listeners"]
        if all(isinstance(item, dict) for item in rows):
            return [dict(item) for item in rows]
    if isinstance(input_data.get("matches"), list):
        matches = input_data["matches"]
        return [{"path": item} for item in matches]

    raise ValidationError("input_ref does not contain supported structured records.")


def _validate_filter(filter_value: Any) -> dict[str, Any]:
    """Validate a constrained equality-only filter map."""

    if filter_value is None:
        return {}
    if not isinstance(filter_value, dict):
        raise ValidationError("filter must be an object with simple equality predicates.")
    validated: dict[str, Any] = {}
    for key, value in filter_value.items():
        field_name = _safe_field_name(key, "filter key")
        if isinstance(value, (dict, list, tuple, set)):
            raise ValidationError("filter supports simple equality values only.")
        validated[field_name] = value
    return validated


def _apply_filter(rows: list[dict[str, Any]], filter_value: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply equality-only filtering to structured rows."""

    if not filter_value:
        return list(rows)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if all(row.get(key) == expected for key, expected in filter_value.items()):
            filtered.append(row)
    return filtered


def _numeric_values(rows: list[dict[str, Any]], field: str) -> tuple[list[float], int]:
    """Extract numeric field values while tracking skipped rows."""

    values: list[float] = []
    skipped_count = 0
    for row in rows:
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            skipped_count += 1
            continue
        values.append(float(value))
    return values, skipped_count


class DataAggregateCapability(BaseCapability):
    """Compute deterministic aggregate summaries over structured records."""

    manifest = CapabilityManifest(
        capability_id="data.aggregate",
        domain="data",
        operation_id="aggregate",
        name="Aggregate Structured Data",
        description="Compute deterministic aggregates over structured list or dictionary data returned by another capability.",
        semantic_verbs=["analyze", "transform", "summarize"],
        object_types=["data.records", "table", "list", "structured_output"],
        argument_schema={
            "input_ref": {"type": "string"},
            "operation": {"type": "string"},
            "field": {"type": "string"},
            "filter": {"type": "object"},
            "label": {"type": "string"},
            "unit": {"type": "string"},
        },
        required_arguments=["input_ref", "operation"],
        optional_arguments=["field", "filter", "label", "unit"],
        output_schema={
            "operation": {"type": "string"},
            "field": {"type": ["string", "null"]},
            "value": {"type": ["number", "null"]},
            "unit": {"type": ["string", "null"]},
            "row_count": {"type": "integer"},
            "used_count": {"type": "integer"},
            "skipped_count": {"type": "integer"},
            "label": {"type": ["string", "null"]},
        },
        execution_backend="internal",
        backend_operation="data.aggregate",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[
            {
                "arguments": {
                    "input_ref": "task-list.output",
                    "operation": "sum",
                    "field": "size",
                    "filter": {"type": "file"},
                }
            }
        ],
        safety_notes=[
            "No arbitrary Python execution.",
            "Supports deterministic equality-only filtering.",
        ],
    )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        validated = super().validate_arguments(arguments)
        operation = str(validated.get("operation") or "").strip().lower()
        if operation not in _ALLOWED_AGGREGATE_OPERATIONS:
            raise ValidationError(f"unsupported aggregate operation: {operation}")
        validated["operation"] = operation
        if operation != "count":
            if "field" not in validated:
                raise ValidationError(f"{operation} requires field.")
            validated["field"] = _safe_field_name(validated["field"])
        elif "field" in validated:
            validated["field"] = _safe_field_name(validated["field"])
        if "filter" in validated:
            validated["filter"] = _validate_filter(validated["filter"])
        if "label" in validated:
            validated["label"] = str(validated["label"]).strip()
        if "unit" in validated:
            validated["unit"] = str(validated["unit"]).strip()
        return validated

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ExecutionResult:
        validated = self.validate_arguments(dict(arguments or {}))
        result_store = context.get("result_store")
        if result_store is None:
            raise ValidationError("result_store is required for data.aggregate.")

        input_data = _resolve_input_data(validated["input_ref"], result_store)
        rows = _extract_records(input_data)
        filtered_rows = _apply_filter(rows, dict(validated.get("filter") or {}))
        operation = validated["operation"]
        row_count = len(rows)
        label = validated.get("label")
        unit = validated.get("unit")
        field = validated.get("field")

        if operation == "count":
            value: float | int | None = len(filtered_rows)
            used_count = len(filtered_rows)
            skipped_count = 0
        else:
            values, skipped_count = _numeric_values(filtered_rows, str(field))
            used_count = len(values)
            if operation == "sum":
                value = sum(values)
            elif operation == "min":
                value = min(values) if values else None
            elif operation == "max":
                value = max(values) if values else None
            else:
                value = mean(values) if values else None

        preview = {
            "operation": operation,
            "field": field,
            "value": value,
            "unit": unit,
            "row_count": row_count,
            "used_count": used_count,
            "skipped_count": skipped_count,
            "label": label,
        }
        return ExecutionResult(
            node_id=str(context.get("node_id") or ""),
            status="success",
            data_preview=preview,
            metadata={"operation": operation, "data_type": "summary"},
        )


class DataProjectCapability(BaseCapability):
    """Project a subset of fields from structured records."""

    manifest = CapabilityManifest(
        capability_id="data.project",
        domain="data",
        operation_id="project",
        name="Project Structured Data",
        description="Select a subset of fields from structured records returned by another capability.",
        semantic_verbs=["transform", "summarize"],
        object_types=["data.records", "table", "list", "structured_output"],
        argument_schema={
            "input_ref": {"type": "string"},
            "fields": {"type": "array"},
        },
        required_arguments=["input_ref", "fields"],
        optional_arguments=[],
        output_schema={"rows": {"type": "array"}, "row_count": {"type": "integer"}},
        execution_backend="internal",
        backend_operation="data.project",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"input_ref": "task-list.output", "fields": ["path", "size"]}}],
        safety_notes=["Deterministic structured projection only."],
    )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        validated = super().validate_arguments(arguments)
        fields = validated.get("fields")
        if not isinstance(fields, list) or not fields:
            raise ValidationError("fields must be a non-empty list.")
        validated["fields"] = [_safe_field_name(field, "field") for field in fields]
        return validated

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ExecutionResult:
        validated = self.validate_arguments(dict(arguments or {}))
        result_store = context.get("result_store")
        if result_store is None:
            raise ValidationError("result_store is required for data.project.")
        input_data = _resolve_input_data(validated["input_ref"], result_store)
        rows = _extract_records(input_data)
        fields = validated["fields"]
        projected = [{field: row.get(field) for field in fields} for row in rows]
        return ExecutionResult(
            node_id=str(context.get("node_id") or ""),
            status="success",
            data_preview={"rows": projected, "row_count": len(projected)},
            metadata={"data_type": "table"},
        )


class DataHeadCapability(BaseCapability):
    """Return the first N structured records."""

    manifest = CapabilityManifest(
        capability_id="data.head",
        domain="data",
        operation_id="head",
        name="Head Structured Data",
        description="Return the first N structured records from another capability output.",
        semantic_verbs=["read", "summarize", "transform"],
        object_types=["data.records", "table", "list", "structured_output"],
        argument_schema={
            "input_ref": {"type": "string"},
            "limit": {"type": "integer"},
        },
        required_arguments=["input_ref"],
        optional_arguments=["limit"],
        output_schema={"rows": {"type": "array"}, "row_count": {"type": "integer"}},
        execution_backend="internal",
        backend_operation="data.head",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"input_ref": "task-search.output", "limit": 5}}],
        safety_notes=["Returns a deterministic prefix of structured records only."],
    )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        validated = super().validate_arguments(arguments)
        validated["limit"] = max(1, int(validated.get("limit", 5)))
        return validated

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ExecutionResult:
        validated = self.validate_arguments(dict(arguments or {}))
        result_store = context.get("result_store")
        if result_store is None:
            raise ValidationError("result_store is required for data.head.")
        input_data = _resolve_input_data(validated["input_ref"], result_store)
        rows = _extract_records(input_data)
        limited = rows[: int(validated["limit"])]
        return ExecutionResult(
            node_id=str(context.get("node_id") or ""),
            status="success",
            data_preview={"rows": limited, "row_count": len(limited), "limit": int(validated["limit"])},
            metadata={"data_type": "table"},
        )


__all__ = [
    "DataAggregateCapability",
    "DataHeadCapability",
    "DataProjectCapability",
]
