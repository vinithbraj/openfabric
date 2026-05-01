"""Deterministic in-memory table transformations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from agent_runtime.capabilities.base import BaseCapability
from agent_runtime.capabilities.schemas import CapabilityManifest
from agent_runtime.core.errors import ValidationError
from agent_runtime.core.types import DataRef, ExecutionResult

ALLOWED_OPERATIONS = {"head", "select_columns", "sort", "filter", "aggregate"}


class TransformTableCapability(BaseCapability):
    """Apply deterministic table transformations to stored tabular data."""

    manifest = CapabilityManifest(
        capability_id="python_data.transform_table",
        domain="python_data",
        operation_id="transform_table",
        name="Transform Table",
        description="Apply a deterministic transformation to tabular data already loaded in memory.",
        semantic_verbs=["transform", "analyze"],
        object_types=["table", "dataset"],
        argument_schema={
            "input_ref": {"type": "string"},
            "operation": {"type": "string"},
            "parameters": {"type": "object"},
        },
        required_arguments=["input_ref", "operation", "parameters"],
        optional_arguments=[],
        output_schema={"rows": {"type": "array"}, "summary": {"type": "object"}},
        execution_backend="local",
        backend_operation="python_data.transform_table",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[
            {
                "arguments": {
                    "input_ref": "data-demo",
                    "operation": "head",
                    "parameters": {"count": 5},
                }
            }
        ],
        safety_notes=["Does not execute arbitrary Python code."],
    )

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ExecutionResult:
        """Transform referenced tabular data using a fixed operation vocabulary."""

        validated = self.validate_arguments(dict(arguments or {}))
        result_store = context.get("result_store")
        if result_store is None:
            raise ValidationError("result_store is required for table transformations.")

        input_value = validated["input_ref"]
        if isinstance(input_value, DataRef):
            input_data = result_store.get(input_value.ref_id)
        elif isinstance(input_value, str):
            input_data = result_store.get(input_value)
        else:
            input_data = input_value
        rows = _extract_rows(input_data)
        operation = str(validated["operation"])
        parameters = dict(validated["parameters"] or {})
        if operation not in ALLOWED_OPERATIONS:
            raise ValidationError(f"unsupported table operation: {operation}")

        if operation == "head":
            count = max(1, int(parameters.get("count", 5)))
            output = rows[:count]
            preview = {"rows": output, "row_count": len(output)}
        elif operation == "select_columns":
            columns = parameters.get("columns")
            if not isinstance(columns, list) or not columns:
                raise ValidationError("select_columns requires a non-empty parameters.columns list.")
            output = [{column: row.get(column) for column in columns} for row in rows]
            preview = {"rows": output, "row_count": len(output)}
        elif operation == "sort":
            column = str(parameters.get("column") or "").strip()
            if not column:
                raise ValidationError("sort requires parameters.column.")
            descending = bool(parameters.get("descending", False))
            output = sorted(rows, key=lambda row: row.get(column), reverse=descending)
            preview = {"rows": output, "row_count": len(output)}
        elif operation == "filter":
            column = str(parameters.get("column") or "").strip()
            if not column:
                raise ValidationError("filter requires parameters.column.")
            output = _filter_rows(rows, column, parameters)
            preview = {"rows": output, "row_count": len(output)}
        else:
            preview = {"summary": _aggregate_rows(rows, parameters)}

        return ExecutionResult(
            node_id=str(context.get("node_id") or ""),
            status="success",
            data_preview=preview,
            metadata={"operation": operation},
        )


def _extract_rows(input_data: Any) -> list[dict[str, Any]]:
    """Extract row-shaped dictionaries from stored input data."""

    if isinstance(input_data, list):
        if all(isinstance(item, dict) for item in input_data):
            return [dict(item) for item in input_data]
        raise ValidationError("input_ref must point to a list of row objects.")
    if isinstance(input_data, dict):
        if isinstance(input_data.get("rows"), list):
            rows = input_data["rows"]
        elif isinstance(input_data.get("entries"), list):
            rows = input_data["entries"]
        else:
            raise ValidationError("input_ref does not contain rows or entries.")
        if not all(isinstance(item, dict) for item in rows):
            raise ValidationError("row collections must contain objects.")
        return [dict(item) for item in rows]
    raise ValidationError("input_ref must point to row-oriented data.")


def _filter_rows(rows: list[dict[str, Any]], column: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
    """Filter rows using a constrained predicate vocabulary."""

    if "equals" in parameters:
        return [row for row in rows if row.get(column) == parameters["equals"]]
    if "contains" in parameters:
        needle = str(parameters["contains"])
        return [row for row in rows if needle in str(row.get(column, ""))]
    if "gt" in parameters:
        threshold = parameters["gt"]
        return [row for row in rows if row.get(column) is not None and row.get(column) > threshold]
    if "lt" in parameters:
        threshold = parameters["lt"]
        return [row for row in rows if row.get(column) is not None and row.get(column) < threshold]
    raise ValidationError("filter requires one of equals, contains, gt, or lt.")


def _aggregate_rows(rows: list[dict[str, Any]], parameters: dict[str, Any]) -> dict[str, Any]:
    """Aggregate rows using a small deterministic metric vocabulary."""

    metric = str(parameters.get("metric") or "count")
    group_by = str(parameters.get("group_by") or "").strip() or None
    column = str(parameters.get("column") or "").strip() or None

    def aggregate_subset(subset: list[dict[str, Any]]) -> Any:
        if metric == "count":
            return len(subset)
        if not column:
            raise ValidationError(f"{metric} aggregate requires parameters.column.")
        numeric_values = [row.get(column) for row in subset if isinstance(row.get(column), (int, float))]
        if metric == "sum":
            return sum(numeric_values)
        if metric == "mean":
            return sum(numeric_values) / len(numeric_values) if numeric_values else None
        raise ValidationError(f"unsupported aggregate metric: {metric}")

    if group_by is None:
        return {"metric": metric, "value": aggregate_subset(rows)}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_by))].append(row)
    return {
        "metric": metric,
        "group_by": group_by,
        "groups": {key: aggregate_subset(value) for key, value in grouped.items()},
    }
