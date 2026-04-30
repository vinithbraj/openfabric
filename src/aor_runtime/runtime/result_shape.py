"""OpenFABRIC Runtime Module: aor_runtime.runtime.result_shape

Purpose:
    Validate that final results satisfy the user-requested output shape.

Responsibilities:
    Check structured primary results for scalar, grouped-count, table/list, JSON, shell-table, and collection-leak mismatches.

Data flow / Interfaces:
    Consumes user goal, StepLog history, and final rendered content after presentation.

Boundaries:
    Separates primary-result correctness from verbose display content such as stats, trace, and DAG sections.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from aor_runtime.core.contracts import StepLog
from aor_runtime.runtime.output_envelope import parse_shell_table
from aor_runtime.runtime.output_shape import (
    TOOL_RESULT_SHAPES,
    grouped_count_field_for_goal,
    is_collection_ref,
    is_grouped_count_goal,
    is_scalar_count_goal,
    scalar_field_for_tool,
    validate_final_output_contract,
)
from aor_runtime.runtime.tool_output_contracts import normalize_tool_ref_path


LIST_GOAL_RE = re.compile(r"\b(?:list|show|display|return)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ResultShapeValidation:
    """Represent result shape validation within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ResultShapeValidation.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.result_shape.ResultShapeValidation and related tests.
    """
    success: bool
    reason: str | None = None
    metadata: dict[str, Any] | None = None


def validate_result_shape(
    goal: str,
    history: list[StepLog],
    final_content: str | None = None,
    *,
    allow_raw_json: bool = False,
) -> ResultShapeValidation:
    """Validate result shape for the surrounding runtime workflow.

    Inputs:
        Receives goal, history, final_content, allow_raw_json for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape.validate_result_shape.
    """
    goal_text = str(goal or "")
    if is_grouped_count_goal(goal_text):
        grouped_validation = _validate_grouped_count_result(goal_text, history)
        if not grouped_validation.success:
            return grouped_validation
        final_validation = _validate_final_output(
            goal_text,
            history,
            final_content=final_content,
            allow_raw_json=allow_raw_json,
            enforce_scalar_text=False,
        )
        if not final_validation.success:
            return final_validation
        return ResultShapeValidation(True)

    if not _is_count_goal(goal_text):
        final_validation = _validate_final_output(goal_text, history, final_content=final_content, allow_raw_json=allow_raw_json)
        if not final_validation.success:
            return final_validation
        return ResultShapeValidation(True)

    collection_validation = _validate_count_final_dataflow(history)
    if not collection_validation.success:
        return collection_validation

    primary_validation = _validate_count_primary_result(goal_text, history)
    if primary_validation is None:
        final_validation = _validate_final_output(goal_text, history, final_content=final_content, allow_raw_json=allow_raw_json)
        if not final_validation.success:
            return final_validation
        contract_validation = validate_final_output_contract(goal_text, history, final_content=final_content)
        if not contract_validation.success:
            return ResultShapeValidation(contract_validation.success, contract_validation.reason, contract_validation.metadata)
        return ResultShapeValidation(True)
    if not primary_validation.success:
        return primary_validation

    final_validation = _validate_final_output(
        goal_text,
        history,
        final_content=final_content,
        allow_raw_json=allow_raw_json,
        enforce_scalar_text=False,
    )
    if not final_validation.success:
        return final_validation
    return ResultShapeValidation(True)


def _validate_count_primary_result(goal: str, history: list[StepLog]) -> ResultShapeValidation | None:
    """Handle the internal validate count primary result helper path for this module.

    Inputs:
        Receives goal, history for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._validate_count_primary_result.
    """
    sql_log = _last_successful_sql_query(history)
    if sql_log is not None:
        return _validate_sql_count_result(sql_log)

    final_log = _last_successful_runtime_return(history)
    if final_log is not None:
        value = final_log.result.get("value")
        if _is_numeric_scalar(value):
            return ResultShapeValidation(
                True,
                metadata={"primary_result_validation": "runtime_return_scalar", "expected_shape": "single numeric scalar"},
            )

    producer_log, field, value = _last_successful_scalar_producer(goal, history)
    if producer_log is None:
        return None
    if _is_numeric_scalar(value):
        return ResultShapeValidation(
            True,
            metadata={
                "primary_result_validation": "tool_scalar_field",
                "tool": producer_log.step.action,
                "field": field,
                "expected_shape": "single numeric scalar",
            },
        )
    return ResultShapeValidation(
        False,
        f"Count request expected numeric scalar from {producer_log.step.action}.{field}.",
        {
            "primary_result_validation": "tool_scalar_not_numeric",
            "tool": producer_log.step.action,
            "field": field,
            "value_type": type(value).__name__,
            "expected_shape": "single numeric scalar",
        },
    )


def _validate_sql_count_result(sql_log: StepLog) -> ResultShapeValidation:
    """Handle the internal validate sql count result helper path for this module.

    Inputs:
        Receives sql_log for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._validate_sql_count_result.
    """
    rows = sql_log.result.get("rows")
    if not isinstance(rows, list):
        return ResultShapeValidation(False, "Count request did not return SQL rows.", {"expected_shape": "single numeric scalar"})
    if len(rows) != 1:
        return ResultShapeValidation(
            False,
            (
                f"Count request returned {len(rows)} rows; expected one numeric aggregate. "
                "If the SQL uses GROUP BY/HAVING to filter entities, wrap it and SELECT COUNT(*) from the grouped subquery."
            ),
            {
                "expected_shape": "single numeric scalar",
                "actual_shape": {"row_count": len(rows), "columns": _row_columns(rows)},
                "failed_sql": str(sql_log.step.args.get("query") or ""),
            },
        )
    row = rows[0]
    if not isinstance(row, dict) or len(row) != 1:
        return ResultShapeValidation(
            False,
            "Count request must return exactly one aggregate column.",
            {
                "expected_shape": "single numeric scalar",
                "actual_shape": {"row_count": 1, "columns": _row_columns(rows)},
                "failed_sql": str(sql_log.step.args.get("query") or ""),
            },
        )
    value = next(iter(row.values()))
    if not _is_numeric_scalar(value):
        return ResultShapeValidation(
            False,
            "Count request returned a non-numeric value.",
            {
                "expected_shape": "single numeric scalar",
                "actual_shape": {"row_count": 1, "columns": _row_columns(rows), "value_type": type(value).__name__},
                "failed_sql": str(sql_log.step.args.get("query") or ""),
            },
        )
    return ResultShapeValidation(True, metadata={"primary_result_validation": "sql_single_numeric_scalar"})


def _validate_final_output(
    goal: str,
    history: list[StepLog],
    *,
    final_content: str | None = None,
    allow_raw_json: bool = False,
    enforce_scalar_text: bool = True,
) -> ResultShapeValidation:
    """Handle the internal validate final output helper path for this module.

    Inputs:
        Receives goal, history, final_content, allow_raw_json, enforce_scalar_text for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._validate_final_output.
    """
    final_log = _last_successful_runtime_return(history)
    if final_log is None:
        return ResultShapeValidation(True)
    content = str(final_content if final_content is not None else final_log.result.get("output") or "")
    aliases = _output_aliases(history)
    normalized_content = _normalize_alias(content)
    if not content.strip() and _has_non_empty_data_result(history):
        return ResultShapeValidation(
            False,
            "Final response was empty even though tool output was available.",
            {"final_output_validation": "empty_final_output"},
        )
    if normalized_content in aliases:
        return ResultShapeValidation(
            False,
            f"Final response returned unresolved output alias: {content.strip()}",
            {"final_output_validation": "literal_reference_output", "literal_alias": content.strip()},
        )
    quoted_alias = _quoted_alias_in_content(content, aliases)
    if quoted_alias:
        return ResultShapeValidation(
            False,
            f"Final response included unresolved output alias: {quoted_alias}",
            {"final_output_validation": "literal_reference_output", "literal_alias": quoted_alias},
        )
    placeholder_alias = _placeholder_alias_in_content(content, aliases)
    if placeholder_alias:
        return ResultShapeValidation(
            False,
            f"Final response included unresolved output placeholder: {placeholder_alias}",
            {"final_output_validation": "literal_reference_output", "literal_alias": placeholder_alias},
        )
    if enforce_scalar_text and _is_count_goal(goal):
        numbers = re.findall(r"\b-?\d+(?:\.\d+)?\b", content)
        if content.strip() and len(numbers) != 1 and len(set(numbers)) != 1:
            return ResultShapeValidation(
                False,
                "Count request final response must include exactly one numeric scalar.",
                {"final_output_validation": "count_not_scalar", "numeric_token_count": len(numbers)},
            )
    if _is_list_or_table_goal(goal) and _raw_parseable_shell_table_returned(content):
        return ResultShapeValidation(
            False,
            "List/table request returned raw shell table text instead of structured local formatting.",
            {"final_output_validation": "raw_parseable_table_output", "expected_shape": "formatted table or artifact"},
        )
    if _looks_like_raw_json(content) and not allow_raw_json:
        return ResultShapeValidation(
            False,
            "Final response returned raw JSON instead of a readable local presentation.",
            {"final_output_validation": "raw_json_output", "expected_shape": "readable markdown presentation"},
        )
    return ResultShapeValidation(True)


def _is_count_goal(goal: str) -> bool:
    """Handle the internal is count goal helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._is_count_goal.
    """
    return is_scalar_count_goal(goal)


def _is_list_or_table_goal(goal: str) -> bool:
    """Handle the internal is list or table goal helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._is_list_or_table_goal.
    """
    text = str(goal or "").lower()
    if _is_count_goal(text):
        return False
    return bool(re.search(r"\b(?:list|show|display|return|give\s+me|all)\b", text))


def _is_status_goal(goal: str) -> bool:
    """Handle the internal is status goal helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._is_status_goal.
    """
    return bool(re.search(r"\b(?:status|summary|overview|health|availability|utilization|usage)\b", str(goal or ""), re.IGNORECASE))


def _explicit_json_goal(goal: str) -> bool:
    """Handle the internal explicit json goal helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._explicit_json_goal.
    """
    return bool(re.search(r"\b(?:json|raw\s+json)\b", str(goal or ""), re.IGNORECASE))


def _looks_like_raw_json(content: str) -> bool:
    """Handle the internal looks like raw json helper path for this module.

    Inputs:
        Receives content for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._looks_like_raw_json.
    """
    text = str(content or "").strip()
    if not text or text[0] not in "[{":
        return False
    try:
        import json

        json.loads(text)
    except Exception:
        return False
    return True


def _raw_parseable_shell_table_returned(content: str) -> bool:
    """Handle the internal raw parseable shell table returned helper path for this module.

    Inputs:
        Receives content for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._raw_parseable_shell_table_returned.
    """
    text = str(content or "").strip()
    if not text or text.startswith("|"):
        return False
    rows = parse_shell_table(text)
    return len(rows) > 0


def _validate_grouped_count_result(goal: str, history: list[StepLog]) -> ResultShapeValidation:
    """Handle the internal validate grouped count result helper path for this module.

    Inputs:
        Receives goal, history for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._validate_grouped_count_result.
    """
    expected_group = grouped_count_field_for_goal(goal)
    for item in reversed(history):
        if not item.success or item.step.action in {"runtime.return", "text.format", "fs.write"}:
            continue
        if not isinstance(item.result, dict):
            continue
        grouped = item.result.get("grouped")
        if isinstance(grouped, dict):
            group_by = str(item.result.get("group_by") or item.step.args.get("group_by") or "").strip().lower().replace(" ", "_")
            if expected_group and group_by and group_by != expected_group:
                return ResultShapeValidation(
                    False,
                    f"Grouped count request expected grouping by {expected_group}, got {group_by}.",
                    {
                        "expected_shape": "grouped count table",
                        "expected_group_by": expected_group,
                        "actual_group_by": group_by,
                    },
                )
            return ResultShapeValidation(
                True,
                metadata={
                    "primary_result_validation": "grouped_count_table",
                    "tool": item.step.action,
                    "group_by": group_by or expected_group,
                    "group_count": len(grouped),
                    "expected_shape": "grouped count table",
                },
            )
        groups = item.result.get("groups")
        if isinstance(groups, list):
            return ResultShapeValidation(
                True,
                metadata={
                    "primary_result_validation": "grouped_count_table",
                    "tool": item.step.action,
                    "group_count": len(groups),
                    "expected_shape": "grouped count table",
                },
            )
    return ResultShapeValidation(
        False,
        "Grouped count request did not return grouped results.",
        {
            "final_output_validation": "grouped_count_missing",
            "expected_group_by": expected_group,
            "expected_shape": "grouped count table",
        },
    )


def _validate_count_final_dataflow(history: list[StepLog]) -> ResultShapeValidation:
    """Handle the internal validate count final dataflow helper path for this module.

    Inputs:
        Receives history for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._validate_count_final_dataflow.
    """
    final_log = _last_successful_runtime_return(history)
    if final_log is None:
        return ResultShapeValidation(True)
    value = final_log.result.get("value")
    if _is_collection_like(value):
        return ResultShapeValidation(
            False,
            "Scalar request returned a collection instead of one numeric value.",
            {"final_output_validation": "scalar_returned_collection", "expected_shape": "single numeric scalar"},
        )
    if _ref_points_to_collection(final_log.step.args.get("value"), history):
        return ResultShapeValidation(
            False,
            "Scalar request final dataflow points to a collection output.",
            {"final_output_validation": "scalar_returned_collection", "expected_shape": "single numeric scalar"},
        )
    return ResultShapeValidation(True)


def _ref_points_to_collection(value: Any, history: list[StepLog]) -> bool:
    """Handle the internal ref points to collection helper path for this module.

    Inputs:
        Receives value, history for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._ref_points_to_collection.
    """
    if isinstance(value, list):
        return True
    if isinstance(value, dict) and "$ref" not in value:
        return True
    ref = _parse_ref(value)
    if ref is None:
        return False
    alias, path = ref
    producer = _producer_for_alias(history, alias)
    if producer is None:
        return False
    normalized_path = normalize_tool_ref_path(producer.step.action, path)
    if (
        producer.step.action == "sql.query"
        and str(normalized_path or "").split(".", 1)[0] == "rows"
        and _validate_sql_count_result(producer).success
    ):
        return False
    if is_collection_ref(producer.step.action, normalized_path):
        return True
    if producer.step.action == "text.format" and (normalized_root := normalized_path):
        if normalized_root.split(".", 1)[0] == "content":
            return _ref_points_to_collection(producer.step.args.get("source"), history)
    return False


def _parse_ref(value: Any) -> tuple[str, str | None] | None:
    """Handle the internal parse ref helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._parse_ref.
    """
    if isinstance(value, dict) and "$ref" in value:
        alias = str(value.get("$ref") or "").strip().lstrip("$")
        path = value.get("path")
        return (alias, None if path is None else str(path).strip() or None) if alias else None
    if isinstance(value, str) and value.strip().startswith("$"):
        text = value.strip()[1:]
        alias, _separator, path = text.partition(".")
        alias = alias.strip()
        return (alias, path.strip() or None) if alias else None
    return None


def _producer_for_alias(history: list[StepLog], alias: str) -> StepLog | None:
    """Handle the internal producer for alias helper path for this module.

    Inputs:
        Receives history, alias for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._producer_for_alias.
    """
    normalized = _normalize_alias(alias)
    for item in reversed(history):
        if not item.success:
            continue
        output = str(item.step.output or "").strip()
        candidates = {_normalize_alias(output), _normalize_alias(str(item.step.id))}
        if normalized in candidates:
            return item
    return None


def _last_successful_scalar_producer(goal: str, history: list[StepLog]) -> tuple[StepLog | None, str | None, Any]:
    """Handle the internal last successful scalar producer helper path for this module.

    Inputs:
        Receives goal, history for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._last_successful_scalar_producer.
    """
    for item in reversed(history):
        if not item.success or item.step.action in {"runtime.return", "text.format"}:
            continue
        if not isinstance(item.result, dict):
            continue
        fields = _scalar_candidate_fields(item.step.action, goal=goal)
        for field in fields:
            value = _value_at_path(item.result, field)
            if value is not None:
                return item, field, value
    return None, None, None


def _scalar_candidate_fields(tool: str, *, goal: str) -> list[str]:
    """Handle the internal scalar candidate fields helper path for this module.

    Inputs:
        Receives tool, goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._scalar_candidate_fields.
    """
    fields: list[str] = []
    preferred = scalar_field_for_tool(tool, goal=goal)
    if preferred:
        fields.append(preferred)
    shape = TOOL_RESULT_SHAPES.get(tool)
    if shape is not None:
        for field in shape.scalar_fields:
            if field not in fields and field != "returned_count":
                fields.append(field)
    return fields


def _value_at_path(value: Any, path: str | None) -> Any:
    """Handle the internal value at path helper path for this module.

    Inputs:
        Receives value, path for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._value_at_path.
    """
    current = value
    if not path:
        return current
    for part in str(path).split("."):
        if isinstance(current, dict):
            current = current.get(part)
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            current = current[index] if 0 <= index < len(current) else None
            continue
        return None
    return current


def _is_collection_like(value: Any) -> bool:
    """Handle the internal is collection like helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._is_collection_like.
    """
    if isinstance(value, (list, tuple, set)):
        return True
    if isinstance(value, dict):
        return True
    return False


def _last_successful_sql_query(history: list[StepLog]) -> StepLog | None:
    """Handle the internal last successful sql query helper path for this module.

    Inputs:
        Receives history for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._last_successful_sql_query.
    """
    for item in reversed(history):
        if item.success and item.step.action == "sql.query":
            return item
    return None


def _last_successful_runtime_return(history: list[StepLog]) -> StepLog | None:
    """Handle the internal last successful runtime return helper path for this module.

    Inputs:
        Receives history for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._last_successful_runtime_return.
    """
    for item in reversed(history):
        if item.success and item.step.action == "runtime.return":
            return item
    return None


def _output_aliases(history: list[StepLog]) -> set[str]:
    """Handle the internal output aliases helper path for this module.

    Inputs:
        Receives history for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._output_aliases.
    """
    aliases: set[str] = set()
    for item in history:
        output = str(item.step.output or "").strip()
        if output:
            aliases.add(_normalize_alias(output))
    return aliases


def _normalize_alias(value: str) -> str:
    """Handle the internal normalize alias helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._normalize_alias.
    """
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def _quoted_alias_in_content(content: str, aliases: set[str]) -> str | None:
    """Handle the internal quoted alias in content helper path for this module.

    Inputs:
        Receives content, aliases for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._quoted_alias_in_content.
    """
    for match in re.finditer(r'"([^"]+)"|`([^`]+)`', str(content or "")):
        value = match.group(1) or match.group(2) or ""
        if _normalize_alias(value) in aliases:
            return value
    return None


def _placeholder_alias_in_content(content: str, aliases: set[str]) -> str | None:
    """Handle the internal placeholder alias in content helper path for this module.

    Inputs:
        Receives content, aliases for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._placeholder_alias_in_content.
    """
    for match in re.finditer(r"\{([A-Za-z0-9_-]+)\}", str(content or "")):
        value = match.group(1) or ""
        if _normalize_alias(value) in aliases:
            return value
    return None


def _has_non_empty_data_result(history: list[StepLog]) -> bool:
    """Handle the internal has non empty data result helper path for this module.

    Inputs:
        Receives history for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._has_non_empty_data_result.
    """
    for item in history:
        if not item.success or item.step.action == "runtime.return":
            continue
        result = item.result
        if not isinstance(result, dict):
            continue
        if isinstance(result.get("rows"), list) and len(result["rows"]) > 0:
            return True
        if isinstance(result.get("content"), str) and result["content"].strip():
            return True
        for key in ("entries", "matches", "jobs", "nodes", "partitions"):
            if isinstance(result.get(key), list) and result[key]:
                return True
        if isinstance(result.get("stdout"), str) and result["stdout"].strip():
            return True
        if isinstance(result.get("catalog"), dict) and result["catalog"].get("tables"):
            return True
    return False


def _row_columns(rows: list[Any]) -> list[str]:
    """Handle the internal row columns helper path for this module.

    Inputs:
        Receives rows for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._row_columns.
    """
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows[:5]:
        if not isinstance(row, dict):
            continue
        for key in row:
            column = str(key)
            if column not in seen:
                columns.append(column)
                seen.add(column)
    return columns


def _is_numeric_scalar(value: Any) -> bool:
    """Handle the internal is numeric scalar helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.result_shape._is_numeric_scalar.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        return bool(re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", value))
    return False
