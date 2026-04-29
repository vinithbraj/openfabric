from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from aor_runtime.core.contracts import StepLog
from aor_runtime.runtime.output_envelope import parse_shell_table
from aor_runtime.runtime.output_shape import validate_final_output_contract


COUNT_GOAL_RE = re.compile(r"\b(?:count|how\s+many|number\s+of|total\s+number\s+of)\b", re.IGNORECASE)
LIST_GOAL_RE = re.compile(r"\b(?:list|show|display|return)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ResultShapeValidation:
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
    goal_text = str(goal or "")
    if not _is_count_goal(goal_text):
        final_validation = _validate_final_output(goal_text, history, final_content=final_content, allow_raw_json=allow_raw_json)
        if not final_validation.success:
            return final_validation
        return ResultShapeValidation(True)

    sql_log = _last_successful_sql_query(history)
    if sql_log is None:
        final_validation = _validate_final_output(goal_text, history, final_content=final_content, allow_raw_json=allow_raw_json)
        if not final_validation.success:
            return final_validation
        contract_validation = validate_final_output_contract(goal_text, history, final_content=final_content)
        if not contract_validation.success:
            return ResultShapeValidation(contract_validation.success, contract_validation.reason, contract_validation.metadata)
        return ResultShapeValidation(True)
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
    final_validation = _validate_final_output(goal_text, history, final_content=final_content, allow_raw_json=allow_raw_json)
    if not final_validation.success:
        return final_validation
    contract_validation = validate_final_output_contract(goal_text, history, final_content=final_content)
    if not contract_validation.success:
        return ResultShapeValidation(contract_validation.success, contract_validation.reason, contract_validation.metadata)
    return ResultShapeValidation(True)


def _validate_final_output(
    goal: str,
    history: list[StepLog],
    *,
    final_content: str | None = None,
    allow_raw_json: bool = False,
) -> ResultShapeValidation:
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
    if _is_count_goal(goal):
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
    text = str(goal or "").lower()
    if not COUNT_GOAL_RE.search(text):
        return False
    if re.search(r"\bcount\b.+\bby\b|\b(?:count|counts?)\s+(?:by|per)\b|\bgroup(?:ed)?\s+by\b|\bby\s+number\s+of\b", text):
        return False
    if re.search(r"\b(?:show|list|display)\b", text) and re.search(
        r"\b(?:ids?|identifiers?)\b.*\bnumber\s+of\b|\bnumber\s+of\b.*\b(?:for|per)\b",
        text,
    ):
        return False
    if re.search(r"\b(?:show|list|display)\b", text) and re.search(
        r"\b[a-z0-9_]+\s+count\s+(?:greater|less|more|fewer|over|under|above|below|>=|<=|>|<)\b",
        text,
    ):
        return False
    if re.search(r"\b(?:top|rank|ranked|most|highest|largest)\b", text) and re.search(
        r"\b(?:show|list|display|return)\b", text
    ):
        return False
    if re.search(r"\b(?:show|list|display|return)\b", text) and re.search(
        r"\b(?:counts|study counts|series counts)\b", text
    ):
        return False
    return True


def _is_list_or_table_goal(goal: str) -> bool:
    text = str(goal or "").lower()
    if _is_count_goal(text):
        return False
    return bool(re.search(r"\b(?:list|show|display|return|give\s+me|all)\b", text))


def _is_status_goal(goal: str) -> bool:
    return bool(re.search(r"\b(?:status|summary|overview|health|availability|utilization|usage)\b", str(goal or ""), re.IGNORECASE))


def _explicit_json_goal(goal: str) -> bool:
    return bool(re.search(r"\b(?:json|raw\s+json)\b", str(goal or ""), re.IGNORECASE))


def _looks_like_raw_json(content: str) -> bool:
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
    text = str(content or "").strip()
    if not text or text.startswith("|"):
        return False
    rows = parse_shell_table(text)
    return len(rows) > 0


def _last_successful_sql_query(history: list[StepLog]) -> StepLog | None:
    for item in reversed(history):
        if item.success and item.step.action == "sql.query":
            return item
    return None


def _last_successful_runtime_return(history: list[StepLog]) -> StepLog | None:
    for item in reversed(history):
        if item.success and item.step.action == "runtime.return":
            return item
    return None


def _output_aliases(history: list[StepLog]) -> set[str]:
    aliases: set[str] = set()
    for item in history:
        output = str(item.step.output or "").strip()
        if output:
            aliases.add(_normalize_alias(output))
    return aliases


def _normalize_alias(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def _quoted_alias_in_content(content: str, aliases: set[str]) -> str | None:
    for match in re.finditer(r'"([^"]+)"|`([^`]+)`', str(content or "")):
        value = match.group(1) or match.group(2) or ""
        if _normalize_alias(value) in aliases:
            return value
    return None


def _placeholder_alias_in_content(content: str, aliases: set[str]) -> str | None:
    for match in re.finditer(r"\{([A-Za-z0-9_-]+)\}", str(content or "")):
        value = match.group(1) or ""
        if _normalize_alias(value) in aliases:
            return value
    return None


def _has_non_empty_data_result(history: list[StepLog]) -> bool:
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
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        return bool(re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", value))
    return False
