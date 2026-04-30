"""OpenFABRIC Runtime Module: aor_runtime.runtime.output_shape

Purpose:
    Infer expected user-output shape from goals and tool contracts.

Responsibilities:
    Classify scalar, grouped-count, table/list, file, status, and text output expectations.

Data flow / Interfaces:
    Provides shared helpers for planners, auto-artifacts, and result-shape validators.

Boundaries:
    Prevents scalar/table intent drift by centralizing count and grouped-count detection.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from aor_runtime.runtime.tool_output_contracts import TOOL_OUTPUT_CONTRACTS, root_path


GoalOutputKind = Literal["scalar", "table", "file", "text", "json", "status", "unknown"]


COUNT_GOAL_RE = re.compile(r"\b(?:count|how\s+many|number\s+of|total\s+number\s+of)\b", re.IGNORECASE)
EXPORT_GOAL_RE = re.compile(r"\b(?:save|write|export)\b.+\b[\w.-]+\.(?:txt|csv|json|md|markdown)\b", re.IGNORECASE)


SHAPE_KIND_ALIASES: dict[str, GoalOutputKind] = {
    "bool": "status",
    "boolean": "status",
    "count": "scalar",
    "csv": "text",
    "integer": "scalar",
    "int": "scalar",
    "list": "table",
    "markdown": "text",
    "md": "text",
    "number": "scalar",
    "numeric": "scalar",
    "record": "table",
    "records": "table",
    "row": "table",
    "rows": "table",
    "string": "text",
}


@dataclass(frozen=True)
class ToolResultShape:
    """Represent tool result shape within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ToolResultShape.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.output_shape.ToolResultShape and related tests.
    """
    scalar_fields: tuple[str, ...] = ()
    collection_fields: tuple[str, ...] = ()
    text_fields: tuple[str, ...] = ()
    file_fields: tuple[str, ...] = ()
    default_field: str | None = None


@dataclass(frozen=True)
class GoalOutputContract:
    """Represent goal output contract within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by GoalOutputContract.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.output_shape.GoalOutputContract and related tests.
    """
    kind: GoalOutputKind = "unknown"
    format: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class FinalOutputContractResult:
    """Represent final output contract result within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by FinalOutputContractResult.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.output_shape.FinalOutputContractResult and related tests.
    """
    success: bool
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


TOOL_RESULT_SHAPES: dict[str, ToolResultShape] = {
    name: ToolResultShape(
        scalar_fields=contract.scalar_paths,
        collection_fields=contract.collection_paths,
        text_fields=contract.text_paths,
        file_fields=contract.file_paths,
        default_field=contract.default_path,
    )
    for name, contract in TOOL_OUTPUT_CONTRACTS.items()
}


def normalize_shape_kind(value: Any) -> GoalOutputKind:
    """Normalize shape kind for the surrounding runtime workflow.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_shape.normalize_shape_kind.
    """
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(value or "unknown").strip().lower()).strip("_")
    if normalized in {"scalar", "table", "file", "text", "json", "status"}:
        return normalized  # type: ignore[return-value]
    return SHAPE_KIND_ALIASES.get(normalized, "unknown")


def infer_goal_output_contract(goal: str, expected_kind: Any = None, *, output_format: str | None = None) -> GoalOutputContract:
    """Infer goal output contract for the surrounding runtime workflow.

    Inputs:
        Receives goal, expected_kind, output_format for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_shape.infer_goal_output_contract.
    """
    expected = normalize_shape_kind(expected_kind)
    goal_text = str(goal or "")
    if is_grouped_count_goal(goal_text):
        return GoalOutputContract(kind="table", format=output_format, reason="grouped_count_goal")
    if is_scalar_count_goal(goal_text):
        return GoalOutputContract(kind="scalar", format=output_format, reason="count_goal")
    if EXPORT_GOAL_RE.search(goal_text):
        return GoalOutputContract(kind="file", format=output_format, reason="export_goal")
    if expected != "unknown":
        return GoalOutputContract(kind=expected, format=output_format, reason="planner_expected_shape")
    return GoalOutputContract(kind="unknown", format=output_format, reason="")


def scalar_field_for_tool(tool: str, *, goal: str = "") -> str | None:
    """Scalar field for tool for the surrounding runtime workflow.

    Inputs:
        Receives tool, goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_shape.scalar_field_for_tool.
    """
    shape = TOOL_RESULT_SHAPES.get(tool)
    if shape is None:
        return None
    if tool == "shell.exec":
        return "returncode" if is_shell_status_goal(goal) else "stdout"
    if tool == "slurm.metrics":
        text = str(goal or "").lower()
        if "pending" in text:
            return "pending_jobs"
        if "running" in text:
            return "running_jobs"
        if "node" in text:
            return "node_count"
        if "queue" in text or "job" in text:
            return "queue_count"
    for field in shape.scalar_fields:
        if field in {"returned_count"}:
            continue
        return field
    return shape.scalar_fields[0] if shape.scalar_fields else None


def is_shell_status_goal(goal: str) -> bool:
    """Is shell status goal for the surrounding runtime workflow.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_shape.is_shell_status_goal.
    """
    text = str(goal or "").lower()
    return bool(
        re.search(
            r"\b(?:exit\s*code|return\s*code|returncode|status\s*code|command\s+status|did\s+.+\s+succeed)\b",
            text,
        )
    )


def is_collection_ref(tool: str, path: str | None) -> bool:
    """Is collection ref for the surrounding runtime workflow.

    Inputs:
        Receives tool, path for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_shape.is_collection_ref.
    """
    shape = TOOL_RESULT_SHAPES.get(tool)
    if shape is None:
        return False
    root = root_path(path)
    return bool(root and root in shape.collection_fields)


def is_scalar_ref(tool: str, path: str | None) -> bool:
    """Is scalar ref for the surrounding runtime workflow.

    Inputs:
        Receives tool, path for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_shape.is_scalar_ref.
    """
    shape = TOOL_RESULT_SHAPES.get(tool)
    if shape is None:
        return False
    root = root_path(path)
    return bool(root and root in shape.scalar_fields)


def validate_final_output_contract(
    goal: str,
    history: list[Any],
    *,
    final_content: str | None = None,
) -> FinalOutputContractResult:
    """Validate final output contract for the surrounding runtime workflow.

    Inputs:
        Receives goal, history, final_content for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_shape.validate_final_output_contract.
    """
    contract = infer_goal_output_contract(goal)
    if contract.kind != "scalar":
        return FinalOutputContractResult(True, metadata={"expected_shape": contract.kind})

    final_log = _last_successful_action(history, "runtime.return")
    if final_log is None:
        return FinalOutputContractResult(True, metadata={"expected_shape": "scalar"})
    content = str(final_content if final_content is not None else getattr(final_log, "result", {}).get("output") or "")
    if not content.strip():
        return FinalOutputContractResult(
            False,
            "Scalar request final response was empty.",
            {"final_output_validation": "empty_scalar_output", "expected_shape": "single numeric scalar"},
        )
    if _looks_like_json_collection(content):
        return FinalOutputContractResult(
            False,
            "Scalar request returned a collection instead of one numeric value.",
            {"final_output_validation": "scalar_returned_collection", "expected_shape": "single numeric scalar"},
        )
    numeric_tokens = re.findall(r"-?\d+(?:\.\d+)?", content)
    if final_content is not None:
        if numeric_tokens and len(set(numeric_tokens)) == 1:
            return FinalOutputContractResult(True, metadata={"expected_shape": "single numeric scalar"})
        return FinalOutputContractResult(
            False,
            "Scalar request final response must contain exactly one numeric value.",
            {
                "final_output_validation": "count_not_scalar",
                "expected_shape": "single numeric scalar",
                "numeric_token_count": len(numeric_tokens),
            },
        )
    if len(numeric_tokens) != 1 or not re.fullmatch(r"\s*(?:count\s*[:=]\s*)?-?\d+(?:\.\d+)?\s*\.?\s*", content, re.IGNORECASE):
        return FinalOutputContractResult(
            False,
            "Scalar request final response must contain exactly one numeric value.",
            {
                "final_output_validation": "count_not_scalar",
                "expected_shape": "single numeric scalar",
                "numeric_token_count": len(numeric_tokens),
            },
        )
    return FinalOutputContractResult(True, metadata={"expected_shape": "single numeric scalar"})


def grouped_count_field_for_goal(goal: str) -> str | None:
    """Grouped count field for goal for the surrounding runtime workflow.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_shape.grouped_count_field_for_goal.
    """
    text = str(goal or "").lower()
    if not COUNT_GOAL_RE.search(text):
        return None
    field_patterns = {
        "partition": r"(?:slurm\s+)?partitions?",
        "state": r"states?",
        "user": r"users?",
        "node": r"nodes?",
        "job_name": r"job\s+names?",
    }
    for field, pattern in field_patterns.items():
        if re.search(rf"\b(?:by|per)\s+{pattern}\b", text):
            return field
        if re.search(rf"\b(?:in|for|across)?\s*(?:each|every)\s+{pattern}\b", text):
            return field
        if re.search(rf"\bfor\s+each\s+{pattern}\b", text):
            return field
        if re.search(rf"\b{pattern}\s+separately\b", text):
            return field
    if "separately" in text:
        for field, pattern in field_patterns.items():
            if re.search(rf"\b{pattern}\b", text):
                return field
    return None


def is_grouped_count_goal(goal: str) -> bool:
    """Is grouped count goal for the surrounding runtime workflow.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_shape.is_grouped_count_goal.
    """
    return grouped_count_field_for_goal(goal) is not None


def is_scalar_count_goal(goal: str) -> bool:
    """Is scalar count goal for the surrounding runtime workflow.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_shape.is_scalar_count_goal.
    """
    text = str(goal or "").lower()
    if not COUNT_GOAL_RE.search(text):
        return False
    if is_grouped_count_goal(text):
        return False
    if re.search(r"\bcount\b[^.?!]*\b(?:by|per)\b|\b(?:count|counts?)\s+(?:by|per)\b|\bgroup(?:ed)?\s+by\b|\bby\s+number\s+of\b", text):
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
    if re.search(r"\b(?:top|rank|ranked|most|highest|largest)\b", text) and re.search(r"\b(?:show|list|display|return)\b", text):
        return False
    if re.search(r"\b(?:show|list|display|return)\b", text) and re.search(r"\b(?:counts|study counts|series counts)\b", text):
        return False
    return True


def _is_scalar_count_goal(goal: str) -> bool:
    """Handle the internal is scalar count goal helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_shape._is_scalar_count_goal.
    """
    return is_scalar_count_goal(goal)


def _last_successful_action(history: list[Any], action: str) -> Any | None:
    """Handle the internal last successful action helper path for this module.

    Inputs:
        Receives history, action for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_shape._last_successful_action.
    """
    for item in reversed(history):
        if bool(getattr(item, "success", False)) and getattr(getattr(item, "step", None), "action", None) == action:
            return item
    return None


def _looks_like_json_collection(content: str) -> bool:
    """Handle the internal looks like json collection helper path for this module.

    Inputs:
        Receives content for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_shape._looks_like_json_collection.
    """
    text = str(content or "").strip()
    if not text or text[0] not in "[{":
        return False
    try:
        parsed = json.loads(text)
    except Exception:
        return False
    if isinstance(parsed, list):
        return True
    if isinstance(parsed, dict):
        return any(isinstance(value, list) for value in parsed.values())
    return False
