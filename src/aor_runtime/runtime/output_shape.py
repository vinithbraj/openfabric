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
OutputCardinality = Literal["single", "multi_scalar", "grouped", "collection", "sectioned", "unknown"]
OutputRenderStyle = Literal["scalar", "metric_table", "record_table", "key_value", "bullets", "sectioned", "unknown"]
OutputIntentSource = Literal["llm", "deterministic_backup", "planner_expected", "merged", "unknown"]


COUNT_GOAL_RE = re.compile(r"\b(?:count|how\s+many|number\s+of|total\s+number\s+of)\b", re.IGNORECASE)
EXPORT_GOAL_RE = re.compile(r"\b(?:save|write|export)\b.+\b[\w.-]+\.(?:txt|csv|json|md|markdown)\b", re.IGNORECASE)
MULTI_COUNT_SEPARATOR_RE = re.compile(r"\bcount(?:s|ing)?\b[^.?!]*(?:,|\band\b)", re.IGNORECASE)


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
class ResolvedOutputIntent:
    """Represent the resolved user-visible output intent.

    Responsibilities:
        Captures LLM-proposed and runtime-canonicalized output shape facts.

    Data flow / Interfaces:
        Created by resolve_output_intent and consumed by planners, validators, and renderers.

    Used by:
        Output-shape inference, action planning, auto-artifacts, and result-shape validation.
    """
    kind: GoalOutputKind = "unknown"
    cardinality: OutputCardinality = "unknown"
    render_style: OutputRenderStyle = "unknown"
    result_entities: tuple[str, ...] = ()
    group_by: tuple[str, ...] = ()
    format: str | None = None
    source: OutputIntentSource = "unknown"
    corrections: tuple[str, ...] = ()
    reason: str = ""

    def as_contract(self) -> GoalOutputContract:
        """Return the legacy goal-output contract view.

        Inputs:
            Uses the resolved output intent instance.

        Returns:
            GoalOutputContract with kind/format/reason.

        Used by:
            Compatibility callers of infer_goal_output_contract.
        """
        return GoalOutputContract(kind=self.kind, format=self.format, reason=self.reason or self.cardinality)


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


def infer_goal_output_contract(
    goal: str,
    expected_kind: Any = None,
    *,
    output_format: str | None = None,
    semantic_output: Any | None = None,
) -> GoalOutputContract:
    """Infer goal output contract for the surrounding runtime workflow.

    Inputs:
        Receives goal, expected_kind, output_format for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_shape.infer_goal_output_contract.
    """
    return resolve_output_intent(
        goal,
        semantic_output=semantic_output,
        planner_expected_shape=expected_kind,
        output_format=output_format,
    ).as_contract()


def resolve_output_intent(
    goal: str,
    *,
    semantic_output: Any | None = None,
    planner_expected_shape: Any | None = None,
    output_format: str | None = None,
) -> ResolvedOutputIntent:
    """Resolve LLM-informed output intent into an enforceable runtime contract.

    Inputs:
        Receives the prompt, optional semantic output contract, planner shape, and output format.

    Returns:
        A canonical output intent with runtime corrections recorded.

    Used by:
        Action planner validation, auto-artifact decisions, and result-shape validation.
    """
    goal_text = str(goal or "")
    output = _output_mapping(semantic_output)
    kind = normalize_shape_kind(output.get("kind"))
    cardinality = _normalize_cardinality(output.get("cardinality"))
    render_style = _normalize_render_style(output.get("render_style"))
    result_entities = tuple(_string_list(output.get("result_entities")))
    group_by = tuple(_string_list(output.get("group_by")))
    corrections: list[str] = []
    raw_source = str(output.get("source") or "").strip().lower()
    if raw_source in {"llm", "deterministic_backup", "planner_expected", "merged"}:
        source: OutputIntentSource = raw_source  # type: ignore[assignment]
    else:
        source = "llm" if output else "unknown"

    if EXPORT_GOAL_RE.search(goal_text):
        if kind != "file":
            corrections.append("explicit_export_forces_file")
        return ResolvedOutputIntent(
            kind="file",
            cardinality=cardinality if cardinality != "unknown" else "collection",
            render_style=render_style if render_style != "unknown" else "record_table",
            result_entities=result_entities,
            group_by=group_by,
            format=output_format or str(output.get("format") or "") or None,
            source="merged" if output else "deterministic_backup",
            corrections=tuple(corrections),
            reason="export_goal",
        )

    if _looks_like_sql_explain_only_goal(goal_text):
        if kind not in {"unknown", "text"}:
            corrections.append("explain_only_forces_text")
        return ResolvedOutputIntent(
            kind="text",
            cardinality=cardinality if cardinality != "unknown" else "single",
            render_style=render_style if render_style != "unknown" else "bullets",
            result_entities=result_entities,
            group_by=group_by,
            format=output_format or str(output.get("format") or "") or None,
            source="merged" if output else "deterministic_backup",
            corrections=tuple(corrections),
            reason="sql_explain_only",
        )

    fallback = _deterministic_output_backup(goal_text)
    if fallback.group_by and not group_by:
        group_by = fallback.group_by
    if fallback.result_entities and not result_entities:
        result_entities = fallback.result_entities
    if kind == "unknown" and fallback.kind != "unknown":
        kind = fallback.kind
        cardinality = fallback.cardinality
        render_style = fallback.render_style
        source = "deterministic_backup"
    if cardinality == "unknown" and fallback.cardinality != "unknown":
        cardinality = fallback.cardinality
    if render_style == "unknown" and fallback.render_style != "unknown":
        render_style = fallback.render_style

    expected = normalize_shape_kind(planner_expected_shape)
    if kind == "unknown" and expected != "unknown":
        kind = expected
        source = "planner_expected"

    if cardinality in {"multi_scalar", "grouped", "collection", "sectioned"} and kind == "scalar":
        kind = "table" if cardinality != "sectioned" else "text"
        corrections.append("non_single_cardinality_forces_non_scalar")
    if group_by and cardinality not in {"grouped", "sectioned"}:
        cardinality = "grouped"
        corrections.append("group_by_forces_grouped_cardinality")
    if group_by and kind == "scalar":
        kind = "table"
        corrections.append("group_by_forces_table")
    if len(result_entities) > 1 and cardinality == "single":
        cardinality = "multi_scalar"
        corrections.append("multiple_entities_force_multi_scalar")
    if len(result_entities) > 1 and kind == "scalar":
        kind = "table"
        corrections.append("multiple_entities_force_table")
    if kind == "unknown" and cardinality in {"multi_scalar", "grouped", "collection"}:
        kind = "table"
    if kind == "unknown" and cardinality == "single":
        kind = "scalar"
    if kind == "json" and "json" not in goal_text.lower():
        kind = "table" if cardinality in {"multi_scalar", "grouped", "collection"} else "text"
        corrections.append("implicit_json_normalized_to_user_mode")
    if cardinality in {"multi_scalar", "grouped"} and render_style == "scalar":
        render_style = "metric_table"
        corrections.append("multi_value_forces_metric_table")
    if render_style == "unknown":
        render_style = _render_style_for(kind, cardinality)
    if cardinality == "unknown":
        cardinality = _cardinality_for_kind(kind)
    if source != "unknown" and corrections:
        source = "merged"
    return ResolvedOutputIntent(
        kind=kind,
        cardinality=cardinality,
        render_style=render_style,
        result_entities=result_entities,
        group_by=group_by,
        format=output_format or str(output.get("format") or "") or None,
        source=source,
        corrections=tuple(corrections),
        reason=str(output.get("reason") or fallback.reason or ""),
    )


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


def _output_mapping(value: Any | None) -> dict[str, Any]:
    """Coerce semantic output contract objects to dictionaries.

    Inputs:
        Receives a Pydantic model, dataclass-like object, dict, or None.

    Returns:
        A plain dictionary of output intent fields.

    Used by:
        resolve_output_intent.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dict(dumped) if isinstance(dumped, dict) else {}
    return {
        key: getattr(value, key)
        for key in ("kind", "cardinality", "render_style", "result_entities", "group_by", "format", "reason")
        if hasattr(value, key)
    }


def _string_list(value: Any) -> list[str]:
    """Normalize arbitrary list-like output-intent fields to strings.

    Inputs:
        Receives a scalar or sequence-like value.

    Returns:
        Ordered non-empty string values.

    Used by:
        resolve_output_intent and deterministic backup helpers.
    """
    if value is None:
        return []
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        raw = [value]
    seen: set[str] = set()
    values: list[str] = []
    for item in raw:
        text = str(item or "").strip().lower()
        if text and text not in seen:
            seen.add(text)
            values.append(text)
    return values


def _normalize_cardinality(value: Any) -> OutputCardinality:
    """Normalize output cardinality labels.

    Inputs:
        Receives a raw cardinality value.

    Returns:
        Canonical OutputCardinality.

    Used by:
        resolve_output_intent.
    """
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(value or "unknown").strip().lower()).strip("_")
    aliases = {
        "one": "single",
        "many_scalars": "multi_scalar",
        "multiple_scalars": "multi_scalar",
        "groups": "grouped",
        "rows": "collection",
        "records": "collection",
        "list": "collection",
        "sections": "sectioned",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in {"single", "multi_scalar", "grouped", "collection", "sectioned", "unknown"}:
        return normalized  # type: ignore[return-value]
    return "unknown"


def _normalize_render_style(value: Any) -> OutputRenderStyle:
    """Normalize output render-style labels.

    Inputs:
        Receives a raw render style.

    Returns:
        Canonical OutputRenderStyle.

    Used by:
        resolve_output_intent.
    """
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(value or "unknown").strip().lower()).strip("_")
    aliases = {
        "table": "record_table",
        "metrics": "metric_table",
        "metric": "metric_table",
        "kv": "key_value",
        "value": "scalar",
        "sections": "sectioned",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in {"scalar", "metric_table", "record_table", "key_value", "bullets", "sectioned", "unknown"}:
        return normalized  # type: ignore[return-value]
    return "unknown"


def _render_style_for(kind: GoalOutputKind, cardinality: OutputCardinality) -> OutputRenderStyle:
    """Choose a default render style from canonical output shape.

    Inputs:
        Receives output kind and cardinality.

    Returns:
        Render style suitable for deterministic presentation.

    Used by:
        resolve_output_intent.
    """
    if cardinality in {"multi_scalar", "grouped"}:
        return "metric_table"
    if cardinality == "collection" or kind == "table":
        return "record_table"
    if cardinality == "sectioned":
        return "sectioned"
    if kind in {"scalar", "status"}:
        return "key_value" if kind == "status" else "scalar"
    return "unknown"


def _cardinality_for_kind(kind: GoalOutputKind) -> OutputCardinality:
    """Choose a default cardinality from output kind.

    Inputs:
        Receives output kind.

    Returns:
        Cardinality fallback.

    Used by:
        resolve_output_intent.
    """
    if kind == "scalar":
        return "single"
    if kind == "table":
        return "collection"
    if kind in {"file", "text"}:
        return "sectioned" if kind == "text" else "collection"
    if kind == "status":
        return "single"
    return "unknown"


def _deterministic_output_backup(goal: str) -> ResolvedOutputIntent:
    """Build minimal deterministic output-shape facts for correction only.

    Inputs:
        Receives the user goal.

    Returns:
        ResolvedOutputIntent with only backup shape facts.

    Used by:
        resolve_output_intent after LLM semantic output is considered.
    """
    text = str(goal or "").lower()
    if _looks_like_sql_explain_only_goal(text):
        return ResolvedOutputIntent(
            kind="text",
            cardinality="single",
            render_style="bullets",
            source="deterministic_backup",
            reason="sql_explain_only",
        )
    if _looks_like_multi_scalar_count(text):
        return ResolvedOutputIntent(
            kind="table",
            cardinality="multi_scalar",
            render_style="metric_table",
            result_entities=tuple(_extract_count_entities(text)),
            source="deterministic_backup",
            reason="multi_scalar_count_backup",
        )
    group = grouped_count_field_for_goal(text)
    if group:
        return ResolvedOutputIntent(
            kind="table",
            cardinality="grouped",
            render_style="metric_table",
            group_by=(group,),
            source="deterministic_backup",
            reason="grouped_count_backup",
        )
    if _looks_like_scalar_count(text):
        return ResolvedOutputIntent(
            kind="scalar",
            cardinality="single",
            render_style="scalar",
            source="deterministic_backup",
            reason="scalar_count_backup",
        )
    if re.search(r"\b(?:list|show|display|return)\b", text):
        return ResolvedOutputIntent(
            kind="table",
            cardinality="collection",
            render_style="record_table",
            source="deterministic_backup",
            reason="collection_backup",
        )
    return ResolvedOutputIntent()


def _looks_like_scalar_count(goal: str) -> bool:
    """Detect simple scalar count as a deterministic backup.

    Inputs:
        Receives the user goal.

    Returns:
        True only for count prompts without grouped or multi-entity shape.

    Used by:
        _deterministic_output_backup and compatibility wrappers.
    """
    text = str(goal or "").lower()
    if not COUNT_GOAL_RE.search(text):
        return False
    if _looks_like_multi_scalar_count(text) or grouped_count_field_for_goal(text):
        return False
    return True


def _looks_like_multi_scalar_count(goal: str) -> bool:
    """Detect multiple independent count outputs as a deterministic backup.

    Inputs:
        Receives the user goal.

    Returns:
        True when wording asks for several count metrics rather than one scalar.

    Used by:
        _deterministic_output_backup and compatibility wrappers.
    """
    text = str(goal or "").lower()
    if not COUNT_GOAL_RE.search(text) or not MULTI_COUNT_SEPARATOR_RE.search(text):
        return False
    if re.search(r"\b(?:by|group(?:ed)?\s+by|per|each|separately|that\s+have|with|having|where)\b", text):
        return False
    return len(_extract_count_entities(text)) > 1


def _extract_count_entities(goal: str) -> list[str]:
    """Extract common independent count entity names.

    Inputs:
        Receives the user goal.

    Returns:
        Ordered normalized entity labels.

    Used by:
        _looks_like_multi_scalar_count and resolve_output_intent metadata.
    """
    patterns = [
        ("patients", r"\b(?:patients?|patieints?)\b"),
        ("studies", r"\bstudies\b"),
        ("series", r"\bseries\b"),
        ("instances", r"\binstances?\b"),
        ("rtplan", r"\brtplans?\b"),
        ("rtdose", r"\brtdoses?\b"),
        ("rtstruct", r"\brtstructs?\b"),
        ("jobs", r"\bjobs?\b"),
        ("files", r"\bfiles?\b"),
    ]
    return [label for label, pattern in patterns if re.search(pattern, goal, re.IGNORECASE)]


def _looks_like_sql_explain_only_goal(goal: str) -> bool:
    """Detect SQL generation/validation requests that should not execute the query.

    Inputs:
        Receives the user goal.

    Returns:
        True when the user asks to generate, validate, or explain SQL rather than run it.

    Used by:
        resolve_output_intent and deterministic backup rules.
    """
    text = str(goal or "").lower()
    if not re.search(r"\b(?:generate|write|draft|produce)\b", text):
        return False
    if not re.search(r"\bsql\b|\bquery\b", text):
        return False
    return bool(re.search(r"\b(?:validate|explain|what\s+it\s+would\s+return|would\s+return)\b", text))


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
    """Compatibility wrapper for scalar count output intent.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.output_shape.is_scalar_count_goal.
    """
    intent = resolve_output_intent(goal)
    return intent.kind == "scalar" and intent.cardinality == "single"


def is_multi_scalar_count_goal(goal: str) -> bool:
    """Compatibility wrapper for multi-scalar count output intent.

    Inputs:
        Receives a user goal.

    Returns:
        True when the goal asks for multiple count values that should be displayed as a table/dashboard.

    Used by:
        infer_goal_output_contract and is_scalar_count_goal.
    """
    intent = resolve_output_intent(goal)
    return intent.cardinality == "multi_scalar"


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
