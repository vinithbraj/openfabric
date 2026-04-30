"""OpenFABRIC Runtime Module: aor_runtime.runtime.sql_llm

Purpose:
    Prepare schema-aware SQL-generation prompts and parse LLM SQL responses.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from aor_runtime.llm.client import LLMClient
from aor_runtime.runtime.sql_catalog import SqlSchemaCatalog, SqlTableRef
from aor_runtime.runtime.sql_constraints import SqlConstraintFrame, validate_sql_constraint_coverage
from aor_runtime.runtime.sql_resolver import ResolvedSqlContext
from aor_runtime.runtime.sql_safety import (
    normalize_pg_relation_quoting,
    quote_pg_identifier,
    quote_pg_relation,
    validate_read_only_sql,
)


SQL_LLM_CONFIDENCE_THRESHOLD = 0.70
TOOLLIKE_OUTPUT_RE = re.compile(r"\b(?:sql\.query|python\.exec|shell\.exec|ExecutionPlan|tool_calls?)\b", re.IGNORECASE)


@dataclass(frozen=True)
class SqlLlmGeneration:
    """Represent sql llm generation within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SqlLlmGeneration.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.sql_llm.SqlLlmGeneration and related tests.
    """
    matched: bool
    sql: str | None = None
    normalized_sql: str | None = None
    confidence: float = 0.0
    reason: str = ""
    tables_used: list[str] = field(default_factory=list)
    columns_used: list[str] = field(default_factory=list)
    constraints_addressed: list[str] = field(default_factory=list)
    validation_failure: str | None = None
    raw_response: dict[str, Any] | None = None


def generate_sql_for_goal(
    *,
    llm: LLMClient,
    goal: str,
    catalog: SqlSchemaCatalog,
    resolved_context: ResolvedSqlContext | None = None,
    constraint_frame: SqlConstraintFrame | None = None,
    repair_context: dict[str, Any] | None = None,
    confidence_threshold: float = SQL_LLM_CONFIDENCE_THRESHOLD,
) -> SqlLlmGeneration:
    """Generate sql for goal for the surrounding runtime workflow.

    Inputs:
        Receives llm, goal, catalog, resolved_context, constraint_frame, repair_context, confidence_threshold for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_llm.generate_sql_for_goal.
    """
    schema_context = build_schema_context(catalog, resolved_context=resolved_context)
    system_prompt = _system_prompt()
    user_prompt = _user_prompt(goal=goal, schema_context=schema_context, constraint_frame=constraint_frame, repair_context=repair_context)
    try:
        payload = llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0)
    except Exception as exc:  # noqa: BLE001
        return SqlLlmGeneration(matched=False, reason=f"Malformed SQL generation JSON: {exc}")
    return coerce_sql_generation(payload, catalog=catalog, constraint_frame=constraint_frame, confidence_threshold=confidence_threshold)


def repair_sql_for_error(
    *,
    llm: LLMClient,
    goal: str,
    catalog: SqlSchemaCatalog,
    original_sql: str,
    error_message: str,
    resolved_context: ResolvedSqlContext | None = None,
    constraint_frame: SqlConstraintFrame | None = None,
    confidence_threshold: float = SQL_LLM_CONFIDENCE_THRESHOLD,
) -> SqlLlmGeneration:
    """Repair sql for error for the surrounding runtime workflow.

    Inputs:
        Receives llm, goal, catalog, original_sql, error_message, resolved_context, constraint_frame, confidence_threshold for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_llm.repair_sql_for_error.
    """
    return generate_sql_for_goal(
        llm=llm,
        goal=goal,
        catalog=catalog,
        resolved_context=resolved_context,
        constraint_frame=constraint_frame,
        repair_context={"original_sql": original_sql, "error": _sanitize_error(error_message)},
        confidence_threshold=confidence_threshold,
    )


def coerce_sql_generation(
    payload: dict[str, Any],
    *,
    catalog: SqlSchemaCatalog,
    constraint_frame: SqlConstraintFrame | None = None,
    confidence_threshold: float = SQL_LLM_CONFIDENCE_THRESHOLD,
) -> SqlLlmGeneration:
    """Coerce sql generation for the surrounding runtime workflow.

    Inputs:
        Receives payload, catalog, constraint_frame, confidence_threshold for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_llm.coerce_sql_generation.
    """
    if not isinstance(payload, dict):
        return SqlLlmGeneration(matched=False, reason="SQL generation response was not an object.")
    sql = str(payload.get("sql") or payload.get("query") or "").strip()
    if not bool(payload.get("matched")) and not sql:
        return SqlLlmGeneration(matched=False, reason=str(payload.get("reason") or "SQL generator did not match."), raw_response=payload)
    confidence = _coerce_float(payload.get("confidence"), default=0.75 if sql else 0.0)
    if confidence < confidence_threshold:
        return SqlLlmGeneration(
            matched=False,
            confidence=confidence,
            reason=f"SQL generation confidence {confidence:.2f} is below {confidence_threshold:.2f}.",
            raw_response=payload,
        )
    if not sql:
        return SqlLlmGeneration(matched=False, confidence=confidence, reason="SQL generation returned an empty query.", raw_response=payload)
    if TOOLLIKE_OUTPUT_RE.search(sql):
        return SqlLlmGeneration(
            matched=False,
            confidence=confidence,
            reason="SQL generation returned tool-call-like output instead of SQL.",
            raw_response=payload,
        )

    normalized_sql = normalize_pg_relation_quoting(sql, catalog)
    validation = validate_read_only_sql(normalized_sql)
    if not validation.valid:
        return SqlLlmGeneration(
            matched=False,
            sql=sql,
            normalized_sql=normalized_sql,
            confidence=confidence,
            reason="Generated SQL failed read-only validation.",
            validation_failure=validation.reason,
            raw_response=payload,
        )

    tables_used = _coerce_string_list(payload.get("tables_used"))
    columns_used = _coerce_string_list(payload.get("columns_used"))
    constraints_addressed = _coerce_string_list(payload.get("constraints_addressed"))
    reference_failure = validate_declared_references(tables_used=tables_used, columns_used=columns_used, catalog=catalog)
    if reference_failure is not None:
        return SqlLlmGeneration(
            matched=False,
            sql=sql,
            normalized_sql=normalized_sql,
            confidence=confidence,
            reason=reference_failure,
            validation_failure=reference_failure,
            raw_response=payload,
        )
    if constraint_frame is not None:
        coverage = validate_sql_constraint_coverage(constraint_frame, str(validation.normalized_sql or normalized_sql))
        if not coverage.valid:
            return SqlLlmGeneration(
                matched=False,
                sql=sql,
                normalized_sql=str(validation.normalized_sql or normalized_sql),
                confidence=confidence,
                reason=coverage.reason or "Generated SQL failed constraint coverage validation.",
                tables_used=tables_used,
                columns_used=columns_used,
                constraints_addressed=constraints_addressed,
                validation_failure=f"SQL constraint coverage failed: {coverage.reason or ', '.join(coverage.missing_constraint_ids)}",
                raw_response=payload,
            )

    return SqlLlmGeneration(
        matched=True,
        sql=sql,
        normalized_sql=str(validation.normalized_sql or normalized_sql),
        confidence=confidence,
        reason=str(payload.get("reason") or "").strip(),
        tables_used=tables_used,
        columns_used=columns_used,
        constraints_addressed=constraints_addressed,
        raw_response=payload,
    )


def validate_declared_references(*, tables_used: list[str], columns_used: list[str], catalog: SqlSchemaCatalog) -> str | None:
    """Validate declared references for the surrounding runtime workflow.

    Inputs:
        Receives tables_used, columns_used, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_llm.validate_declared_references.
    """
    for item in tables_used:
        schema, table = _parse_declared_table(item)
        if not schema or not table or catalog.table_by_name(schema, table) is None:
            return f"Generated SQL referenced unknown table: {item}."
    for item in columns_used:
        schema, table, column = _parse_declared_column(item)
        table_ref = catalog.table_by_name(schema, table) if schema and table else None
        if table_ref is None or column is None or table_ref.column_by_name(column) is None:
            return f"Generated SQL referenced unknown column: {item}."
    return None


def build_schema_context(
    catalog: SqlSchemaCatalog,
    *,
    resolved_context: ResolvedSqlContext | None = None,
    max_tables: int = 12,
    max_columns_per_table: int = 36,
) -> str:
    """Build schema context for the surrounding runtime workflow.

    Inputs:
        Receives catalog, resolved_context, max_tables, max_columns_per_table for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_llm.build_schema_context.
    """
    preferred: list[SqlTableRef] = (
        [*resolved_context.tables, *resolved_context.ambiguous_tables] if resolved_context is not None else []
    )
    if resolved_context is not None:
        mentioned = _tables_mentioned_in_goal(resolved_context.goal, catalog)
        preferred = [*preferred, *[table for table in mentioned if table.qualified_name not in {item.qualified_name for item in preferred}]]
    seen = {table.qualified_name for table in preferred}
    remaining = [table for table in catalog.tables if table.qualified_name not in seen]
    selected = [*preferred, *remaining[: max(0, max_tables - len(preferred))]]
    lines = [f"database={catalog.database} dialect={catalog.dialect or 'unknown'}"]
    for table in selected[:max_tables]:
        columns = []
        for column in table.columns[:max_columns_per_table]:
            nullable = " nullable" if column.nullable else ""
            pk = " pk" if column.primary_key else ""
            columns.append(f"{quote_pg_identifier(column.column_name)} {column.data_type or 'unknown'}{nullable}{pk}".strip())
        lines.append(f"- {quote_pg_relation(table.schema_name, table.table_name)}: {', '.join(columns)}")
    if len(catalog.tables) > len(selected):
        lines.append(f"- ... {len(catalog.tables) - len(selected)} additional tables omitted; use only listed tables unless required.")
    return "\n".join(lines)


def _system_prompt() -> str:
    """Handle the internal system prompt helper path for this module.

    Inputs:
        Uses module or instance state; no caller-supplied data parameters are required.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_llm._system_prompt.
    """
    return (
        "You are a PostgreSQL SQL generator inside a read-only SQL capability. "
        "Return JSON only. Generate exactly one PostgreSQL SELECT query, or WITH query ending in SELECT. "
        "Do not emit tool calls, execution plans, shell commands, Python, markdown, comments, or explanations outside JSON. "
        "Use only the provided schemas, tables, and columns. Quote mixed-case identifiers correctly, for example flathr.\"Patient\" and p.\"PatientID\". "
        "For count questions, alias the scalar as count_value. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, COPY, CALL, DO, EXECUTE, GRANT, or REVOKE."
    )


def _user_prompt(
    *,
    goal: str,
    schema_context: str,
    constraint_frame: SqlConstraintFrame | None,
    repair_context: dict[str, Any] | None,
) -> str:
    """Handle the internal user prompt helper path for this module.

    Inputs:
        Receives goal, schema_context, constraint_frame, repair_context for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_llm._user_prompt.
    """
    payload: dict[str, Any] = {
        "goal": goal,
        "schema_context": schema_context,
        "constraints": constraint_frame.to_dict() if constraint_frame is not None else None,
        "response_schema": {
            "matched": True,
            "sql": "single SELECT query",
            "confidence": 0.0,
            "reason": "brief reason",
            "tables_used": ["schema.table"],
            "columns_used": ["schema.table.column"],
            "constraints_addressed": ["constraint_id"],
        },
    }
    if repair_context:
        payload["repair"] = repair_context
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _coerce_float(value: Any, *, default: float = 0.0) -> float:
    """Handle the internal coerce float helper path for this module.

    Inputs:
        Receives value, default for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_llm._coerce_float.
    """
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def _tables_mentioned_in_goal(goal: str, catalog: SqlSchemaCatalog) -> list[SqlTableRef]:
    """Handle the internal tables mentioned in goal helper path for this module.

    Inputs:
        Receives goal, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_llm._tables_mentioned_in_goal.
    """
    tokens = set(re.findall(r"[a-z0-9]+", str(goal or "").lower()))
    expanded = set(tokens)
    for token in tokens:
        if token.endswith("ies") and len(token) > 4:
            expanded.add(token[:-3] + "y")
        elif token.endswith("s") and len(token) > 3:
            expanded.add(token[:-1])
    mentioned: list[SqlTableRef] = []
    scored: list[tuple[int, SqlTableRef]] = []
    for table in catalog.tables:
        table_tokens = set(re.findall(r"[a-z0-9]+", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", table.table_name).lower()))
        if expanded & table_tokens:
            scored.append((len(expanded & table_tokens) * 10 - len(table_tokens), table))
    scored.sort(key=lambda item: (-item[0], item[1].qualified_name.lower()))
    return [table for _, table in scored]


def _coerce_string_list(value: Any) -> list[str]:
    """Handle the internal coerce string list helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_llm._coerce_string_list.
    """
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _parse_declared_table(value: str) -> tuple[str | None, str | None]:
    """Handle the internal parse declared table helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_llm._parse_declared_table.
    """
    cleaned = _unquote_declared(value)
    if "." not in cleaned:
        return None, cleaned or None
    schema, table = cleaned.split(".", 1)
    return schema or None, table or None


def _parse_declared_column(value: str) -> tuple[str | None, str | None, str | None]:
    """Handle the internal parse declared column helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_llm._parse_declared_column.
    """
    cleaned = _unquote_declared(value)
    parts = cleaned.split(".")
    if len(parts) < 3:
        return None, None, parts[-1] if parts else None
    return parts[-3] or None, parts[-2] or None, parts[-1] or None


def _unquote_declared(value: str) -> str:
    """Handle the internal unquote declared helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_llm._unquote_declared.
    """
    text = str(value or "").strip()
    parts = []
    for part in text.split("."):
        stripped = part.strip()
        if len(stripped) >= 2 and stripped[0] == stripped[-1] == '"':
            stripped = stripped[1:-1].replace('""', '"')
        parts.append(stripped)
    return ".".join(parts)


def _sanitize_error(value: str) -> str:
    """Handle the internal sanitize error helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_llm._sanitize_error.
    """
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:700]
