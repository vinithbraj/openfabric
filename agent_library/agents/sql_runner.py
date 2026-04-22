import json
import os
import re
import sqlite3
import time
from contextlib import closing
from typing import Any
from urllib.parse import unquote, urlparse

import requests
from web_compat import FastAPI, JSONResponse

from agent_library.common import EventRequest, EventResponse, shared_llm_api_settings, task_plan_context, with_node_envelope
from agent_library.reduction import (
    build_sql_reduction_request,
    execute_reduction_request,
    generate_sql_reduction_command,
    should_reduce_sql_result,
    summarize_sql_rows,
)
from agent_library.template import agent_api, agent_descriptor, emit, failure_result, needs_decomposition, noop
from runtime.console import log_debug, log_raw

app = FastAPI()

SQL_EXECUTION_MODEL = "llm_selected_local_execution"
SQL_DETERMINISTIC_CATALOG_VERSION = "v4-initial"
SQL_DETERMINISTIC_PRIMITIVES = [
    {
        "primitive_id": "sql.schema.list_schemas",
        "family": "schema",
        "summary": "List database schemas.",
        "required_params": [],
        "optional_params": [],
        "intent_tags": ["schema", "schemas", "database"],
        "read_only": True,
    },
    {
        "primitive_id": "sql.schema.list_tables",
        "family": "schema",
        "summary": "List tables across the database or within one schema.",
        "required_params": [],
        "optional_params": ["schema_name"],
        "intent_tags": ["schema", "tables", "database"],
        "read_only": True,
    },
    {
        "primitive_id": "sql.schema.count_tables",
        "family": "schema",
        "summary": "Count tables across the database or within one schema.",
        "required_params": [],
        "optional_params": ["schema_name"],
        "intent_tags": ["schema", "tables", "count", "database"],
        "read_only": True,
    },
    {
        "primitive_id": "sql.schema.list_columns",
        "family": "schema",
        "summary": "List columns globally or for one table.",
        "required_params": [],
        "optional_params": ["schema_name", "table_name"],
        "intent_tags": ["schema", "columns", "table"],
        "read_only": True,
    },
    {
        "primitive_id": "sql.schema.list_relationships",
        "family": "schema",
        "summary": "List foreign-key relationships globally or for one table.",
        "required_params": [],
        "optional_params": ["schema_name", "table_name"],
        "intent_tags": ["schema", "relationships", "foreign_keys"],
        "read_only": True,
    },
    {
        "primitive_id": "sql.table.exists",
        "family": "table",
        "summary": "Check whether a table exists.",
        "required_params": ["table_name"],
        "optional_params": ["schema_name"],
        "intent_tags": ["table", "exists", "schema"],
        "read_only": True,
    },
    {
        "primitive_id": "sql.table.row_count",
        "family": "table",
        "summary": "Count rows in one table.",
        "required_params": ["table_name"],
        "optional_params": ["schema_name", "filters"],
        "intent_tags": ["count", "rows", "table"],
        "read_only": True,
    },
    {
        "primitive_id": "sql.table.sample_rows",
        "family": "table",
        "summary": "Return sample rows from one table.",
        "required_params": ["table_name"],
        "optional_params": ["schema_name", "columns", "filters", "limit", "order_by", "order_direction"],
        "intent_tags": ["sample", "rows", "table"],
        "read_only": True,
    },
    {
        "primitive_id": "sql.rows.select",
        "family": "rows",
        "summary": "Return selected rows from one table with validated filters.",
        "required_params": ["table_name"],
        "optional_params": ["schema_name", "columns", "filters", "limit", "order_by", "order_direction"],
        "intent_tags": ["rows", "select", "filter"],
        "read_only": True,
    },
    {
        "primitive_id": "sql.agg.count",
        "family": "aggregate",
        "summary": "Count rows from one table with validated filters.",
        "required_params": ["table_name"],
        "optional_params": ["schema_name", "filters"],
        "intent_tags": ["count", "aggregate", "filter"],
        "read_only": True,
    },
    {
        "primitive_id": "sql.table.distinct_values",
        "family": "table",
        "summary": "List distinct values for one validated column.",
        "required_params": ["table_name", "column_name"],
        "optional_params": ["schema_name", "filters", "limit"],
        "intent_tags": ["distinct", "values", "column"],
        "read_only": True,
    },
    {
        "primitive_id": "sql.group.count_by_column",
        "family": "group",
        "summary": "Return grouped counts for one validated column.",
        "required_params": ["table_name", "column_name"],
        "optional_params": ["schema_name", "filters", "limit", "order_direction"],
        "intent_tags": ["group", "count", "column"],
        "read_only": True,
    },
]
SQL_DETERMINISTIC_PRIMITIVE_IDS = {
    item["primitive_id"] for item in SQL_DETERMINISTIC_PRIMITIVES if isinstance(item, dict) and item.get("primitive_id")
}
SQL_DETERMINISTIC_CATALOG_FAMILIES = sorted(
    {
        str(item.get("family")).strip()
        for item in SQL_DETERMINISTIC_PRIMITIVES
        if isinstance(item, dict) and str(item.get("family") or "").strip()
    }
)

AGENT_DESCRIPTOR = agent_descriptor(
    name="sql_runner",
    role="executor",
    description=(
        "Connects to a configured SQL database, introspects schemas/tables/columns/"
        "relationships, asks an LLM to select the best local execution strategy, "
        "executes local read-only database primitives when selected, and otherwise runs generated read-only SQL."
    ),
    capability_domains=[
        "sql",
        "database",
        "schema_introspection",
        "deterministic_primitives",
        "query_generation",
        "read_only_analytics",
        "data_retrieval",
    ],
    action_verbs=[
        "connect",
        "introspect",
        "describe",
        "query",
        "select",
        "join",
        "aggregate",
        "count",
        "filter",
        "group",
        "sort",
        "explain",
    ],
    execution_model=SQL_EXECUTION_MODEL,
    deterministic_catalog_version=SQL_DETERMINISTIC_CATALOG_VERSION,
    deterministic_catalog_families=SQL_DETERMINISTIC_CATALOG_FAMILIES,
    deterministic_catalog_size=len(SQL_DETERMINISTIC_PRIMITIVES),
    deterministic_primitives=[item["primitive_id"] for item in SQL_DETERMINISTIC_PRIMITIVES],
    deterministic_catalog_reference="docs/VERSION_4_PRIMITIVE_CATALOG.md",
    fallback_policy=(
        "Use the LLM-selected SQL strategy. When it selects a local primitive, execute it locally. "
        "Otherwise execute the LLM-selected fallback SQL or generated SQL."
    ),
    side_effect_policy="read_only_sql_with_safety_checks",
    safety_enforced_by_agent=True,
    routing_notes=[
        "Use for requests about SQL databases, schemas, tables, columns, relationships, or data questions that should be answered by querying a configured database.",
        "Requires SQL_AGENT_DSN or SQL_DATABASE_URL to be configured in the agent environment.",
        "Only read-only SQL is executed. Mutating statements are rejected.",
        "For natural-language database questions, this agent now uses deterministic primitives first and only falls back to free-form SQL generation when needed.",
    ],
    planning_hints={
        "keywords": [
            "sql",
            "database",
            "db",
            "schema",
            "schemas",
            "table",
            "tables",
            "column",
            "columns",
            "rows",
            "query",
            "queries",
            "join",
            "relationship",
            "relationships",
            "foreign key",
            "record",
            "records",
        ],
        "anti_keywords": ["repo file search", "docker", "terminal logs", "slurm queue", "cluster nodes"],
        "preferred_task_shapes": ["count", "boolean_check", "schema_summary", "list", "compare", "lookup", "save_artifact"],
        "instruction_operations": ["inspect_schema", "query_from_request", "execute_sql", "sample_rows"],
        "structured_followup": True,
        "native_count_preferred": True,
        "routing_priority": 45,
    },
    apis=[
        agent_api(
            name="introspect_database",
            trigger_event="task.plan",
            emits=["sql.result", "task.result"],
            summary="Lists schemas, tables, columns, and relationships for the configured SQL database.",
            when="Lists schemas, tables, columns, and relationships for the configured SQL database.",
            intent_tags=["sql_schema", "database_introspection"],
            examples=["show database schema", "list all SQL tables", "describe relationships in the database"],
            deterministic=True,
            side_effect_level="read_only",
            planning_hints={
                "keywords": ["schema", "schemas", "tables", "columns", "relationships", "foreign keys"],
                "preferred_task_shapes": ["schema_summary", "list", "lookup"],
                "instruction_operations": ["inspect_schema"],
            },
        ),
        agent_api(
            name="select_deterministic_sql_primitive",
            trigger_event="task.plan",
            emits=["sql.result", "task.result"],
            summary="Selects one deterministic SQL primitive plus a fallback SQL plan for a natural-language database question.",
            when="Selects one deterministic SQL primitive plus a fallback SQL plan for a natural-language database question.",
            intent_tags=["sql_primitive_selection", "database_question", "analytics"],
            examples=["how many rows are in patients", "list all tables in mydb", "show distinct patient genders"],
            deterministic=True,
            side_effect_level="read_only",
            planning_hints={
                "keywords": ["count", "rows", "database question", "analytics", "distinct", "group by"],
                "preferred_task_shapes": ["count", "list", "lookup", "compare"],
                "instruction_operations": ["query_from_request", "sample_rows"],
                "native_count_preferred": True,
            },
        ),
        agent_api(
            name="execute_sql_fallback_when_needed",
            trigger_event="task.plan",
            emits=["sql.result", "task.result"],
            summary="Executes generated fallback SQL only when the deterministic SQL primitive cannot answer cleanly.",
            when="Executes generated fallback SQL only when the deterministic SQL primitive cannot answer cleanly.",
            intent_tags=["sql_query_fallback", "database_question", "analytics"],
            examples=["which customers spent the most money last month", "show top 10 customers by revenue"],
            deterministic=False,
            side_effect_level="read_only",
            planning_hints={
                "keywords": ["query", "join", "aggregate", "top customers", "revenue", "database question"],
                "preferred_task_shapes": ["list", "compare", "lookup"],
                "instruction_operations": ["execute_sql", "query_from_request"],
            },
        ),
    ],
)
AGENT_METADATA = AGENT_DESCRIPTOR

READ_ONLY_PREFIXES = ("select", "with", "show", "describe", "explain")
MUTATING_PATTERN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|merge|replace|grant|revoke|vacuum|attach|detach)\b",
    re.IGNORECASE,
)
IDENTIFIER_ERROR_PATTERN = re.compile(
    r"(undefinedcolumn|undefinedtable|no such column|no such table|unknown column|unknown table|column .* does not exist|relation .* does not exist|syntax error)",
    re.IGNORECASE,
)


def _debug_enabled() -> bool:
    return os.getenv("SQL_AGENT_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _debug_log(message: str):
    if _debug_enabled():
        log_debug("SQL_AGENT_DEBUG", message)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _llm_api_settings():
    return shared_llm_api_settings("gpt-4o-mini")


def _sql_llm_transport_settings(default_model: str) -> tuple[str, str, float, str]:
    return shared_llm_api_settings(default_model)


def _sql_catalog_prompt_text() -> str:
    lines = []
    for primitive in SQL_DETERMINISTIC_PRIMITIVES:
        primitive_id = str(primitive.get("primitive_id") or "").strip()
        if not primitive_id:
            continue
        summary = str(primitive.get("summary") or "").strip()
        required_params = primitive.get("required_params") if isinstance(primitive.get("required_params"), list) else []
        optional_params = primitive.get("optional_params") if isinstance(primitive.get("optional_params"), list) else []
        lines.append(
            f"- {primitive_id}: {summary} "
            f"required=[{', '.join(str(item) for item in required_params) or 'none'}] "
            f"optional=[{', '.join(str(item) for item in optional_params) or 'none'}]"
        )
    return "\n".join(lines)


def _normalize_positive_int(value: Any, default: int, minimum: int = 1, maximum: int = 1000) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


def _resolve_table_spec(schema: dict, table_name: str, schema_name: str = "") -> dict[str, Any] | None:
    raw_table = str(table_name or "").strip()
    raw_schema = str(schema_name or "").strip()
    if not raw_table:
        return None
    if "." in raw_table and not raw_schema:
        prefix, suffix = raw_table.split(".", 1)
        raw_schema = prefix.strip()
        raw_table = suffix.strip()
    if not raw_table:
        return None
    exact_matches = []
    loose_matches = []
    for table in schema.get("tables", []):
        if not isinstance(table, dict):
            continue
        candidate_table = str(table.get("name") or "").strip()
        candidate_schema = str(table.get("schema") or "").strip()
        if not candidate_table:
            continue
        if candidate_table == raw_table and (not raw_schema or candidate_schema == raw_schema):
            exact_matches.append(table)
        if candidate_table.lower() == raw_table.lower() and (not raw_schema or candidate_schema.lower() == raw_schema.lower()):
            loose_matches.append(table)
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(loose_matches) == 1:
        return loose_matches[0]
    return None


def _resolve_column_name(table: dict[str, Any], column_name: str) -> str | None:
    raw = str(column_name or "").strip()
    if not raw:
        return None
    exact_matches = []
    loose_matches = []
    for column in table.get("columns", []):
        if not isinstance(column, dict):
            continue
        candidate = str(column.get("name") or "").strip()
        if not candidate:
            continue
        if candidate == raw:
            exact_matches.append(candidate)
        if candidate.lower() == raw.lower():
            loose_matches.append(candidate)
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(loose_matches) == 1:
        return loose_matches[0]
    return None


def _sql_literal(value: Any, dialect: str) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        if dialect == "mysql":
            return "1" if value else "0"
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def _normalize_sql_filters(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if not isinstance(item, dict):
            continue
        column = str(item.get("column") or "").strip()
        op = str(item.get("op") or "").strip().lower()
        if not column or not op:
            continue
        normalized.append(
            {
                "column": column,
                "op": op,
                "value": item.get("value"),
                "values": item.get("values"),
                "lower": item.get("lower"),
                "upper": item.get("upper"),
            }
        )
    return normalized


def _build_sql_where_clauses(filters: list[dict[str, Any]], table: dict[str, Any], dialect: str) -> list[str] | None:
    clauses: list[str] = []
    for item in filters:
        column_name = _resolve_column_name(table, item.get("column", ""))
        if not column_name:
            return None
        quoted_column = _quote_identifier(column_name, dialect)
        op = str(item.get("op") or "").strip().lower()
        if op == "eq":
            clauses.append(f"{quoted_column} = {_sql_literal(item.get('value'), dialect)}")
        elif op == "neq":
            clauses.append(f"{quoted_column} <> {_sql_literal(item.get('value'), dialect)}")
        elif op == "gt":
            clauses.append(f"{quoted_column} > {_sql_literal(item.get('value'), dialect)}")
        elif op == "gte":
            clauses.append(f"{quoted_column} >= {_sql_literal(item.get('value'), dialect)}")
        elif op == "lt":
            clauses.append(f"{quoted_column} < {_sql_literal(item.get('value'), dialect)}")
        elif op == "lte":
            clauses.append(f"{quoted_column} <= {_sql_literal(item.get('value'), dialect)}")
        elif op == "like":
            clauses.append(f"{quoted_column} LIKE {_sql_literal(item.get('value'), dialect)}")
        elif op == "ilike":
            if dialect == "postgres":
                clauses.append(f"{quoted_column} ILIKE {_sql_literal(item.get('value'), dialect)}")
            else:
                clauses.append(
                    f"LOWER(CAST({quoted_column} AS TEXT)) LIKE LOWER({_sql_literal(item.get('value'), dialect)})"
                )
        elif op == "is_null":
            clauses.append(f"{quoted_column} IS NULL")
        elif op == "is_not_null":
            clauses.append(f"{quoted_column} IS NOT NULL")
        elif op in {"in", "not_in"}:
            values = item.get("values")
            if not isinstance(values, list) or not values:
                return None
            value_sql = ", ".join(_sql_literal(entry, dialect) for entry in values)
            comparator = "IN" if op == "in" else "NOT IN"
            clauses.append(f"{quoted_column} {comparator} ({value_sql})")
        elif op == "between":
            clauses.append(
                f"{quoted_column} BETWEEN {_sql_literal(item.get('lower'), dialect)} "
                f"AND {_sql_literal(item.get('upper'), dialect)}"
            )
        else:
            return None
    return clauses


def _qualified_table_name(table: dict[str, Any], dialect: str) -> str:
    schema_name = str(table.get("schema") or "").strip()
    table_name = str(table.get("name") or "").strip()
    qualified = ".".join(part for part in (schema_name, table_name) if part)
    return _quote_qualified_identifier(qualified, dialect)


def _quoted_columns_for_selection(table: dict[str, Any], dialect: str, columns: Any) -> list[str] | None:
    if not isinstance(columns, list) or not columns:
        return None
    resolved = []
    for item in columns:
        column_name = _resolve_column_name(table, item)
        if not column_name:
            return None
        resolved.append(_quote_identifier(column_name, dialect))
    return resolved


def _selection_should_use_fallback(task: str, selection: dict[str, Any]) -> bool:
    fallback_sql = str(selection.get("fallback_sql") or "").strip()
    if not fallback_sql:
        return False
    primitive_id = str(selection.get("primitive_id") or "").strip()
    if primitive_id == "fallback_only":
        return True
    task_lc = str(task or "").lower()
    unsupported_aggregate_tokens = (
        "average",
        "avg",
        "mean",
        "median",
        "percentile",
        "percentage",
        "percent ",
        "ratio",
    )
    if primitive_id in {"sql.agg.count", "sql.table.row_count"} and any(token in task_lc for token in unsupported_aggregate_tokens):
        return True
    return False


def _llm_select_sql_strategy(task: str, schema: dict) -> dict[str, Any]:
    api_key, base_url, timeout_seconds, model = _sql_llm_transport_settings("gpt-4o-mini")
    prompt = (
        "You are the deterministic SQL primitive selector for a database agent.\n"
        "Choose exactly one deterministic primitive from the executable primitive catalog below.\n"
        "Also provide a fallback SQL query that would answer the same user request if deterministic execution cannot answer cleanly.\n"
        "Return STRICT JSON only with exactly these top-level keys:\n"
        "{\"primitive_id\":\"...\",\"selection_reason\":\"...\",\"parameters\":{...},\"fallback_sql\":\"...\",\"fallback_reason\":\"...\"}\n"
        "Rules:\n"
        "- primitive_id MUST be one of the executable primitive IDs listed below, or fallback_only.\n"
        "- Use exact schema, table, and column identifiers from the schema catalog.\n"
        "- Prefer the simplest primitive that can fully answer the request.\n"
        "- If the request is fundamentally outside the executable deterministic primitives, use primitive_id=fallback_only.\n"
        "- fallback_sql must be read-only SQL or an empty string when no fallback can be proposed.\n"
        "- parameters must only include fields that help execute the chosen primitive, such as schema_name, table_name, column_name, columns, filters, limit, order_by, order_direction.\n"
        "Executable deterministic primitives:\n"
        f"{_sql_catalog_prompt_text()}\n"
        "Schema summary:\n"
        f"{_schema_summary(schema)}\n"
        "Exact identifier catalog:\n"
        f"{_schema_identifier_catalog(schema)}\n"
        f"User request: {task}"
    )
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You produce strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    log_raw("SQL_SELECTOR_LLM_RAW", content)
    parsed = _parse_strict_json_object(content)
    return parsed if isinstance(parsed, dict) else {}


def _normalize_sql_selection(selection: Any) -> dict[str, Any]:
    if not isinstance(selection, dict):
        return {"primitive_id": "fallback_only", "selection_reason": "", "parameters": {}, "fallback_sql": "", "fallback_reason": ""}
    primitive_id = str(selection.get("primitive_id") or "").strip()
    if primitive_id not in SQL_DETERMINISTIC_PRIMITIVE_IDS:
        primitive_id = "fallback_only"
    parameters = selection.get("parameters")
    fallback_sql = str(selection.get("fallback_sql") or "").strip()
    fallback_reason = str(selection.get("fallback_reason") or "").strip()
    return {
        "primitive_id": primitive_id,
        "selection_reason": str(selection.get("selection_reason") or "").strip(),
        "parameters": parameters if isinstance(parameters, dict) else {},
        "fallback_sql": fallback_sql,
        "fallback_reason": fallback_reason,
    }


def _sql_fallback_query_specs_from_selection(selection: dict[str, Any]) -> list[dict[str, str]]:
    fallback_sql = str(selection.get("fallback_sql") or "").strip()
    if not fallback_sql:
        return []
    label = str(selection.get("fallback_reason") or "selector fallback").strip() or "selector fallback"
    return [{"label": label, "sql": fallback_sql}]


def _deterministic_sql_exists_result(table: dict[str, Any]) -> dict[str, Any]:
    return {
        "columns": ["schema", "table", "exists"],
        "rows": [
            {
                "schema": table.get("schema"),
                "table": table.get("name"),
                "exists": True,
            }
        ],
        "row_count": 1,
        "returned_row_count": 1,
        "total_matching_rows": 1,
        "truncated": False,
        "limit": 1,
    }


def _execute_sql_deterministic_selection(
    conn,
    dialect: str,
    schema: dict,
    task: str,
    selection: dict[str, Any],
    limit: int,
) -> dict[str, Any] | None:
    primitive_id = str(selection.get("primitive_id") or "").strip()
    params = selection.get("parameters") if isinstance(selection.get("parameters"), dict) else {}
    if primitive_id == "sql.schema.list_schemas":
        return {"detail": "Database schemas listed.", "sql": "", "result": _schema_schemas_result(schema)}
    if primitive_id == "sql.schema.count_tables":
        schema_name = str(params.get("schema_name") or "").strip()
        rows = _schema_tables_result(schema)["rows"]
        if schema_name:
            rows = [
                row for row in rows
                if str(row.get("schema") or "").strip().lower() == schema_name.lower()
            ]
        count = len(rows)
        result = {
            "columns": ["count"],
            "rows": [{"count": count}],
            "row_count": 1,
            "returned_row_count": 1,
            "total_matching_rows": 1,
            "truncated": False,
            "limit": 1,
        }
        if schema_name:
            return {"detail": f"Counted tables for schema {schema_name}.", "sql": "", "result": result}
        return {"detail": "Counted tables across the database.", "sql": "", "result": result}
    if primitive_id == "sql.schema.list_tables":
        schema_name = str(params.get("schema_name") or "").strip()
        if not schema_name:
            return {"detail": "Database tables listed.", "sql": "", "result": _schema_tables_result(schema)}
        rows = [
            row for row in _schema_tables_result(schema)["rows"]
            if str(row.get("schema") or "").strip().lower() == schema_name.lower()
        ]
        return {
            "detail": f"Database tables listed for schema {schema_name}.",
            "sql": "",
            "result": {
                "columns": ["schema", "table", "type"],
                "rows": rows,
                "row_count": len(rows),
                "returned_row_count": len(rows),
                "total_matching_rows": len(rows),
                "truncated": False,
                "limit": len(rows),
            },
        }
    if primitive_id == "sql.schema.list_columns":
        schema_name = str(params.get("schema_name") or "").strip()
        table_name = str(params.get("table_name") or "").strip()
        if table_name:
            table = _resolve_table_spec(schema, table_name, schema_name)
            if not table:
                return None
            rows = []
            for column in table.get("columns", []):
                rows.append(
                    {
                        "schema": table.get("schema"),
                        "table": table.get("name"),
                        "column": column.get("name"),
                        "type": column.get("type"),
                        "nullable": column.get("nullable"),
                    }
                )
            return {
                "detail": f"Database columns listed for table {table.get('name')}.",
                "sql": "",
                "result": {
                    "columns": ["schema", "table", "column", "type", "nullable"],
                    "rows": rows,
                    "row_count": len(rows),
                    "returned_row_count": len(rows),
                    "total_matching_rows": len(rows),
                    "truncated": False,
                    "limit": len(rows),
                },
            }
        if not schema_name:
            return {"detail": "Database columns listed.", "sql": "", "result": _schema_columns_result(schema)}
        rows = [
            row for row in _schema_columns_result(schema)["rows"]
            if str(row.get("schema") or "").strip().lower() == schema_name.lower()
        ]
        return {
            "detail": f"Database columns listed for schema {schema_name}.",
            "sql": "",
            "result": {
                "columns": ["schema", "table", "column", "type", "nullable"],
                "rows": rows,
                "row_count": len(rows),
                "returned_row_count": len(rows),
                "total_matching_rows": len(rows),
                "truncated": False,
                "limit": len(rows),
            },
        }
    if primitive_id == "sql.schema.list_relationships":
        schema_name = str(params.get("schema_name") or "").strip()
        table_name = str(params.get("table_name") or "").strip()
        if table_name:
            table = _resolve_table_spec(schema, table_name, schema_name)
            if not table:
                return None
            rows = []
            for foreign_key in table.get("foreign_keys", []):
                rows.append(
                    {
                        "schema": table.get("schema"),
                        "table": table.get("name"),
                        "column": foreign_key.get("column"),
                        "references": ".".join(
                            part
                            for part in (
                                str(foreign_key.get("references_schema") or "").strip(),
                                str(foreign_key.get("references_table") or "").strip(),
                                str(foreign_key.get("references_column") or "").strip(),
                            )
                            if part
                        ),
                    }
                )
            return {
                "detail": f"Database relationships listed for table {table.get('name')}.",
                "sql": "",
                "result": {
                    "columns": ["schema", "table", "column", "references"],
                    "rows": rows,
                    "row_count": len(rows),
                    "returned_row_count": len(rows),
                    "total_matching_rows": len(rows),
                    "truncated": False,
                    "limit": len(rows),
                },
            }
        if not schema_name:
            return {"detail": "Database relationships listed.", "sql": "", "result": _schema_relationships_result(schema)}
        rows = [
            row for row in _schema_relationships_result(schema)["rows"]
            if str(row.get("schema") or "").strip().lower() == schema_name.lower()
        ]
        return {
            "detail": f"Database relationships listed for schema {schema_name}.",
            "sql": "",
            "result": {
                "columns": ["schema", "table", "column", "references"],
                "rows": rows,
                "row_count": len(rows),
                "returned_row_count": len(rows),
                "total_matching_rows": len(rows),
                "truncated": False,
                "limit": len(rows),
            },
        }

    table = _resolve_table_spec(schema, params.get("table_name"), params.get("schema_name", ""))
    if not table:
        return None
    table_sql = _qualified_table_name(table, dialect)
    filters = _normalize_sql_filters(params.get("filters"))
    where_clauses = _build_sql_where_clauses(filters, table, dialect)
    if where_clauses is None:
        return None
    where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    if primitive_id == "sql.table.exists":
        return {"detail": f"Table {table.get('name')} exists.", "sql": "", "result": _deterministic_sql_exists_result(table)}

    if primitive_id in {"sql.table.row_count", "sql.agg.count"}:
        sql = f"SELECT COUNT(*) AS {_quote_identifier('count', dialect)} FROM {table_sql}{where_sql}"
        result = _execute_sql(conn, sql, 1, schema)
        return {"detail": "SQL deterministic count executed.", "sql": result.get("sql", sql), "result": result}

    if primitive_id == "sql.table.sample_rows" or primitive_id == "sql.rows.select":
        columns = _quoted_columns_for_selection(table, dialect, params.get("columns"))
        select_columns = ", ".join(columns) if columns else "*"
        order_by = _resolve_column_name(table, params.get("order_by"))
        order_direction = str(params.get("order_direction") or "asc").strip().lower()
        if order_direction not in {"asc", "desc"}:
            order_direction = "asc"
        order_sql = f" ORDER BY {_quote_identifier(order_by, dialect)} {order_direction.upper()}" if order_by else ""
        row_limit = _normalize_positive_int(params.get("limit"), 5 if primitive_id == "sql.table.sample_rows" else limit, maximum=limit)
        sql = f"SELECT {select_columns} FROM {table_sql}{where_sql}{order_sql} LIMIT {row_limit}"
        result = _execute_sql(conn, sql, row_limit, schema)
        detail = "SQL deterministic row selection executed." if primitive_id == "sql.rows.select" else "SQL deterministic sample rows executed."
        return {"detail": detail, "sql": result.get("sql", sql), "result": result}

    if primitive_id == "sql.table.distinct_values":
        column_name = _resolve_column_name(table, params.get("column_name"))
        if not column_name:
            return None
        row_limit = _normalize_positive_int(params.get("limit"), limit, maximum=limit)
        quoted_column = _quote_identifier(column_name, dialect)
        sql = (
            f"SELECT DISTINCT {quoted_column} AS {quoted_column} "
            f"FROM {table_sql}{where_sql} ORDER BY {quoted_column} ASC LIMIT {row_limit}"
        )
        result = _execute_sql(conn, sql, row_limit, schema)
        return {"detail": "SQL deterministic distinct-values query executed.", "sql": result.get("sql", sql), "result": result}

    if primitive_id == "sql.group.count_by_column":
        column_name = _resolve_column_name(table, params.get("column_name"))
        if not column_name:
            return None
        row_limit = _normalize_positive_int(params.get("limit"), limit, maximum=limit)
        order_direction = str(params.get("order_direction") or "desc").strip().lower()
        if order_direction not in {"asc", "desc"}:
            order_direction = "desc"
        quoted_column = _quote_identifier(column_name, dialect)
        sql = (
            f"SELECT {quoted_column} AS {quoted_column}, COUNT(*) AS {_quote_identifier('count', dialect)} "
            f"FROM {table_sql}{where_sql} GROUP BY {quoted_column} "
            f"ORDER BY {_quote_identifier('count', dialect)} {order_direction.upper()} LIMIT {row_limit}"
        )
        result = _execute_sql(conn, sql, row_limit, schema)
        return {"detail": "SQL deterministic grouped count query executed.", "sql": result.get("sql", sql), "result": result}

    return None


def _finalize_sql_success_payload(
    task: str,
    detail: str,
    sql: str,
    schema: dict,
    stats: dict[str, float],
    result: dict[str, Any] | None,
    selection: dict[str, Any] | None = None,
    *,
    execution_strategy: str,
) -> dict[str, Any]:
    reduction_request = build_sql_reduction_request(task, sql, result)

    payload: dict[str, Any] = {
        "detail": detail,
        "sql": sql,
        "schema": schema,
        "stats": stats,
        "result": result,
        "execution_strategy": execution_strategy,
    }
    if isinstance(reduction_request, dict):
        payload["reduction_request"] = reduction_request
    if isinstance(selection, dict):
        payload["deterministic_primitive"] = selection.get("primitive_id")
        payload["deterministic_selection_reason"] = selection.get("selection_reason")
        payload["fallback_sql"] = selection.get("fallback_sql") or None
    payload["fallback_used"] = execution_strategy in {"selector_fallback_sql", "llm_generated_sql"}
    return payload


def _llm_summarize_rows(task: str, sql: str, columns: list[str], rows: list[dict[str, Any]]) -> str:
    return summarize_sql_rows(task, sql, columns, rows)


def _llm_generate_sql_reduction_command(
    task: str,
    sql: str,
    columns: list[str],
    sample_rows: list[dict[str, Any]],
    row_count: int,
    previous_command: str = "",
    previous_error: str = "",
) -> str:
    return generate_sql_reduction_command(
        task,
        sql,
        columns,
        sample_rows,
        row_count,
        previous_command,
        previous_error,
    )


def _should_reduce_sql_result(task: str, result: dict[str, Any]) -> bool:
    return should_reduce_sql_result(task, result)


def _llm_reduce_sql_result(
    task: str,
    sql: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    row_count: int,
) -> tuple[str, str]:
    reduction_request = {
        "kind": "sql.local_reducer",
        "task": task,
        "source_sql": sql,
        "columns": columns,
        "row_count": row_count,
        "sample_rows": rows[:10],
        "input_format": "json",
    }
    reduction = execute_reduction_request(
        reduction_request,
        {
            "task": task,
            "sql": sql,
            "columns": columns,
            "rows": rows,
            "row_count": row_count,
            "returned_row_count": len(rows),
        },
    )
    reduced_result = reduction.reduced_result if isinstance(reduction.reduced_result, str) else ""
    return reduced_result, reduction.local_reduction_command


def _dsn() -> str | None:
    return os.getenv("SQL_AGENT_DSN") or os.getenv("SQL_DATABASE_URL")


def _dsn_health_status() -> tuple[bool, dict[str, Any]]:
    dsn = _dsn()
    if not dsn:
        return False, {
            "status": "error",
            "configured": False,
            "detail": "SQL agent is not configured. Set SQL_AGENT_DSN or SQL_DATABASE_URL.",
            "dsn_scheme": "",
            "agent_name": os.getenv("SQL_AGENT_NAME", "").strip(),
        }
    dialect = _dialect_from_dsn(dsn)
    if dialect not in {"sqlite", "postgres", "mysql"}:
        return False, {
            "status": "error",
            "configured": False,
            "detail": f"Unsupported SQL_AGENT_DSN scheme: {dialect}",
            "dsn_scheme": dialect,
            "agent_name": os.getenv("SQL_AGENT_NAME", "").strip(),
        }
    return True, {
        "status": "ok",
        "configured": True,
        "detail": "",
        "dsn_scheme": dialect,
        "agent_name": os.getenv("SQL_AGENT_NAME", "").strip(),
    }


def _row_limit() -> int:
    raw = os.getenv("SQL_AGENT_ROW_LIMIT", "100")
    try:
        return max(1, min(int(raw), 1000))
    except ValueError:
        return 100


def _sql_repair_max_attempts() -> int:
    raw = os.getenv("SQL_AGENT_MAX_REPAIR_ATTEMPTS", "10")
    try:
        return max(1, min(int(raw), 50))
    except ValueError:
        return 10


def _dialect_from_dsn(dsn: str) -> str:
    parsed = urlparse(dsn)
    scheme = parsed.scheme.lower()
    if scheme in {"sqlite", "sqlite3"}:
        return "sqlite"
    if scheme in {"postgres", "postgresql"}:
        return "postgres"
    if scheme in {"mysql", "mariadb"}:
        return "mysql"
    return scheme or "unknown"


def _connect(dsn: str):
    dialect = _dialect_from_dsn(dsn)
    parsed = urlparse(dsn)

    if dialect == "sqlite":
        if dsn in {"sqlite:///:memory:", "sqlite://:memory:"}:
            path = ":memory:"
        elif parsed.path:
            path = unquote(parsed.path)
        else:
            path = unquote(dsn.removeprefix("sqlite:///").removeprefix("sqlite://"))
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return dialect, conn

    if dialect == "postgres":
        try:
            import psycopg2
        except ImportError:
            try:
                import psycopg
            except ImportError as exc:
                raise RuntimeError("Postgres support requires psycopg2 or psycopg to be installed.") from exc
            return dialect, psycopg.connect(dsn)
        return dialect, psycopg2.connect(dsn)

    if dialect == "mysql":
        try:
            import pymysql
        except ImportError as exc:
            raise RuntimeError("MySQL/MariaDB support requires pymysql to be installed.") from exc
        return dialect, pymysql.connect(
            host=parsed.hostname,
            port=parsed.port or 3306,
            user=unquote(parsed.username or ""),
            password=unquote(parsed.password or ""),
            database=unquote(parsed.path.lstrip("/")),
            cursorclass=pymysql.cursors.DictCursor,
        )

    raise RuntimeError(f"Unsupported SQL_AGENT_DSN scheme: {dialect}")


def _rows_from_cursor(cursor, limit: int):
    columns = [item[0] for item in cursor.description or []]
    raw_rows = cursor.fetchmany(limit)
    rows = []
    for raw in raw_rows:
        if isinstance(raw, dict):
            rows.append({key: _json_safe(value) for key, value in raw.items()})
        else:
            rows.append({columns[index]: _json_safe(value) for index, value in enumerate(raw)})
    return columns, rows


def _count_query_rows(conn, sql: str):
    count_sql = f"SELECT COUNT(*) AS total_matching_rows FROM ({sql.rstrip().rstrip(';')}) AS openfabric_count_subquery"
    with closing(conn.cursor()) as cursor:
        try:
            cursor.execute(count_sql)
        except Exception:
            _rollback_quietly(conn)
            return None
        row = cursor.fetchone()
    if isinstance(row, dict):
        value = row.get("total_matching_rows")
    elif isinstance(row, (list, tuple)) and row:
        value = row[0]
    else:
        value = None
    return int(value) if isinstance(value, int) else None


def _json_safe(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _rollback_quietly(conn) -> None:
    rollback = getattr(conn, "rollback", None)
    if not callable(rollback):
        return
    try:
        rollback()
    except Exception as exc:
        _debug_log(f"Rollback failed: {type(exc).__name__}: {exc}")


def _identifier_quote_char(dialect: str) -> str:
    if dialect == "mysql":
        return "`"
    return '"'


def _quote_identifier(identifier: str, dialect: str) -> str:
    text = identifier.strip()
    if not text:
        return text
    quote = _identifier_quote_char(dialect)
    if text.startswith(quote) and text.endswith(quote):
        return text
    escaped = text.replace(quote, quote * 2)
    return f"{quote}{escaped}{quote}"


def _quote_qualified_identifier(identifier: str, dialect: str) -> str:
    parts = [part.strip() for part in identifier.split(".") if part.strip()]
    if not parts:
        return identifier.strip()
    return ".".join(_quote_identifier(part, dialect) for part in parts)


def _schema_prompt_text(schema: dict) -> str:
    dialect = schema.get("dialect", "unknown")
    lines = [f"Dialect: {dialect}"]
    for table in schema.get("tables", []):
        schema_name = str(table.get("schema", "") or "").strip()
        table_name = str(table.get("name", "") or "").strip()
        qualified = ".".join(part for part in (schema_name, table_name) if part)
        quoted_table = _quote_qualified_identifier(qualified, dialect) if qualified else ""
        columns = []
        for col in table.get("columns", []):
            col_name = str(col.get("name", "") or "").strip()
            col_type = str(col.get("type", "") or "").strip()
            quoted_col = _quote_identifier(col_name, dialect) if col_name else ""
            nullable = " nullable" if col.get("nullable") else " not null"
            columns.append(f"{quoted_col} {col_type}{nullable}".strip())
        lines.append(f"- table {quoted_table}: {', '.join(columns)}")
        for fk in table.get("foreign_keys", []):
            source = _quote_identifier(str(fk.get('column', '') or '').strip(), dialect)
            target_qualified = ".".join(
                part
                for part in (
                    str(fk.get("references_schema", "") or "").strip(),
                    str(fk.get("references_table", "") or "").strip(),
                )
                if part
            )
            target_column = _quote_identifier(str(fk.get("references_column", "") or "").strip(), dialect)
            lines.append(
                f"  FK {source} -> {_quote_qualified_identifier(target_qualified, dialect)}.{target_column}"
            )
    return "\n".join(lines)


def _sqlite_schema(conn) -> dict:
    tables = []
    with closing(conn.cursor()) as cursor:
        cursor.execute(
            "select name, type from sqlite_master where type in ('table','view') and name not like 'sqlite_%' order by type, name"
        )
        objects = cursor.fetchall()
        for item in objects:
            name = item["name"]
            cursor.execute(f'pragma table_info("{name}")')
            columns = [
                {
                    "name": row["name"],
                    "type": row["type"],
                    "nullable": not bool(row["notnull"]),
                    "primary_key": bool(row["pk"]),
                }
                for row in cursor.fetchall()
            ]
            cursor.execute(f'pragma foreign_key_list("{name}")')
            foreign_keys = [
                {
                    "column": row["from"],
                    "references_table": row["table"],
                    "references_column": row["to"],
                }
                for row in cursor.fetchall()
            ]
            tables.append({"schema": "main", "name": name, "type": item["type"], "columns": columns, "foreign_keys": foreign_keys})
    return {"dialect": "sqlite", "tables": tables}


def _postgres_schema(conn) -> dict:
    query = """
    select table_schema, table_name, column_name, data_type, is_nullable, ordinal_position
    from information_schema.columns
    where table_schema not in ('pg_catalog', 'information_schema')
    order by table_schema, table_name, ordinal_position
    """
    fk_query = """
    select
      tc.table_schema,
      tc.table_name,
      kcu.column_name,
      ccu.table_schema as foreign_table_schema,
      ccu.table_name as foreign_table_name,
      ccu.column_name as foreign_column_name
    from information_schema.table_constraints tc
    join information_schema.key_column_usage kcu
      on tc.constraint_name = kcu.constraint_name and tc.table_schema = kcu.table_schema
    join information_schema.constraint_column_usage ccu
      on ccu.constraint_name = tc.constraint_name and ccu.table_schema = tc.table_schema
    where tc.constraint_type = 'FOREIGN KEY'
    """
    return _information_schema(conn, "postgres", query, fk_query)


def _mysql_schema(conn) -> dict:
    query = """
    select table_schema, table_name, column_name, data_type, is_nullable, ordinal_position
    from information_schema.columns
    where table_schema = database()
    order by table_schema, table_name, ordinal_position
    """
    fk_query = """
    select
      table_schema,
      table_name,
      column_name,
      referenced_table_schema as foreign_table_schema,
      referenced_table_name as foreign_table_name,
      referenced_column_name as foreign_column_name
    from information_schema.key_column_usage
    where referenced_table_name is not null and table_schema = database()
    """
    return _information_schema(conn, "mysql", query, fk_query)


def _information_schema(conn, dialect: str, columns_query: str, fk_query: str) -> dict:
    tables: dict[tuple[str, str], dict] = {}
    with closing(conn.cursor()) as cursor:
        cursor.execute(columns_query)
        for row in cursor.fetchall():
            if not isinstance(row, dict):
                table_schema, table_name, column_name, data_type, is_nullable, _ordinal = row
            else:
                table_schema = row["table_schema"]
                table_name = row["table_name"]
                column_name = row["column_name"]
                data_type = row["data_type"]
                is_nullable = row["is_nullable"]
            key = (table_schema, table_name)
            tables.setdefault(key, {"schema": table_schema, "name": table_name, "type": "table", "columns": [], "foreign_keys": []})
            tables[key]["columns"].append({"name": column_name, "type": data_type, "nullable": str(is_nullable).upper() == "YES"})

        cursor.execute(fk_query)
        for row in cursor.fetchall():
            if not isinstance(row, dict):
                table_schema, table_name, column_name, foreign_schema, foreign_table, foreign_column = row
            else:
                table_schema = row["table_schema"]
                table_name = row["table_name"]
                column_name = row["column_name"]
                foreign_schema = row["foreign_table_schema"]
                foreign_table = row["foreign_table_name"]
                foreign_column = row["foreign_column_name"]
            key = (table_schema, table_name)
            if key in tables:
                tables[key]["foreign_keys"].append(
                    {
                        "column": column_name,
                        "references_schema": foreign_schema,
                        "references_table": foreign_table,
                        "references_column": foreign_column,
                    }
                )
    return {"dialect": dialect, "tables": list(tables.values())}


def _introspect(dialect: str, conn) -> dict:
    if dialect == "sqlite":
        return _sqlite_schema(conn)
    if dialect == "postgres":
        return _postgres_schema(conn)
    if dialect == "mysql":
        return _mysql_schema(conn)
    raise RuntimeError(f"Unsupported SQL dialect: {dialect}")


def _schema_summary(schema: dict) -> str:
    return _schema_prompt_text(schema)


def _schema_count_like_request(focus: str | None, task: str | None) -> bool:
    text = " ".join(part for part in (str(focus or "").strip(), str(task or "").strip()) if part).lower()
    if not text:
        return False
    return text.startswith("count ") or any(
        token in text for token in ("how many", "count ", "number of", "total count", "total number")
    )


def _schema_listing_request(focus: str | None, task: str | None) -> bool:
    text = " ".join(part for part in (str(focus or "").strip(), str(task or "").strip()) if part).lower()
    if not text:
        return False
    return any(
        token in text
        for token in (
            "list ",
            "show ",
            "display ",
            "describe ",
            "inspect ",
            "include ",
            "first few",
        )
    )


def _schema_focus_terms(focus: str | None, task: str | None) -> set[str]:
    tokens = set()
    for text in (focus or "", task or ""):
        lowered = str(text).lower()
        if re.search(r"\bschemas\b", lowered) or "schema names" in lowered:
            tokens.add("schemas")
        if "table" in lowered:
            tokens.add("tables")
        if "column" in lowered:
            tokens.add("columns")
        if "relationship" in lowered or "foreign key" in lowered or "relation" in lowered:
            tokens.add("relationships")
        if "schema" in lowered:
            tokens.add("schema")
    return tokens


def _schemas_only_schema_request(focus: str | None, task: str | None) -> bool:
    if _schema_count_like_request(focus, task) and not _schema_listing_request(focus, task):
        return False
    terms = _schema_focus_terms(focus, task)
    return "schemas" in terms and "tables" not in terms and "columns" not in terms and "relationships" not in terms


def _tables_only_schema_request(focus: str | None, task: str | None) -> bool:
    if _schema_count_like_request(focus, task) and not _schema_listing_request(focus, task):
        return False
    terms = _schema_focus_terms(focus, task)
    return "tables" in terms and "schemas" not in terms and "columns" not in terms and "relationships" not in terms


def _columns_only_schema_request(focus: str | None, task: str | None) -> bool:
    if _schema_count_like_request(focus, task) and not _schema_listing_request(focus, task):
        return False
    terms = _schema_focus_terms(focus, task)
    return "columns" in terms and "schemas" not in terms and "tables" not in terms and "relationships" not in terms


def _relationships_only_schema_request(focus: str | None, task: str | None) -> bool:
    if _schema_count_like_request(focus, task) and not _schema_listing_request(focus, task):
        return False
    terms = _schema_focus_terms(focus, task)
    return "relationships" in terms and "schemas" not in terms and "tables" not in terms and "columns" not in terms


def _schema_schemas_result(schema: dict) -> dict[str, Any]:
    seen = sorted(
        {
            str(table.get("schema") or "").strip()
            for table in schema.get("tables", [])
            if str(table.get("schema") or "").strip()
        }
    )
    rows = [{"schema": name} for name in seen]
    return {
        "columns": ["schema"],
        "rows": rows,
        "row_count": len(rows),
        "limit": len(rows),
    }


def _schema_tables_result(schema: dict) -> dict[str, Any]:
    rows = []
    for table in schema.get("tables", []):
        rows.append(
            {
                "schema": table.get("schema"),
                "table": table.get("name"),
                "type": table.get("type", "table"),
            }
        )
    return {
        "columns": ["schema", "table", "type"],
        "rows": rows,
        "row_count": len(rows),
        "limit": len(rows),
    }


def _schema_columns_result(schema: dict) -> dict[str, Any]:
    rows = []
    for table in schema.get("tables", []):
        schema_name = table.get("schema")
        table_name = table.get("name")
        for column in table.get("columns", []):
            rows.append(
                {
                    "schema": schema_name,
                    "table": table_name,
                    "column": column.get("name"),
                    "type": column.get("type"),
                    "nullable": column.get("nullable"),
                }
            )
    return {
        "columns": ["schema", "table", "column", "type", "nullable"],
        "rows": rows,
        "row_count": len(rows),
        "limit": len(rows),
    }


def _schema_relationships_result(schema: dict) -> dict[str, Any]:
    rows = []
    for table in schema.get("tables", []):
        schema_name = table.get("schema")
        table_name = table.get("name")
        for foreign_key in table.get("foreign_keys", []):
            ref_schema = foreign_key.get("references_schema")
            ref_table = foreign_key.get("references_table")
            ref_column = foreign_key.get("references_column")
            qualified_ref = ".".join(part for part in (str(ref_schema or "").strip(), str(ref_table or "").strip()) if part)
            rows.append(
                {
                    "schema": schema_name,
                    "table": table_name,
                    "column": foreign_key.get("column"),
                    "references": f"{qualified_ref}.{ref_column}" if qualified_ref and ref_column else qualified_ref or ref_column,
                }
            )
    return {
        "columns": ["schema", "table", "column", "references"],
        "rows": rows,
        "row_count": len(rows),
        "limit": len(rows),
    }


def _focused_schema_result(focus: str | None, task: str | None, schema: dict) -> tuple[str, dict[str, Any]] | None:
    if _schemas_only_schema_request(focus, task):
        return "Database schemas listed.", _schema_schemas_result(schema)
    if _tables_only_schema_request(focus, task):
        return "Database tables listed.", _schema_tables_result(schema)
    if _columns_only_schema_request(focus, task):
        return "Database columns listed.", _schema_columns_result(schema)
    if _relationships_only_schema_request(focus, task):
        return "Database relationships listed.", _schema_relationships_result(schema)
    return None


def _schema_identifier_catalog(schema: dict) -> str:
    lines = ["Exact identifiers:"]
    for table in schema.get("tables", []):
        schema_name = str(table.get("schema", "") or "").strip()
        table_name = str(table.get("name", "") or "").strip()
        if not table_name:
            continue
        qualified = ".".join(part for part in (schema_name, table_name) if part)
        lines.append(f"- table: {qualified}")
        columns = [
            str(col.get("name", "") or "").strip()
            for col in table.get("columns", [])
            if str(col.get("name", "") or "").strip()
        ]
        if columns:
            lines.append(f"  columns: {', '.join(columns)}")
    return "\n".join(lines)


def _schema_identifier_sets(schema: dict) -> dict[str, set[str]]:
    schemas: set[str] = set()
    tables: set[str] = set()
    qualified_tables: set[str] = set()
    columns: set[str] = set()
    qualified_columns: set[str] = set()

    for table in schema.get("tables", []):
        schema_name = str(table.get("schema", "") or "").strip()
        table_name = str(table.get("name", "") or "").strip()
        if schema_name:
            schemas.add(schema_name)
        if table_name:
            tables.add(table_name)
        qualified_table = ".".join(part for part in (schema_name, table_name) if part)
        if qualified_table:
            qualified_tables.add(qualified_table)

        for column in table.get("columns", []):
            column_name = str(column.get("name", "") or "").strip()
            if not column_name:
                continue
            columns.add(column_name)
            if table_name:
                qualified_columns.add(f"{table_name}.{column_name}")
            if qualified_table:
                qualified_columns.add(f"{qualified_table}.{column_name}")

    return {
        "schemas": schemas,
        "tables": tables,
        "qualified_tables": qualified_tables,
        "columns": columns,
        "qualified_columns": qualified_columns,
    }


SQL_TOKEN_PATTERN = re.compile(
    r"""
    (?P<double_quote>"(?:[^"]|"")*")
    |(?P<single_quote>'(?:''|[^'])*')
    |(?P<line_comment>--[^\n]*)
    |(?P<block_comment>/\*.*?\*/)
    |(?P<whitespace>\s+)
    |(?P<identifier>[A-Za-z_][A-Za-z0-9_$]*)
    |(?P<symbol>.)
    """,
    re.VERBOSE | re.DOTALL,
)


def _tokenize_sql(sql: str) -> list[dict[str, str]]:
    return [
        {"kind": match.lastgroup or "symbol", "text": match.group(0)}
        for match in SQL_TOKEN_PATTERN.finditer(sql)
    ]


def _next_meaningful_token(tokens: list[dict[str, str]], start: int) -> tuple[int, dict[str, str]] | None:
    for index in range(start, len(tokens)):
        if tokens[index]["kind"] != "whitespace":
            return index, tokens[index]
    return None


def _postgres_safe_sql(sql: str, schema: dict) -> str:
    if schema.get("dialect") != "postgres":
        return sql

    identifiers = _schema_identifier_sets(schema)
    tokens = _tokenize_sql(sql)
    rewritten: list[str] = []
    index = 0

    while index < len(tokens):
        token = tokens[index]
        kind = token["kind"]
        text = token["text"]

        if kind != "identifier":
            rewritten.append(text)
            index += 1
            continue

        next_token_info = _next_meaningful_token(tokens, index + 1)
        if next_token_info and next_token_info[1]["text"] == "(":
            rewritten.append(text)
            index += 1
            continue

        dot_token_info = _next_meaningful_token(tokens, index + 1)
        if dot_token_info and dot_token_info[1]["text"] == ".":
            rhs_token_info = _next_meaningful_token(tokens, dot_token_info[0] + 1)
            if rhs_token_info and rhs_token_info[1]["kind"] == "identifier":
                qualified_pair = f"{text}.{rhs_token_info[1]['text']}"
                if qualified_pair in identifiers["qualified_tables"] or qualified_pair in identifiers["qualified_columns"]:
                    rewritten.append(_quote_identifier(text, "postgres"))
                elif text in identifiers["schemas"] or text in identifiers["tables"]:
                    rewritten.append(_quote_identifier(text, "postgres"))
                else:
                    rewritten.append(text)
                index += 1
                continue

        if text in identifiers["tables"] or text in identifiers["columns"] or text in identifiers["schemas"]:
            rewritten.append(_quote_identifier(text, "postgres"))
        else:
            rewritten.append(text)
        index += 1

    return "".join(rewritten)


def _parse_strict_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _single_sql_query_spec_from_object(value: dict[str, Any]) -> list[dict[str, str]]:
    if set(value.keys()) != {"sql", "reason"}:
        return []
    sql = value.get("sql")
    reason = value.get("reason")
    if not isinstance(sql, str) or not sql.strip():
        return []
    if not isinstance(reason, str) or not reason.strip():
        return []
    return [{"label": reason.strip(), "sql": sql.strip()}]


def _strict_sql_query_specs_from_object(value: dict[str, Any]) -> list[dict[str, str]]:
    if set(value.keys()) == {"queries"}:
        queries = value.get("queries")
        if not isinstance(queries, list) or not queries:
            return []
        specs = []
        seen = set()
        for item in queries:
            if not isinstance(item, dict) or set(item.keys()) != {"label", "sql", "reason"}:
                return []
            label = item.get("label")
            sql = item.get("sql")
            reason = item.get("reason")
            if not all(isinstance(field, str) and field.strip() for field in (label, sql, reason)):
                return []
            sql_text = sql.strip()
            if sql_text in seen:
                continue
            seen.add(sql_text)
            specs.append({"label": label.strip(), "sql": sql_text})
        return specs
    return _single_sql_query_spec_from_object(value)


def _sql_generation_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "sql_generation_response",
            "strict": True,
            "schema": {
                "oneOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "sql": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["sql", "reason"],
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "queries": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "label": {"type": "string"},
                                        "sql": {"type": "string"},
                                        "reason": {"type": "string"},
                                    },
                                    "required": ["label", "sql", "reason"],
                                },
                            }
                        },
                        "required": ["queries"],
                    },
                ]
            },
        },
    }


def _sql_repair_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "sql_repair_response",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "sql": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["sql", "reason"],
            },
        },
    }


def _normalize_sql_for_compare(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).strip().lower()


def _same_query_specs(left: list[dict[str, str]], right: list[dict[str, str]]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        _normalize_sql_for_compare(left_spec.get("sql", "")) == _normalize_sql_for_compare(right_spec.get("sql", ""))
        for left_spec, right_spec in zip(left, right)
    )


def _llm_sql_queries(task: str, schema: dict) -> list[dict[str, str]]:
    api_key, base_url, timeout_seconds, model = _sql_llm_transport_settings("gpt-4o-mini")

    prompt = (
        "You generate read-only SQL for a configured database.\n"
        "Return exactly one JSON object and no surrounding prose, code fences, or markdown.\n"
        "For one query return exactly {\"sql\":\"...\",\"reason\":\"...\"}. "
        "When the user explicitly asks for separate queries, return exactly "
        "{\"queries\":[{\"label\":\"...\",\"sql\":\"...\",\"reason\":\"...\"}]}.\n"
        "3. AGGREGATION & COUNTS:\n"
        "- For 'how many' or 'count' questions about entities matching a criteria:\n"
        "  * If the criteria involves a simple filter, use COUNT(*).\n"
        "  * If the criteria involves a count or aggregate of related items (requiring GROUP BY and HAVING), ALWAYS use a subquery to get the final total.\n"
        "  * INCORRECT: SELECT COUNT(ID) FROM table GROUP BY ID HAVING COUNT(items) > 5 (This returns many rows of '1').\n"
        "  * CORRECT: SELECT COUNT(*) FROM (SELECT ID FROM table GROUP BY ID HAVING COUNT(items) > 5) as sub.\n\n"

        "Rules:\n"
        "- Generate one read-only query unless the user explicitly asks for separate queries.\n"
        "- If multiple queries are requested, put all query objects inside one top-level JSON object under the queries array.\n"
        "- Do not return any keys other than sql and reason for a single-query response.\n"
        "- Do not return any keys other than queries for a multi-query response.\n"
        "- In each queries item, return only label, sql, and reason.\n"
        "- Use only tables and columns present in the schema.\n"
        "- Use schema-qualified table names exactly as shown in the schema, for example schema_name.table_name.\n"
        "- Never refer to a table by bare name when the schema summary shows it as schema.table.\n"
        "- Identifiers are exact code tokens, not natural-language paraphrases.\n"
        "- Never singularize, pluralize, translate, normalize, or rename schema, table, or column identifiers.\n"
        "- Copy schema, table, and column names verbatim from the schema catalog.\n"
        "- Quote identifiers exactly as shown in the schema whenever case, spaces, punctuation, or reserved words could matter.\n"
        "- For postgres and sqlite use double quotes for identifiers. For mysql use backticks.\n"
        "- You may quote all schema, table, and column identifiers consistently, even when not strictly required.\n"
        "- Use foreign-key relationships from the schema for joins.\n"
        "- Prefer SELECT or WITH queries. Do not generate mutating SQL.\n"
        "- Add a reasonable LIMIT unless the user asks for aggregation only.\n"
        "- Use the SQL dialect from the schema.\n\n"

        "EXAMPLES:\n"
        "Q: how many patients have more than 5 studies\n"
        "A: {\"sql\":\"SELECT COUNT(*) FROM (SELECT \\\"PatientID\\\" FROM \\\"flathr\\\".\\\"Study\\\" GROUP BY \\\"PatientID\\\" HAVING COUNT(\\\"StudyInstanceUID\\\") > 5) AS sub;\",\"reason\":\"Counting total patients meeting the study count threshold using a subquery to avoid per-group counts.\"}\n\n"

        "Schema:\n"
        f"{_schema_summary(schema)}\n"
        f"{_schema_identifier_catalog(schema)}\n"
        f"User question: {task}"
    )
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You produce strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            "response_format": _sql_generation_response_format(),
            "temperature": 0,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    log_raw("SQL_LLM_RAW", content)
    parsed = _parse_strict_json_object(content)
    if parsed is None:
        return []
    return _strict_sql_query_specs_from_object(parsed)


def _repair_sql_query(
    task: str,
    schema: dict,
    failing_sql: str,
    error_text: str,
    previous_repair_sql: str = "",
    previous_repair_error: str = "",
) -> list[dict[str, str]]:
    api_key, base_url, timeout_seconds, model = _sql_llm_transport_settings("gpt-4o-mini")

    prompt = (
        "You repair one read-only SQL query after execution failed.\n"
        "Return exactly one JSON object in the form {\"sql\":\"...\",\"reason\":\"...\"} and no surrounding prose, code fences, or markdown.\n"
        "Rules:\n"
        "- Return only the keys sql and reason.\n"
        "- Preserve the user's intent.\n"
        "- Use only tables and columns present in the schema.\n"
        "- Fix identifier quoting, schema qualification, reserved words, join paths, and dialect syntax issues.\n"
        "- Identifiers are exact code tokens, not natural-language paraphrases.\n"
        "- Never singularize, pluralize, translate, normalize, or rename schema, table, or column identifiers.\n"
        "- Copy schema, table, and column names verbatim from the schema catalog.\n"
        "- Quote identifiers exactly as shown in the schema whenever case, spaces, punctuation, or reserved words could matter.\n"
        "- For postgres and sqlite use double quotes for identifiers. For mysql use backticks.\n"
        "- You must return a materially different SQL query when the failing query is invalid. Do not repeat the same broken SQL.\n"
        "- If a referenced column does not exist on an alias, find a valid join path using the schema foreign keys instead of reusing the same join.\n"
        "- Validate every selected column, filter column, and JOIN predicate against the schema catalog before answering.\n"
        "- Return a single read-only SQL query.\n"
        "Schema:\n"
        f"{_schema_summary(schema)}\n"
        f"{_schema_identifier_catalog(schema)}\n"
        f"Original user request: {task}\n"
        f"Failing SQL:\n{failing_sql}\n"
        f"Database error:\n{error_text}"
    )
    if previous_repair_sql.strip():
        prompt += (
            f"\nPrevious failed repair SQL:\n{previous_repair_sql}\n"
            f"Previous failed repair error:\n{previous_repair_error}\n"
            "Your previous repair was still invalid. Return a different SQL query that corrects the failure.\n"
        )
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You produce strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            "response_format": _sql_repair_response_format(),
            "temperature": 0,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    log_raw("SQL_LLM_REPAIR_RAW", content)
    parsed = _parse_strict_json_object(content)
    if parsed is None:
        return []
    return _single_sql_query_spec_from_object(parsed)


def _read_only_sql(sql: str) -> bool:
    stripped = sql.strip().rstrip(";").strip()
    if ";" in stripped:
        return False
    if MUTATING_PATTERN.search(stripped):
        return False
    return stripped.lower().startswith(READ_ONLY_PREFIXES)


def _execute_sql(conn, sql: str, limit: int, schema: dict):
    if not _read_only_sql(sql):
        raise RuntimeError("Rejected non-read-only SQL.")
    sql = _postgres_safe_sql(sql, schema)
    with closing(conn.cursor()) as cursor:
        try:
            cursor.execute(sql)
        except Exception:
            _rollback_quietly(conn)
            raise
        columns, rows = _rows_from_cursor(cursor, limit)
    returned_row_count = len(rows)
    total_matching_rows = _count_query_rows(conn, sql)
    if not isinstance(total_matching_rows, int):
        total_matching_rows = returned_row_count
    return {
        "sql": sql,
        "columns": columns,
        "rows": rows,
        "row_count": total_matching_rows,
        "returned_row_count": returned_row_count,
        "total_matching_rows": total_matching_rows,
        "truncated": total_matching_rows > returned_row_count,
        "limit": limit,
    }


def _execute_sql_queries(conn, query_specs: list[dict[str, str]], limit: int, schema: dict):
    if len(query_specs) == 1:
        return _execute_sql(conn, query_specs[0]["sql"], limit, schema)

    query_results = []
    rows = []
    scalar_rows = True
    for index, spec in enumerate(query_specs, start=1):
        label = spec.get("label") or f"query {index}"
        result = _execute_sql(conn, spec["sql"], limit, schema)
        query_results.append({"label": label, **result})

        result_rows = result.get("rows", [])
        columns = result.get("columns", [])
        if len(result_rows) == 1 and len(columns) == 1:
            rows.append({"query": label, "value": result_rows[0].get(columns[0])})
        else:
            scalar_rows = False
            rows.append(
                {
                    "query": label,
                    "row_count": result.get("total_matching_rows", result.get("row_count", 0)),
                    "result": json.dumps(result_rows, ensure_ascii=True),
                }
            )

    if scalar_rows:
        columns = ["query", "value"]
    else:
        columns = ["query", "row_count", "result"]
    return {
        "sql": "\n\n".join(item["sql"] for item in query_results),
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "returned_row_count": len(rows),
        "total_matching_rows": len(rows),
        "truncated": False,
        "limit": limit,
        "queries": query_results,
    }


def _should_retry_sql_repair(exc: Exception) -> bool:
    return bool(IDENTIFIER_ERROR_PATTERN.search(f"{type(exc).__name__}: {exc}"))


def _repair_sql_with_retries(
    conn,
    schema: dict,
    query_task: str,
    query_specs: list[dict[str, str]],
    limit: int,
    stats: dict[str, float],
):
    max_attempts = _sql_repair_max_attempts()
    repair_started = time.perf_counter()
    previous_repair_sql = ""
    previous_repair_error = ""
    current_specs = query_specs

    try:
        result = _execute_sql_queries(conn, current_specs, limit, schema)
        stats["sql_repair_attempts"] = 0.0
        stats["sql_repair_ms"] = _elapsed_ms(repair_started)
        return result, current_specs
    except Exception as exc:
        last_error = exc
        last_error_text = f"{type(exc).__name__}: {exc}"
        if not _should_retry_sql_repair(exc):
            stats["sql_repair_attempts"] = 0.0
            stats["sql_repair_ms"] = _elapsed_ms(repair_started)
            raise

    for attempt in range(1, max_attempts + 1):
        _rollback_quietly(conn)
        failing_sql = "\n\n".join(spec.get("sql", "") for spec in current_specs if isinstance(spec, dict))
        repaired_specs = _repair_sql_query(
            query_task,
            schema,
            failing_sql,
            last_error_text,
            previous_repair_sql=previous_repair_sql,
            previous_repair_error=previous_repair_error,
        )
        stats["sql_repair_attempts"] = float(attempt)

        if not repaired_specs:
            stats["sql_repair_ms"] = _elapsed_ms(repair_started)
            raise last_error

        repaired_sql = "\n\n".join(spec.get("sql", "") for spec in repaired_specs if isinstance(spec, dict))
        if _same_query_specs(repaired_specs, current_specs):
            previous_repair_sql = repaired_sql
            previous_repair_error = last_error_text
        else:
            previous_repair_sql = ""
            previous_repair_error = ""
        current_specs = repaired_specs

        try:
            result = _execute_sql_queries(conn, current_specs, limit, schema)
            stats["sql_repair_ms"] = _elapsed_ms(repair_started)
            return result, current_specs
        except Exception as exc:
            last_error = exc
            last_error_text = f"{type(exc).__name__}: {exc}"
            if not _should_retry_sql_repair(exc):
                stats["sql_repair_ms"] = _elapsed_ms(repair_started)
                raise

    stats["sql_repair_ms"] = _elapsed_ms(repair_started)
    raise last_error


def _is_schema_request(task: str) -> bool:
    task_lc = task.lower()
    return any(
        token in task_lc
        for token in (
            "schema",
            "schemas",
            "tables",
            "columns",
            "relationship",
            "relationships",
            "relation",
            "relations",
            "foreign key",
            "foreign keys",
            "describe database",
        )
    )


def _is_sql_task(task: str) -> bool:
    task_lc = task.lower()
    return any(
        token in task_lc
        for token in (
            "sql",
            "database",
            "db",
            "schema",
            "table",
            "tables",
            "query",
            "rows",
            "columns",
            "join",
            "foreign key",
        )
    )


def _needs_decomposition(detail: str):
    return {
        "emits": [
            {
                "event": "task.result",
                "payload": {
                    "detail": detail,
                    "status": "needs_decomposition",
                    "error": detail,
                    "replan_hint": {
                        "reason": detail,
                        "failure_class": "needs_decomposition",
                        "suggested_capabilities": ["sql_runner"],
                    },
                },
            }
        ]
    }


@app.get("/healthz")
def healthz():
    ok, payload = _dsn_health_status()
    return JSONResponse(payload, status_code=200 if ok else 503)


@app.post("/handle", response_model=EventResponse)
@with_node_envelope("sql_runner", "executor")
def handle_event(req: EventRequest):
    if req.event == "sql.query":
        task = str(req.payload.get("question") or req.payload.get("query") or "")
        classification_task = task
        execution_task = task
        provided_sql = req.payload.get("sql")
        instruction = {"operation": "query_from_request", "question": task}
    elif req.event == "task.plan":
        plan_context = task_plan_context(req.payload)
        classification_task = plan_context.classification_task
        execution_task = plan_context.execution_task
        explicitly_targeted = plan_context.targets("sql_runner")
        instruction = req.payload.get("instruction") if isinstance(req.payload.get("instruction"), dict) else {}
        if not classification_task or (not explicitly_targeted and not _is_sql_task(classification_task)):
            return noop()
        provided_sql = req.payload.get("sql")
    else:
        return noop()

    dsn = _dsn()
    if not dsn:
        return failure_result(
            "SQL agent is not configured. Set SQL_AGENT_DSN or SQL_DATABASE_URL.",
            error="SQL agent is not configured. Set SQL_AGENT_DSN or SQL_DATABASE_URL.",
        )

    total_started = time.perf_counter()
    stats: dict[str, float] = {}
    try:
        started = time.perf_counter()
        dialect, conn = _connect(dsn)
        stats["connect_ms"] = _elapsed_ms(started)
        with closing(conn):
            started = time.perf_counter()
            schema = _introspect(dialect, conn)
            stats["schema_ms"] = _elapsed_ms(started)
            operation = instruction.get("operation") if isinstance(instruction, dict) else None
            if operation == "inspect_schema":
                focus = instruction.get("focus") if isinstance(instruction, dict) else None
                stats["total_ms"] = _elapsed_ms(total_started)
                focused_schema = _focused_schema_result(
                    focus if isinstance(focus, str) else None,
                    classification_task,
                    schema,
                )
                if focused_schema is not None:
                    detail, focused_result = focused_schema
                    return emit(
                        "sql.result",
                        {
                            "detail": detail,
                            "stats": stats,
                            "result": focused_result,
                        },
                    )
                return emit(
                    "sql.result",
                    {
                        "detail": "Database schema introspected.",
                        "schema": schema,
                        "stats": stats,
                        # NOTE: result intentionally omitted for schema introspection.
                        # Setting result=schema caused the full schema to be sent twice
                        # to the synthesizer, once via schema and once via result.
                    },
                )
            if operation == "execute_sql":
                provided_sql = instruction.get("sql")
            elif operation == "sample_rows":
                table = instruction.get("table")
                limit = instruction.get("limit", 5)
                if isinstance(table, str) and table.strip():
                    provided_sql = (
                        f"SELECT * FROM {_quote_qualified_identifier(table.strip(), dialect)} "
                        f"LIMIT {int(limit) if isinstance(limit, (int, float)) else 5}"
                    )
            query_task = (
                instruction.get("question")
                if isinstance(instruction, dict) and isinstance(instruction.get("question"), str)
                else execution_task
            )
            selection: dict[str, Any] | None = None
            execution_strategy = "explicit_sql"
            if isinstance(provided_sql, str) and provided_sql.strip():
                query_specs = [{"label": "provided SQL", "sql": provided_sql.strip()}]
                stats["sql_generation_ms"] = 0
            else:
                execution_strategy = "llm_generated_sql"
                selection_started = time.perf_counter()
                raw_selection = _llm_select_sql_strategy(query_task, schema)
                stats["deterministic_selection_ms"] = _elapsed_ms(selection_started)
                selection = _normalize_sql_selection(raw_selection)
                if _selection_should_use_fallback(query_task, selection):
                    selection = dict(selection)
                    selection["primitive_id"] = "fallback_only"

                deterministic_payload = None
                if selection.get("primitive_id") and selection.get("primitive_id") != "fallback_only":
                    deterministic_started = time.perf_counter()
                    deterministic_payload = _execute_sql_deterministic_selection(
                        conn,
                        dialect,
                        schema,
                        query_task,
                        selection,
                        _row_limit(),
                    )
                    stats["deterministic_execution_ms"] = _elapsed_ms(deterministic_started)
                if isinstance(deterministic_payload, dict):
                    stats["sql_generation_ms"] = 0
                    stats["query_ms"] = stats.get("deterministic_execution_ms", 0.0)
                    stats["total_ms"] = _elapsed_ms(total_started)
                    return emit(
                        "sql.result",
                        _finalize_sql_success_payload(
                            query_task,
                            str(deterministic_payload.get("detail") or "SQL deterministic primitive executed."),
                            str(deterministic_payload.get("sql") or ""),
                            schema,
                            stats,
                            deterministic_payload.get("result")
                            if isinstance(deterministic_payload.get("result"), dict)
                            else None,
                            selection,
                            execution_strategy="deterministic",
                        ),
                    )

                query_specs = _sql_fallback_query_specs_from_selection(selection)
                if query_specs:
                    stats["sql_generation_ms"] = 0
                    execution_strategy = "selector_fallback_sql"
                else:
                    started = time.perf_counter()
                    query_specs = _llm_sql_queries(query_task, schema)
                    stats["sql_generation_ms"] = _elapsed_ms(started)
            if not query_specs:
                stats["total_ms"] = _elapsed_ms(total_started)
                return needs_decomposition(
                    "Could not generate a SQL query.",
                    suggested_capabilities=["sql_runner"],
                    result={"ok": False, "stats": stats},
                )
            started = time.perf_counter()
            result, query_specs = _repair_sql_with_retries(conn, schema, query_task, query_specs, _row_limit(), stats)
            stats["query_ms"] = _elapsed_ms(started)
            stats["total_ms"] = _elapsed_ms(total_started)
            sql = result.get("sql", "")
            return emit(
                "sql.result",
                _finalize_sql_success_payload(
                    query_task,
                    "SQL query executed.",
                    sql,
                    schema,
                    stats,
                    result,
                    selection,
                    execution_strategy=execution_strategy,
                ),
            )
    except Exception as exc:
        stats["total_ms"] = _elapsed_ms(total_started)
        _debug_log(f"SQL task failed: {type(exc).__name__}: {exc}")
        return failure_result(
            f"SQL task failed: {type(exc).__name__}: {exc}",
            error=f"{type(exc).__name__}: {exc}",
            result={"ok": False, "stats": stats} if stats else None,
        )
