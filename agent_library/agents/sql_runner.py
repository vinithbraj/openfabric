import json
import os
import re
import sqlite3
import time
from contextlib import closing
from typing import Any
from urllib.parse import unquote, urlparse

import requests
from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse, task_plan_context
from runtime.console import log_debug, log_raw

app = FastAPI()

AGENT_METADATA = {
    "description": (
        "Connects to a configured SQL database, introspects schemas/tables/columns/"
        "relationships, generates read-only SQL from natural language, executes it, "
        "and returns query results."
    ),
    "capability_domains": [
        "sql",
        "database",
        "schema_introspection",
        "query_generation",
        "read_only_analytics",
        "data_retrieval",
    ],
    "action_verbs": [
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
    "side_effect_policy": "read_only_sql_with_safety_checks",
    "safety_enforced_by_agent": True,
    "routing_notes": [
        "Use for requests about SQL databases, schemas, tables, columns, relationships, or data questions that should be answered by querying a configured database.",
        "Requires SQL_AGENT_DSN or SQL_DATABASE_URL to be configured in the agent environment.",
        "Only read-only SQL is executed. Mutating statements are rejected.",
    ],
    "methods": [
        {
            "name": "introspect_database",
            "event": "task.plan",
            "when": "Lists schemas, tables, columns, and relationships for the configured SQL database.",
            "intent_tags": ["sql_schema", "database_introspection"],
            "examples": ["show database schema", "list all SQL tables", "describe relationships in the database"],
        },
        {
            "name": "answer_database_question",
            "event": "task.plan",
            "when": "Generates and executes read-only SQL to answer a natural-language database question.",
            "intent_tags": ["sql_query", "database_question", "analytics"],
            "examples": ["how many users signed up last week", "show top 10 customers by revenue"],
        },
    ],
}

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


def _dsn() -> str | None:
    return os.getenv("SQL_AGENT_DSN") or os.getenv("SQL_DATABASE_URL")


def _row_limit() -> int:
    raw = os.getenv("SQL_AGENT_ROW_LIMIT", "100")
    try:
        return max(1, min(int(raw), 1000))
    except ValueError:
        return 100


def _sql_repair_max_attempts() -> int:
    raw = os.getenv("SQL_AGENT_MAX_REPAIR_ATTEMPTS", "2")
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


def _extract_json_values(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    values = []
    for match in re.finditer(r"{", text):
        try:
            value, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def _sql_query_specs_from_json(values: list[dict[str, Any]]) -> list[dict[str, str]]:
    specs = []
    for value in values:
        queries = value.get("queries")
        if isinstance(queries, list):
            for index, item in enumerate(queries, start=1):
                if not isinstance(item, dict):
                    continue
                sql = item.get("sql")
                if isinstance(sql, str) and sql.strip():
                    label = item.get("label") or item.get("name") or item.get("reason") or f"query {index}"
                    specs.append({"label": str(label), "sql": sql.strip()})
        sql = value.get("sql")
        if isinstance(sql, str) and sql.strip():
            label = value.get("label") or value.get("name") or value.get("reason") or f"query {len(specs) + 1}"
            specs.append({"label": str(label), "sql": sql.strip()})
    deduped = []
    seen = set()
    for spec in specs:
        key = spec["sql"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)
    return deduped


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
    api_key = os.getenv("LLM_OPS_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_OPS_BASE_URL")
    model = os.getenv("LLM_OPS_MODEL")
    timeout_seconds = float(os.getenv("LLM_OPS_TIMEOUT_SECONDS", "300"))
    if not api_key:
        raise RuntimeError("LLM_OPS_API_KEY is not set")
    if api_key.startswith("sk-") and api_key.lower() != "dummy":
        base_url = "https://api.openai.com/v1"
        if not model:
            model = "gpt-4.1"
    elif not base_url:
        base_url = "https://api.openai.com/v1"
    if not model:
        model = "gpt-4o-mini"

    prompt = (
        "You generate read-only SQL for a configured database.\n"
        "Return JSON only. For one query return {\"sql\":\"...\",\"reason\":\"...\"}. "
        "When the user explicitly asks for separate queries, return {\"queries\":[{\"label\":\"...\",\"sql\":\"...\",\"reason\":\"...\"}]}.\n"
        "Rules:\n"
        "- Generate one read-only query unless the user explicitly asks for separate queries.\n"
        "- If multiple queries are requested, put all query objects inside one top-level JSON object under the queries array.\n"
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
        "- Use the SQL dialect from the schema.\n"
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
            "temperature": 0,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    log_raw("SQL_LLM_RAW", content)
    return _sql_query_specs_from_json(_extract_json_values(content))


def _repair_sql_query(
    task: str,
    schema: dict,
    failing_sql: str,
    error_text: str,
    previous_repair_sql: str = "",
    previous_repair_error: str = "",
) -> list[dict[str, str]]:
    api_key = os.getenv("LLM_OPS_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_OPS_BASE_URL")
    model = os.getenv("LLM_OPS_MODEL")
    timeout_seconds = float(os.getenv("LLM_OPS_TIMEOUT_SECONDS", "300"))
    if not api_key:
        raise RuntimeError("LLM_OPS_API_KEY is not set")
    if api_key.startswith("sk-") and api_key.lower() != "dummy":
        base_url = "https://api.openai.com/v1"
        if not model:
            model = "gpt-4.1"
    elif not base_url:
        base_url = "https://api.openai.com/v1"
    if not model:
        model = "gpt-4o-mini"

    prompt = (
        "You repair one read-only SQL query after execution failed.\n"
        "Return JSON only in the form {\"sql\":\"...\",\"reason\":\"...\"}.\n"
        "Rules:\n"
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
            "temperature": 0,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    log_raw("SQL_LLM_REPAIR_RAW", content)
    return _sql_query_specs_from_json(_extract_json_values(content))


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
    return {"sql": sql, "columns": columns, "rows": rows, "row_count": len(rows), "limit": limit}


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
                    "row_count": result.get("row_count", 0),
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


@app.post("/handle", response_model=EventResponse)
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
            return {"emits": []}
        provided_sql = req.payload.get("sql")
    else:
        return {"emits": []}

    dsn = _dsn()
    if not dsn:
        return {
            "emits": [
                {
                    "event": "task.result",
                    "payload": {
                        "detail": "SQL agent is not configured. Set SQL_AGENT_DSN or SQL_DATABASE_URL.",
                        "status": "failed",
                        "error": "SQL agent is not configured. Set SQL_AGENT_DSN or SQL_DATABASE_URL.",
                        "result": None,
                    },
                }
            ]
        }

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
            if operation == "inspect_schema" or (_is_schema_request(classification_task) and not provided_sql):
                stats["total_ms"] = _elapsed_ms(total_started)
                return {
                    "emits": [
                        {
                            "event": "sql.result",
                            "payload": {
                                "detail": "Database schema introspected.",
                                "schema": schema,
                                "stats": stats,
                                "result": schema,
                            },
                        }
                    ]
                }
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
            if isinstance(provided_sql, str) and provided_sql.strip():
                query_specs = [{"label": "provided SQL", "sql": provided_sql.strip()}]
                stats["sql_generation_ms"] = 0
            else:
                started = time.perf_counter()
                query_task = instruction.get("question") if isinstance(instruction, dict) and isinstance(instruction.get("question"), str) else execution_task
                query_specs = _llm_sql_queries(query_task, schema)
                stats["sql_generation_ms"] = _elapsed_ms(started)
            if not query_specs:
                stats["total_ms"] = _elapsed_ms(total_started)
                return {
                    "emits": [
                        {
                            "event": "task.result",
                            "payload": {
                                "detail": "Could not generate a SQL query.",
                                "status": "needs_decomposition",
                                "error": "Could not generate a SQL query.",
                                "result": {"ok": False, "stats": stats},
                                "replan_hint": {
                                    "reason": "Could not generate a SQL query.",
                                    "failure_class": "needs_decomposition",
                                    "suggested_capabilities": ["sql_runner"],
                                },
                            },
                        }
                    ]
                }
            started = time.perf_counter()
            query_task = instruction.get("question") if isinstance(instruction, dict) and isinstance(instruction.get("question"), str) else execution_task
            result, query_specs = _repair_sql_with_retries(conn, schema, query_task, query_specs, _row_limit(), stats)
            stats["query_ms"] = _elapsed_ms(started)
            stats["total_ms"] = _elapsed_ms(total_started)
            sql = result.get("sql", "")
            return {
                "emits": [
                    {
                        "event": "sql.result",
                        "payload": {
                            "detail": "SQL query executed.",
                            "sql": sql,
                            "schema": schema,
                            "stats": stats,
                            "result": result,
                        },
                    }
                ]
            }
    except Exception as exc:
        stats["total_ms"] = _elapsed_ms(total_started)
        _debug_log(f"SQL task failed: {type(exc).__name__}: {exc}")
        return {
            "emits": [
                {
                    "event": "task.result",
                    "payload": {
                        "detail": f"SQL task failed: {type(exc).__name__}: {exc}",
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "result": {"ok": False, "stats": stats} if stats else None,
                    },
                }
            ]
        }
