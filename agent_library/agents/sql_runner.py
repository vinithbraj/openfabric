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

from agent_library.common import EventRequest, EventResponse
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
    lines = [f"Dialect: {schema.get('dialect', 'unknown')}"]
    for table in schema.get("tables", []):
        columns = ", ".join(f"{col['name']} {col.get('type', '')}".strip() for col in table.get("columns", []))
        lines.append(f"- {table.get('schema')}.{table.get('name')}: {columns}")
        for fk in table.get("foreign_keys", []):
            lines.append(
                "  FK {column} -> {schema}.{table}.{column_ref}".format(
                    column=fk.get("column"),
                    schema=fk.get("references_schema"),
                    table=fk.get("references_table"),
                    column_ref=fk.get("references_column"),
                )
            )
    return "\n".join(lines)


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
        "- Use foreign-key relationships from the schema for joins.\n"
        "- Prefer SELECT or WITH queries. Do not generate mutating SQL.\n"
        "- Add a reasonable LIMIT unless the user asks for aggregation only.\n"
        "- Use the SQL dialect from the schema.\n"
        "Schema:\n"
        f"{_schema_summary(schema)}\n"
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


def _read_only_sql(sql: str) -> bool:
    stripped = sql.strip().rstrip(";").strip()
    if ";" in stripped:
        return False
    if MUTATING_PATTERN.search(stripped):
        return False
    return stripped.lower().startswith(READ_ONLY_PREFIXES)


def _execute_sql(conn, sql: str, limit: int):
    if not _read_only_sql(sql):
        raise RuntimeError("Rejected non-read-only SQL.")
    with closing(conn.cursor()) as cursor:
        cursor.execute(sql)
        columns, rows = _rows_from_cursor(cursor, limit)
    return {"sql": sql, "columns": columns, "rows": rows, "row_count": len(rows), "limit": limit}


def _execute_sql_queries(conn, query_specs: list[dict[str, str]], limit: int):
    if len(query_specs) == 1:
        return _execute_sql(conn, query_specs[0]["sql"], limit)

    query_results = []
    rows = []
    scalar_rows = True
    for index, spec in enumerate(query_specs, start=1):
        label = spec.get("label") or f"query {index}"
        result = _execute_sql(conn, spec["sql"], limit)
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


def _is_schema_request(task: str) -> bool:
    task_lc = task.lower()
    return any(token in task_lc for token in ("schema", "schemas", "tables", "columns", "relationships", "foreign key", "foreign keys", "describe database"))


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


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event == "sql.query":
        task = str(req.payload.get("question") or req.payload.get("query") or "")
        provided_sql = req.payload.get("sql")
    elif req.event == "task.plan":
        task = req.payload.get("task", "")
        original_task = req.payload.get("original_task")
        if isinstance(original_task, str) and original_task.strip():
            task = f"{task}\nOriginal user request: {original_task.strip()}"
        target_agent = req.payload.get("target_agent")
        explicitly_targeted = isinstance(target_agent, str) and target_agent.startswith("sql_runner")
        if not isinstance(task, str) or (not explicitly_targeted and not _is_sql_task(task)):
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
            if _is_schema_request(task) and not provided_sql:
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
            if isinstance(provided_sql, str) and provided_sql.strip():
                query_specs = [{"label": "provided SQL", "sql": provided_sql.strip()}]
                stats["sql_generation_ms"] = 0
            else:
                started = time.perf_counter()
                query_specs = _llm_sql_queries(task, schema)
                stats["sql_generation_ms"] = _elapsed_ms(started)
            if not query_specs:
                stats["total_ms"] = _elapsed_ms(total_started)
                return {
                    "emits": [
                        {
                            "event": "task.result",
                            "payload": {
                                "detail": "Could not generate a SQL query.",
                                "status": "failed",
                                "error": "Could not generate a SQL query.",
                                "result": {"ok": False, "stats": stats},
                            },
                        }
                    ]
                }
            started = time.perf_counter()
            result = _execute_sql_queries(conn, query_specs, _row_limit())
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
