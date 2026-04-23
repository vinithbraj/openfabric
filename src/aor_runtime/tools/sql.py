from __future__ import annotations

import json
import multiprocessing as mp
import re
from functools import lru_cache
from typing import Any

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import ToolSpec
from aor_runtime.tools.base import BaseTool, ToolArgsModel, ToolExecutionError, ToolResultModel


UNSAFE_SQL_RE = re.compile(
    r"\b(drop|delete|alter|insert|update|truncate|create|grant|revoke|attach|detach|replace)\b",
    re.IGNORECASE,
)
READ_ONLY_SQL_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")


def _tokenize(value: str) -> set[str]:
    normalized = _normalize_name(value)
    return set(TOKEN_RE.findall(normalized))


def _coerce_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _validate_safe_query(query: str) -> str:
    normalized = str(query or "").strip()
    if not normalized:
        raise ToolExecutionError("Empty SQL query.")
    if ";" in normalized.rstrip(";"):
        raise ToolExecutionError("Multiple SQL statements are not allowed.")
    if UNSAFE_SQL_RE.search(normalized):
        raise ToolExecutionError("Unsafe query")
    if not READ_ONLY_SQL_RE.match(normalized):
        raise ToolExecutionError("Only SELECT and WITH queries are allowed.")
    return normalized.rstrip(";")


class ColumnSchema(ToolResultModel):
    name: str
    type: str


class TableSchema(ToolResultModel):
    name: str
    columns: list[ColumnSchema]


class DatabaseSchema(ToolResultModel):
    name: str
    tables: list[TableSchema]
    error: str | None = None


class SchemaInfo(ToolResultModel):
    databases: list[DatabaseSchema]


def resolve_sql_databases(settings: Settings) -> dict[str, str]:
    mapping: dict[str, str] = {}

    if settings.sql_databases_json:
        try:
            payload = json.loads(settings.sql_databases_json)
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(f"Invalid AOR_SQL_DATABASES JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ToolExecutionError("AOR_SQL_DATABASES must be a JSON object mapping names to URLs.")
        for name, value in payload.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise ToolExecutionError("AOR_SQL_DATABASES entries must map string names to string URLs.")
            normalized_name = _normalize_name(name)
            if not normalized_name:
                raise ToolExecutionError("AOR_SQL_DATABASES contains an empty database name.")
            mapping[normalized_name] = value.strip()
        return mapping

    if settings.sql_database_url:
        return {"default_db": settings.sql_database_url.strip()}

    return {}


def resolve_default_database(settings: Settings, databases: dict[str, str]) -> str | None:
    if not databases:
        return None
    if settings.sql_default_database:
        requested = _normalize_name(settings.sql_default_database)
        if requested not in databases:
            raise ToolExecutionError(f"Configured default database {settings.sql_default_database!r} is not defined.")
        return requested
    if len(databases) == 1:
        return next(iter(databases))
    return None


def resolve_database_selection(settings: Settings, requested_database: str | None) -> tuple[str, str]:
    databases = resolve_sql_databases(settings)
    if not databases:
        raise ToolExecutionError("SQL tool is not configured. Set AOR_SQL_DATABASES or AOR_SQL_DATABASE_URL.")

    if requested_database is not None and str(requested_database).strip():
        normalized_name = _normalize_name(str(requested_database))
        if normalized_name not in databases:
            raise ToolExecutionError(f"Unknown database {requested_database!r}.")
        return normalized_name, databases[normalized_name]

    if len(databases) == 1:
        only_name = next(iter(databases))
        return only_name, databases[only_name]

    default_name = resolve_default_database(settings, databases)
    if default_name and len(databases) == 1:
        return default_name, databases[default_name]
    raise ToolExecutionError("Database selection is required when multiple SQL databases are configured.")


@lru_cache(maxsize=32)
def _engine_for_url(database_url: str) -> Engine:
    return create_engine(database_url, future=True, pool_pre_ping=True)


def _inspect_tables_for_engine(engine: Engine) -> list[TableSchema]:
    inspector = inspect(engine)
    tables: list[TableSchema] = []

    if engine.dialect.name == "postgresql":
        schema_names = [name for name in inspector.get_schema_names() if name not in {"information_schema", "pg_catalog"}]
        for schema_name in schema_names:
            for table_name in inspector.get_table_names(schema=schema_name):
                columns = [
                    ColumnSchema(name=str(column["name"]), type=str(column.get("type", "")))
                    for column in inspector.get_columns(table_name, schema=schema_name)
                ]
                display_name = table_name if schema_name == "public" else f"{schema_name}.{table_name}"
                tables.append(TableSchema(name=display_name, columns=columns))
    else:
        for table_name in inspector.get_table_names():
            if table_name.startswith("sqlite_"):
                continue
            columns = [
                ColumnSchema(name=str(column["name"]), type=str(column.get("type", "")))
                for column in inspector.get_columns(table_name)
            ]
            tables.append(TableSchema(name=table_name, columns=columns))

    tables.sort(key=lambda item: item.name)
    return tables


def get_schema(settings: Settings | None = None) -> SchemaInfo:
    configured = settings or get_settings()
    databases = resolve_sql_databases(configured)
    if not databases:
        return SchemaInfo(databases=[])

    discovered: list[DatabaseSchema] = []
    for database_name, database_url in sorted(databases.items()):
        try:
            engine = _engine_for_url(database_url)
            discovered.append(DatabaseSchema(name=database_name, tables=_inspect_tables_for_engine(engine)))
        except Exception as exc:  # noqa: BLE001
            discovered.append(DatabaseSchema(name=database_name, tables=[], error=str(exc)))
    return SchemaInfo(databases=discovered)


def _database_score(database: DatabaseSchema, goal_tokens: set[str], goal_text: str, default_database: str | None) -> int:
    score = 0
    name_tokens = _tokenize(database.name)
    if database.name.lower() in goal_text:
        score += 10
    score += len(goal_tokens & name_tokens) * 4
    if default_database and database.name == default_database:
        score += 1
    return score


def _table_score(table: TableSchema, goal_tokens: set[str]) -> int:
    score = len(goal_tokens & _tokenize(table.name)) * 6
    for column in table.columns:
        overlap = len(goal_tokens & _tokenize(column.name))
        if overlap:
            score += overlap * 2
    return score


def prune_schema(
    schema: SchemaInfo,
    goal: str,
    *,
    settings: Settings | None = None,
    max_databases: int = 3,
    max_tables_per_database: int = 4,
) -> SchemaInfo:
    if not schema.databases:
        return schema

    configured = settings or get_settings()
    goal_text = str(goal or "").lower()
    goal_tokens = _tokenize(goal_text)
    default_database = None
    try:
        default_database = resolve_default_database(configured, resolve_sql_databases(configured))
    except Exception:  # noqa: BLE001
        default_database = None

    scored_databases: list[tuple[int, DatabaseSchema]] = [
        (_database_score(database, goal_tokens, goal_text, default_database), database) for database in schema.databases
    ]
    positive = [item for item in scored_databases if item[0] > 0]
    if positive:
        positive.sort(key=lambda item: (-item[0], item[1].name))
        selected_databases = [database for _, database in positive[:max_databases]]
    else:
        selected_databases = sorted(schema.databases, key=lambda item: item.name)[:max_databases]

    pruned_databases: list[DatabaseSchema] = []
    for database in selected_databases:
        if database.error:
            pruned_databases.append(database)
            continue
        scored_tables = [(_table_score(table, goal_tokens), table) for table in database.tables]
        positive_tables = [item for item in scored_tables if item[0] > 0]
        if positive_tables:
            positive_tables.sort(key=lambda item: (-item[0], item[1].name))
            tables = [table for _, table in positive_tables[:max_tables_per_database]]
        else:
            tables = sorted(database.tables, key=lambda item: item.name)[:max_tables_per_database]
        pruned_databases.append(DatabaseSchema(name=database.name, tables=tables, error=database.error))

    return SchemaInfo(databases=pruned_databases)


def _sql_query_worker(queue: mp.Queue, database_url: str, query: str, row_limit: int) -> None:
    try:
        engine = create_engine(database_url, future=True, pool_pre_ping=True)
        with engine.connect() as connection:
            result = connection.execute(text(query))
            if not result.returns_rows:
                raise ToolExecutionError("SQL query did not return rows.")
            rows: list[dict[str, Any]] = []
            for index, row in enumerate(result):
                if index >= row_limit:
                    break
                rows.append({str(key): _coerce_value(value) for key, value in row._mapping.items()})
        queue.put({"ok": True, "rows": rows, "row_count": len(rows)})
    except Exception as exc:  # noqa: BLE001
        queue.put({"ok": False, "error": str(exc)})


def sql_query(settings: Settings, query: str, database: str | None = None) -> dict[str, Any]:
    safe_query = _validate_safe_query(query)
    database_name, database_url = resolve_database_selection(settings, database)
    queue: mp.Queue = mp.Queue()
    process = mp.Process(
        target=_sql_query_worker,
        args=(queue, database_url, safe_query, max(1, int(settings.sql_row_limit))),
    )
    process.start()
    process.join(max(1, int(settings.sql_timeout_seconds)))
    if process.is_alive():
        process.terminate()
        process.join()
        raise ToolExecutionError(f"SQL query timed out after {settings.sql_timeout_seconds} seconds.")
    if queue.empty():
        raise ToolExecutionError("SQL query did not return a result.")
    payload = queue.get()
    if not payload.get("ok"):
        raise ToolExecutionError(str(payload.get("error") or "SQL query failed."))
    rows = payload.get("rows", [])
    row_count = payload.get("row_count", len(rows) if isinstance(rows, list) else 0)
    return {"database": database_name, "rows": rows, "row_count": row_count}


class SQLQueryTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        database: str | None = None
        query: str

    class ToolResult(ToolResultModel):
        database: str
        rows: list[dict[str, Any]]
        row_count: int

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="sql.query",
            description="Execute a read-only SQL query against a configured named database.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "database": {"type": ["string", "null"]},
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult.model_validate(sql_query(self.settings, query=arguments.query, database=arguments.database))
