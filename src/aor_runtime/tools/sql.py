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
from aor_runtime.runtime.sql_catalog import SqlColumnRef, SqlSchemaCatalog, SqlTableRef
from aor_runtime.runtime.sql_safety import (
    ensure_read_only_sql,
    normalize_pg_relation_quoting,
    validate_read_only_sql,
)
from aor_runtime.tools.base import BaseTool, ToolArgsModel, ToolExecutionError, ToolResultModel


TOKEN_RE = re.compile(r"[a-z0-9_]+")
SYSTEM_SCHEMAS = {"information_schema", "pg_catalog"}
_SCHEMA_CATALOG_CACHE: dict[tuple[str, str], SqlSchemaCatalog] = {}


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")


def _tokenize(value: str) -> set[str]:
    normalized = _normalize_name(value)
    return set(TOKEN_RE.findall(normalized))


def _coerce_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def validate_safe_query(query: str) -> str:
    try:
        return ensure_read_only_sql(query)
    except ValueError as exc:
        raise ToolExecutionError(str(exc)) from exc


_validate_safe_query = validate_safe_query


class ColumnSchema(ToolResultModel):
    name: str
    type: str


class TableSchema(ToolResultModel):
    name: str
    columns: list[ColumnSchema]


class DatabaseSchema(ToolResultModel):
    name: str
    dialect: str | None = None
    tables: list[TableSchema]
    error: str | None = None


class SchemaInfo(ToolResultModel):
    databases: list[DatabaseSchema]


def resolve_sql_databases(settings: Settings) -> dict[str, str]:
    mapping: dict[str, str] = {}

    if settings.sql_databases:
        payload = dict(settings.sql_databases)
        for name, value in payload.items():
            normalized_name = _normalize_name(str(name))
            normalized_url = str(value or "").strip()
            if not normalized_name:
                raise ToolExecutionError("SQL database config contains an empty database name.")
            if not normalized_url:
                raise ToolExecutionError(f"SQL database URL is empty for database {name!r}.")
            mapping[normalized_name] = normalized_url
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
        raise ToolExecutionError("SQL tool is not configured. Add sql.database_url or sql.databases to config.yaml.")

    if requested_database is not None and str(requested_database).strip():
        normalized_name = _normalize_name(str(requested_database))
        if normalized_name not in databases:
            raise ToolExecutionError(f"Unknown database {requested_database!r}.")
        return normalized_name, databases[normalized_name]

    if len(databases) == 1:
        only_name = next(iter(databases))
        return only_name, databases[only_name]

    default_name = resolve_default_database(settings, databases)
    if default_name:
        return default_name, databases[default_name]
    raise ToolExecutionError("Database selection is required when multiple SQL databases are configured.")


@lru_cache(maxsize=32)
def _engine_for_url(database_url: str) -> Engine:
    return create_engine(database_url, future=True, pool_pre_ping=True)


def _is_system_schema(schema_name: str) -> bool:
    lowered = str(schema_name or "").lower()
    return lowered in SYSTEM_SCHEMAS or lowered.startswith("pg_toast") or lowered.startswith("pg_temp")


def refresh_schema_cache(settings: Settings | None = None, database: str | None = None) -> None:
    configured = settings or get_settings()
    if database:
        database_name, database_url = resolve_database_selection(configured, database)
        _SCHEMA_CATALOG_CACHE.pop((database_name, database_url), None)
        return
    for item in list(_SCHEMA_CATALOG_CACHE):
        _SCHEMA_CATALOG_CACHE.pop(item, None)


def get_sql_catalog(settings: Settings | None = None, database: str | None = None, *, refresh: bool = False) -> SqlSchemaCatalog:
    configured = settings or get_settings()
    database_name, database_url = resolve_database_selection(configured, database)
    cache_key = (database_name, database_url)
    if refresh:
        _SCHEMA_CATALOG_CACHE.pop(cache_key, None)
    cached = _SCHEMA_CATALOG_CACHE.get(cache_key)
    if cached is not None:
        return cached.model_copy(deep=True)

    engine = _engine_for_url(database_url)
    catalog = _inspect_catalog_for_engine(engine, database_name)
    _SCHEMA_CATALOG_CACHE[cache_key] = catalog
    return catalog.model_copy(deep=True)


def get_all_sql_catalogs(settings: Settings | None = None, *, refresh: bool = False) -> list[SqlSchemaCatalog]:
    configured = settings or get_settings()
    catalogs: list[SqlSchemaCatalog] = []
    for database_name, database_url in sorted(resolve_sql_databases(configured).items()):
        try:
            if refresh:
                _SCHEMA_CATALOG_CACHE.pop((database_name, database_url), None)
            catalogs.append(get_sql_catalog(configured, database_name))
        except Exception as exc:  # noqa: BLE001
            dialect: str | None = None
            try:
                dialect = _engine_for_url(database_url).dialect.name
            except Exception:  # noqa: BLE001
                dialect = None
            catalogs.append(SqlSchemaCatalog(database=database_name, dialect=dialect, tables=[], error=str(exc)))
    return catalogs


def _inspect_catalog_for_engine(engine: Engine, database_name: str) -> SqlSchemaCatalog:
    inspector = inspect(engine)
    tables: list[SqlTableRef] = []

    if engine.dialect.name == "postgresql":
        schema_names = [name for name in inspector.get_schema_names() if not _is_system_schema(name)]
        for schema_name in schema_names:
            for table_name in [*inspector.get_table_names(schema=schema_name), *inspector.get_view_names(schema=schema_name)]:
                tables.append(_inspect_table_ref(inspector, schema_name=schema_name, table_name=table_name))
    else:
        for table_name in inspector.get_table_names():
            if table_name.startswith("sqlite_"):
                continue
            tables.append(_inspect_table_ref(inspector, schema_name="main", table_name=table_name))

    tables.sort(key=lambda item: item.qualified_name.lower())
    return SqlSchemaCatalog(database=database_name, dialect=engine.dialect.name, tables=tables)


def _inspect_table_ref(inspector: Any, *, schema_name: str, table_name: str) -> SqlTableRef:
    try:
        raw_pk = inspector.get_pk_constraint(table_name, schema=schema_name if schema_name != "main" else None) or {}
    except Exception:  # noqa: BLE001
        raw_pk = {}
    pk_columns = [str(column) for column in raw_pk.get("constrained_columns", []) if str(column)]
    pk_set = set(pk_columns)
    try:
        raw_fks = inspector.get_foreign_keys(table_name, schema=schema_name if schema_name != "main" else None) or []
    except Exception:  # noqa: BLE001
        raw_fks = []
    foreign_by_column: dict[str, str] = {}
    foreign_keys: list[dict[str, Any]] = []
    for raw_fk in raw_fks:
        constrained = [str(column) for column in raw_fk.get("constrained_columns", []) if str(column)]
        referred_schema = str(raw_fk.get("referred_schema") or schema_name)
        referred_table = str(raw_fk.get("referred_table") or "")
        referred_columns = [str(column) for column in raw_fk.get("referred_columns", []) if str(column)]
        for local, remote in zip(constrained, referred_columns, strict=False):
            foreign_by_column[local] = f"{referred_schema}.{referred_table}.{remote}"
        foreign_keys.append(
            {
                "columns": constrained,
                "referred_schema": referred_schema,
                "referred_table": referred_table,
                "referred_columns": referred_columns,
            }
        )

    try:
        raw_columns = inspector.get_columns(table_name, schema=schema_name if schema_name != "main" else None)
    except Exception:
        raw_columns = []
    columns: list[SqlColumnRef] = []
    for ordinal, column in enumerate(raw_columns, start=1):
        column_name = str(column.get("name") or "")
        if not column_name:
            continue
        columns.append(
            SqlColumnRef(
                schema_name=schema_name,
                table_name=table_name,
                column_name=column_name,
                data_type=str(column.get("type", "")) or None,
                ordinal_position=int(column.get("ordinal_position") or ordinal),
                nullable=column.get("nullable") if isinstance(column.get("nullable"), bool) else None,
                primary_key=column_name in pk_set,
                foreign_key=foreign_by_column.get(column_name),
            )
        )
    return SqlTableRef(
        schema_name=schema_name,
        table_name=table_name,
        columns=columns,
        primary_key_columns=pk_columns,
        foreign_keys=foreign_keys,
    )


def _catalog_table_to_schema(table: SqlTableRef, dialect: str | None) -> TableSchema:
    display_name = table.table_name if table.schema_name in {"main", "public"} and dialect != "postgresql" else table.qualified_name
    if dialect == "postgresql":
        display_name = table.qualified_name
    return TableSchema(
        name=display_name,
        columns=[ColumnSchema(name=column.column_name, type=column.data_type or "") for column in table.columns],
    )


def get_schema(settings: Settings | None = None) -> SchemaInfo:
    configured = settings or get_settings()
    if not resolve_sql_databases(configured):
        return SchemaInfo(databases=[])

    discovered: list[DatabaseSchema] = []
    for catalog in get_all_sql_catalogs(configured):
        tables = [_catalog_table_to_schema(table, catalog.dialect) for table in catalog.tables]
        discovered.append(DatabaseSchema(name=catalog.database, dialect=catalog.dialect, tables=tables, error=catalog.error))
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
        pruned_databases.append(
            DatabaseSchema(name=database.name, dialect=database.dialect, tables=tables, error=database.error)
        )

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
    database_name, database_url = resolve_database_selection(settings, database)
    safe_query = _normalize_and_validate_query(settings, database_name, query)
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


def explain_sql_query(settings: Settings, query: str, database: str | None = None) -> None:
    database_name, database_url = resolve_database_selection(settings, database)
    safe_query = _normalize_and_validate_query(settings, database_name, query)
    engine = _engine_for_url(database_url)
    if engine.dialect.name != "postgresql":
        return
    with engine.connect() as connection:
        connection.execute(text(f"EXPLAIN {safe_query}"))


def _normalize_and_validate_query(settings: Settings, database_name: str, query: str) -> str:
    safe_query = _validate_safe_query(query)
    try:
        catalog = get_sql_catalog(settings, database_name)
    except Exception:  # noqa: BLE001
        catalog = None
    if catalog is not None and catalog.dialect == "postgresql":
        safe_query = normalize_pg_relation_quoting(safe_query, catalog)
    validation = validate_read_only_sql(safe_query)
    if not validation.valid:
        raise ToolExecutionError(validation.reason or "SQL query failed read-only validation.")
    return str(validation.normalized_sql or safe_query)


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


class SQLSchemaTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        database: str | None = None
        refresh: bool = False

    class ToolResult(ToolResultModel):
        catalog: dict[str, Any]

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="sql.schema",
            description="Return schema, table, and column metadata for a configured SQL database.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "database": {"type": ["string", "null"]},
                    "refresh": {"type": "boolean"},
                },
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        catalog = get_sql_catalog(self.settings, arguments.database, refresh=arguments.refresh)
        return self.ToolResult.model_validate({"catalog": catalog.model_dump()})
