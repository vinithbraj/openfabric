"""OpenFABRIC Runtime Module: aor_runtime.tools.sql

Purpose:
    Implement read-only SQL schema, query, and validation tools.

Responsibilities:
    Execute validated SQL through bounded workers, expose schema catalogs, and return structured row/validation results.

Data flow / Interfaces:
    Receives tool arguments from PlanExecutor and Settings-defined database connections.

Boundaries:
    Enforces read-only SQL, timeout/lifecycle controls, and result-shape contracts before data reaches presentation.
"""

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
    validate_read_only_sql,
)
from aor_runtime.runtime.sql_ast_validation import normalize_and_validate_sql_ast
from aor_runtime.runtime.lifecycle import CancellationError, ToolInvocationContext, run_process_with_queue
from aor_runtime.tools.base import BaseTool, ToolArgsModel, ToolExecutionError, ToolResultModel


TOKEN_RE = re.compile(r"[a-z0-9_]+")
SYSTEM_SCHEMAS = {"information_schema", "pg_catalog"}
_SCHEMA_CATALOG_CACHE: dict[tuple[str, str], SqlSchemaCatalog] = {}


def _normalize_name(value: str) -> str:
    """Handle the internal normalize name helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql._normalize_name.
    """
    return re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")


def _tokenize(value: str) -> set[str]:
    """Handle the internal tokenize helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql._tokenize.
    """
    normalized = _normalize_name(value)
    return set(TOKEN_RE.findall(normalized))


def _coerce_value(value: Any) -> Any:
    """Handle the internal coerce value helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql._coerce_value.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def validate_safe_query(query: str) -> str:
    """Validate safe query for the surrounding runtime workflow.

    Inputs:
        Receives query for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql.validate_safe_query.
    """
    try:
        return ensure_read_only_sql(query)
    except ValueError as exc:
        raise ToolExecutionError(str(exc)) from exc


_validate_safe_query = validate_safe_query


class ColumnSchema(ToolResultModel):
    """Represent column schema within the OpenFABRIC runtime. It extends ToolResultModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ColumnSchema.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.sql.ColumnSchema and related tests.
    """
    name: str
    type: str


class TableSchema(ToolResultModel):
    """Represent table schema within the OpenFABRIC runtime. It extends ToolResultModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by TableSchema.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.sql.TableSchema and related tests.
    """
    name: str
    columns: list[ColumnSchema]


class DatabaseSchema(ToolResultModel):
    """Represent database schema within the OpenFABRIC runtime. It extends ToolResultModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by DatabaseSchema.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.sql.DatabaseSchema and related tests.
    """
    name: str
    dialect: str | None = None
    tables: list[TableSchema]
    error: str | None = None


class SchemaInfo(ToolResultModel):
    """Represent schema info within the OpenFABRIC runtime. It extends ToolResultModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SchemaInfo.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.sql.SchemaInfo and related tests.
    """
    databases: list[DatabaseSchema]


def resolve_sql_databases(settings: Settings) -> dict[str, str]:
    """Resolve sql databases for the surrounding runtime workflow.

    Inputs:
        Receives settings for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql.resolve_sql_databases.
    """
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
    """Resolve default database for the surrounding runtime workflow.

    Inputs:
        Receives settings, databases for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql.resolve_default_database.
    """
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
    """Resolve database selection for the surrounding runtime workflow.

    Inputs:
        Receives settings, requested_database for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql.resolve_database_selection.
    """
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
    """Handle the internal engine for url helper path for this module.

    Inputs:
        Receives database_url for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql._engine_for_url.
    """
    return create_engine(database_url, future=True, pool_pre_ping=True)


def _is_system_schema(schema_name: str) -> bool:
    """Handle the internal is system schema helper path for this module.

    Inputs:
        Receives schema_name for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql._is_system_schema.
    """
    lowered = str(schema_name or "").lower()
    return lowered in SYSTEM_SCHEMAS or lowered.startswith("pg_toast") or lowered.startswith("pg_temp")


def refresh_schema_cache(settings: Settings | None = None, database: str | None = None) -> None:
    """Refresh schema cache for the surrounding runtime workflow.

    Inputs:
        Receives settings, database for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql.refresh_schema_cache.
    """
    configured = settings or get_settings()
    if database:
        database_name, database_url = resolve_database_selection(configured, database)
        _SCHEMA_CATALOG_CACHE.pop((database_name, database_url), None)
        return
    for item in list(_SCHEMA_CATALOG_CACHE):
        _SCHEMA_CATALOG_CACHE.pop(item, None)


def get_sql_catalog(settings: Settings | None = None, database: str | None = None, *, refresh: bool = False) -> SqlSchemaCatalog:
    """Get sql catalog for the surrounding runtime workflow.

    Inputs:
        Receives settings, database, refresh for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql.get_sql_catalog.
    """
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
    """Get all sql catalogs for the surrounding runtime workflow.

    Inputs:
        Receives settings, refresh for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql.get_all_sql_catalogs.
    """
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
    """Handle the internal inspect catalog for engine helper path for this module.

    Inputs:
        Receives engine, database_name for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql._inspect_catalog_for_engine.
    """
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
    """Handle the internal inspect table ref helper path for this module.

    Inputs:
        Receives inspector, schema_name, table_name for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql._inspect_table_ref.
    """
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
    """Handle the internal catalog table to schema helper path for this module.

    Inputs:
        Receives table, dialect for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql._catalog_table_to_schema.
    """
    display_name = table.table_name if table.schema_name in {"main", "public"} and dialect != "postgresql" else table.qualified_name
    if dialect == "postgresql":
        display_name = table.qualified_name
    return TableSchema(
        name=display_name,
        columns=[ColumnSchema(name=column.column_name, type=column.data_type or "") for column in table.columns],
    )


def get_schema(settings: Settings | None = None) -> SchemaInfo:
    """Get schema for the surrounding runtime workflow.

    Inputs:
        Receives settings for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql.get_schema.
    """
    configured = settings or get_settings()
    if not resolve_sql_databases(configured):
        return SchemaInfo(databases=[])

    discovered: list[DatabaseSchema] = []
    for catalog in get_all_sql_catalogs(configured):
        tables = [_catalog_table_to_schema(table, catalog.dialect) for table in catalog.tables]
        discovered.append(DatabaseSchema(name=catalog.database, dialect=catalog.dialect, tables=tables, error=catalog.error))
    return SchemaInfo(databases=discovered)


def _database_score(database: DatabaseSchema, goal_tokens: set[str], goal_text: str, default_database: str | None) -> int:
    """Handle the internal database score helper path for this module.

    Inputs:
        Receives database, goal_tokens, goal_text, default_database for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql._database_score.
    """
    score = 0
    name_tokens = _tokenize(database.name)
    if database.name.lower() in goal_text:
        score += 10
    score += len(goal_tokens & name_tokens) * 4
    if default_database and database.name == default_database:
        score += 1
    return score


def _table_score(table: TableSchema, goal_tokens: set[str]) -> int:
    """Handle the internal table score helper path for this module.

    Inputs:
        Receives table, goal_tokens for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql._table_score.
    """
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
    """Prune schema for the surrounding runtime workflow.

    Inputs:
        Receives schema, goal, settings, max_databases, max_tables_per_database for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql.prune_schema.
    """
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


def _sql_query_worker(queue: mp.Queue, database_url: str, query: str, row_limit: int | None = None) -> None:
    """Handle the internal sql query worker helper path for this module.

    Inputs:
        Receives queue, database_url, query, row_limit for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql._sql_query_worker.
    """
    try:
        engine = create_engine(database_url, future=True, pool_pre_ping=True)
        with engine.connect() as connection:
            result = connection.execute(text(query))
            if not result.returns_rows:
                raise ToolExecutionError("SQL query did not return rows.")
            rows: list[dict[str, Any]] = []
            truncated = False
            for index, row in enumerate(result):
                if row_limit is not None and index >= row_limit:
                    truncated = True
                    break
                rows.append({str(key): _coerce_value(value) for key, value in row._mapping.items()})
        queue.put(
            {
                "ok": True,
                "rows": rows,
                "row_count": len(rows),
                "returned_count": len(rows),
                "limit": row_limit,
                "truncated": truncated,
            }
        )
    except Exception as exc:  # noqa: BLE001
        queue.put({"ok": False, "error": str(exc)})


def sql_query(settings: Settings, query: str, database: str | None = None) -> dict[str, Any]:
    """Sql query for the surrounding runtime workflow.

    Inputs:
        Receives settings, query, database for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql.sql_query.
    """
    return _sql_query(settings, query, database=database, context=None)


def _sql_query(
    settings: Settings,
    query: str,
    *,
    database: str | None = None,
    context: ToolInvocationContext | None = None,
) -> dict[str, Any]:
    """Handle the internal sql query helper path for this module.

    Inputs:
        Receives settings, query, database, context for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql._sql_query.
    """
    database_name, database_url = resolve_database_selection(settings, database)
    safe_query = _normalize_and_validate_query(settings, database_name, query)
    try:
        payload = run_process_with_queue(
            target=_sql_query_worker,
            # SQL reads intentionally do not impose a row cap. Large result presentation is handled
            # downstream by the auto-artifact policy, which writes full returned collections to files
            # and keeps only small results inline in the UI.
            args=(database_url, safe_query, None),
            timeout_seconds=max(1, int(settings.sql_timeout_seconds)),
            timeout_message=f"SQL query timed out after {settings.sql_timeout_seconds} seconds.",
            context=context,
            process_name="aor-sql-query",
        )
    except TimeoutError as exc:
        raise ToolExecutionError(str(exc)) from exc
    except RuntimeError as exc:
        if isinstance(exc, CancellationError):
            raise
        raise ToolExecutionError(str(exc)) from exc
    if not payload.get("ok"):
        raise ToolExecutionError(str(payload.get("error") or "SQL query failed."))
    rows = payload.get("rows", [])
    row_count = payload.get("row_count", len(rows) if isinstance(rows, list) else 0)
    return {
        "database": database_name,
        "rows": rows,
        "row_count": row_count,
        "returned_count": payload.get("returned_count", row_count),
        "limit": payload.get("limit"),
        "truncated": bool(payload.get("truncated")),
    }


def explain_sql_query(settings: Settings, query: str, database: str | None = None) -> None:
    """Explain sql query for the surrounding runtime workflow.

    Inputs:
        Receives settings, query, database for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql.explain_sql_query.
    """
    database_name, database_url = resolve_database_selection(settings, database)
    safe_query = _normalize_and_validate_query(settings, database_name, query)
    engine = _engine_for_url(database_url)
    if engine.dialect.name != "postgresql":
        return
    with engine.connect() as connection:
        connection.execute(text(f"EXPLAIN {safe_query}"))


def _normalize_and_validate_query(settings: Settings, database_name: str, query: str) -> str:
    """Handle the internal normalize and validate query helper path for this module.

    Inputs:
        Receives settings, database_name, query for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql._normalize_and_validate_query.
    """
    safe_query = _validate_safe_query(query)
    try:
        catalog = get_sql_catalog(settings, database_name)
    except Exception:  # noqa: BLE001
        catalog = None
    if catalog is not None and catalog.dialect == "postgresql":
        ast_validation = normalize_and_validate_sql_ast(safe_query, catalog)
        if not ast_validation.valid:
            raise ToolExecutionError("; ".join(ast_validation.messages) or "SQL failed catalog validation.")
        safe_query = ast_validation.normalized_sql
    validation = validate_read_only_sql(safe_query)
    if not validation.valid:
        raise ToolExecutionError(validation.reason or "SQL query failed read-only validation.")
    return str(validation.normalized_sql or safe_query)


class SQLQueryTool(BaseTool):
    """Represent s q l query tool within the OpenFABRIC runtime. It extends BaseTool.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SQLQueryTool.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.sql.SQLQueryTool and related tests.
    """
    class ToolArgs(ToolArgsModel):
        """Represent tool args within the OpenFABRIC runtime. It extends ToolArgsModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolArgs.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.sql.ToolArgs and related tests.
        """
        database: str | None = None
        query: str

    class ToolResult(ToolResultModel):
        """Represent tool result within the OpenFABRIC runtime. It extends ToolResultModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolResult.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.sql.ToolResult and related tests.
        """
        database: str
        rows: list[dict[str, Any]]
        row_count: int
        returned_count: int
        limit: int | None = None
        truncated: bool = False

    def __init__(self, settings: Settings | None = None) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings for this SQLQueryTool method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through SQLQueryTool.__init__ calls and related tests.
        """
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
        """Run for SQLQueryTool instances.

        Inputs:
            Receives arguments for this SQLQueryTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through SQLQueryTool.run calls and related tests.
        """
        return self.ToolResult.model_validate(sql_query(self.settings, query=arguments.query, database=arguments.database))

    def run_with_context(self, arguments: ToolArgs, context: ToolInvocationContext) -> ToolResult:
        """Run with context for SQLQueryTool instances.

        Inputs:
            Receives arguments, context for this SQLQueryTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through SQLQueryTool.run_with_context calls and related tests.
        """
        return self.ToolResult.model_validate(
            _sql_query(self.settings, arguments.query, database=arguments.database, context=context)
        )


class SQLSchemaTool(BaseTool):
    """Represent s q l schema tool within the OpenFABRIC runtime. It extends BaseTool.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SQLSchemaTool.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.sql.SQLSchemaTool and related tests.
    """
    class ToolArgs(ToolArgsModel):
        """Represent tool args within the OpenFABRIC runtime. It extends ToolArgsModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolArgs.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.sql.ToolArgs and related tests.
        """
        database: str | None = None
        refresh: bool = False

    class ToolResult(ToolResultModel):
        """Represent tool result within the OpenFABRIC runtime. It extends ToolResultModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolResult.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.sql.ToolResult and related tests.
        """
        catalog: dict[str, Any]

    def __init__(self, settings: Settings | None = None) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings for this SQLSchemaTool method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through SQLSchemaTool.__init__ calls and related tests.
        """
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
        """Run for SQLSchemaTool instances.

        Inputs:
            Receives arguments for this SQLSchemaTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through SQLSchemaTool.run calls and related tests.
        """
        catalog = get_sql_catalog(self.settings, arguments.database, refresh=arguments.refresh)
        return self.ToolResult.model_validate({"catalog": catalog.model_dump()})


class SQLValidateTool(BaseTool):
    """Represent s q l validate tool within the OpenFABRIC runtime. It extends BaseTool.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SQLValidateTool.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.sql.SQLValidateTool and related tests.
    """
    class ToolArgs(ToolArgsModel):
        """Represent tool args within the OpenFABRIC runtime. It extends ToolArgsModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolArgs.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.sql.ToolArgs and related tests.
        """
        database: str | None = None
        query: str

    class ToolResult(ToolResultModel):
        """Represent tool result within the OpenFABRIC runtime. It extends ToolResultModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolResult.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.sql.ToolResult and related tests.
        """
        database: str
        query: str
        valid: bool
        reason: str | None = None
        explanation: str

    def __init__(self, settings: Settings | None = None) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings for this SQLValidateTool method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through SQLValidateTool.__init__ calls and related tests.
        """
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="sql.validate",
            description="Validate read-only SQL against the configured catalog without executing the query.",
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
        """Run for SQLValidateTool instances.

        Inputs:
            Receives arguments for this SQLValidateTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through SQLValidateTool.run calls and related tests.
        """
        database_name, _database_url = resolve_database_selection(self.settings, arguments.database)
        safe_query = _normalize_and_validate_query(self.settings, database_name, arguments.query)
        validation = validate_read_only_sql(safe_query)
        valid = bool(validation.valid)
        reason = None if valid else validation.reason
        explanation = _explain_validated_sql(str(validation.normalized_sql or safe_query), valid=valid, reason=reason)
        return self.ToolResult.model_validate(
            {
                "database": database_name,
                "query": str(validation.normalized_sql or safe_query),
                "valid": valid,
                "reason": reason,
                "explanation": explanation,
            }
        )


def _explain_validated_sql(query: str, *, valid: bool, reason: str | None = None) -> str:
    """Handle the internal explain validated sql helper path for this module.

    Inputs:
        Receives query, valid, reason for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.sql._explain_validated_sql.
    """
    if not valid:
        return f"The SQL was not executed. Validation failed: {reason or 'unknown reason'}"
    summary = "The SQL was validated as a single read-only SELECT/WITH statement and was not executed."
    if re.search(r"(?is)\bgroup\s+by\b", query):
        summary += " It groups rows and returns aggregate rows for each group."
    elif re.search(r"(?is)\bcount\s*\(", query):
        summary += " It returns an aggregate count."
    elif re.search(r"(?is)\bmin\s*\(|\bmax\s*\(|\bavg\s*\(", query):
        summary += " It returns aggregate summary values."
    else:
        summary += " It returns rows selected by the query."
    return summary
