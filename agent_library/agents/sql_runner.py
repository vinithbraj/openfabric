import json
import os
import re
import sqlite3
import time
from contextlib import closing
from difflib import get_close_matches
import heapq
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
TABLE_REFERENCE_PATTERN = re.compile(
    r"""\b(?:from|join)\s+((?:"[^"]+"|`[^`]+`|[A-Za-z_][\w$]*)(?:\s*\.\s*(?:"[^"]+"|`[^`]+`|[A-Za-z_][\w$]*))?)""",
    re.IGNORECASE,
)
TABLE_ALIAS_PATTERN = re.compile(
    r"""\b(?:from|join)\s+((?:"[^"]+"|`[^`]+`|[A-Za-z_][\w$]*)(?:\s*\.\s*(?:"[^"]+"|`[^`]+`|[A-Za-z_][\w$]*))?)(?:\s+(?:as\s+)?([A-Za-z_][\w$]*))?""",
    re.IGNORECASE,
)
MISSING_RELATION_PATTERN = re.compile(r"""relation\s+"?([^"\s]+)"?\s+does\s+not\s+exist""", re.IGNORECASE)
UNQUOTED_QUALIFIED_IDENTIFIER_PATTERN = re.compile(
    r"""(?<!["`])\b([A-Za-z_][\w$]*)\s*\.\s*([A-Za-z_][\w$]*)\b(?!["`])"""
)
UNQUOTED_IDENTIFIER_PATTERN = re.compile(r"""(?<!["`])\b([A-Za-z_][\w$]*)\b(?!["`])""")
QUALIFIED_IDENTIFIER_REFERENCE_PATTERN = re.compile(
    r"""(?<!["`])\b([A-Za-z_][\w$]*)\s*\.\s*(?:"([^"]+)"|`([^`]+)`|([A-Za-z_][\w$]*))"""
)
SQL_KEYWORDS = {
    "all",
    "and",
    "as",
    "asc",
    "between",
    "by",
    "case",
    "count",
    "desc",
    "distinct",
    "else",
    "end",
    "exists",
    "false",
    "from",
    "group",
    "having",
    "in",
    "inner",
    "is",
    "join",
    "left",
    "like",
    "limit",
    "not",
    "null",
    "on",
    "or",
    "order",
    "outer",
    "right",
    "select",
    "sum",
    "then",
    "true",
    "union",
    "when",
    "where",
    "with",
}


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


def _unquote_identifier_part(part: str) -> str:
    text = part.strip()
    if len(text) >= 2 and ((text[0] == '"' and text[-1] == '"') or (text[0] == "`" and text[-1] == "`")):
        return text[1:-1]
    return text


def _normalize_identifier(identifier: str) -> str:
    parts = [_unquote_identifier_part(part) for part in re.split(r"\s*\.\s*", identifier.strip()) if part.strip()]
    return ".".join(parts)


def _schema_table_names(schema: dict) -> list[str]:
    names = []
    for table in schema.get("tables", []):
        schema_name = str(table.get("schema", "") or "").strip()
        table_name = str(table.get("name", "") or "").strip()
        if table_name:
            names.append(".".join(part for part in (schema_name, table_name) if part))
    return names


def _schema_table_lookup(schema: dict) -> dict[str, str]:
    lookup = {}
    bare_lookup: dict[str, str | None] = {}
    for name in _schema_table_names(schema):
        lookup[name.lower()] = name
        bare = name.split(".")[-1]
        bare_lc = bare.lower()
        if bare_lc not in bare_lookup:
            bare_lookup[bare_lc] = name
        elif bare_lookup[bare_lc] != name:
            bare_lookup[bare_lc] = None
    for bare_lc, qualified in bare_lookup.items():
        if qualified:
            lookup.setdefault(bare_lc, qualified)
    return lookup


def _schema_column_names(schema: dict) -> list[str]:
    names = []
    for table in schema.get("tables", []):
        schema_name = str(table.get("schema", "") or "").strip()
        table_name = str(table.get("name", "") or "").strip()
        qualified_table = ".".join(part for part in (schema_name, table_name) if part)
        for column in table.get("columns", []):
            column_name = str(column.get("name", "") or "").strip()
            if not column_name:
                continue
            names.append(column_name)
            if qualified_table:
                names.append(f"{qualified_table}.{column_name}")
    return names


def _schema_column_lookup(schema: dict) -> dict[str, str]:
    lookup = {}
    for name in _schema_column_names(schema):
        lookup.setdefault(name.lower(), name)
    return lookup


def _schema_columns_by_table(schema: dict) -> dict[str, set[str]]:
    by_table = {}
    for table in schema.get("tables", []):
        schema_name = str(table.get("schema", "") or "").strip()
        table_name = str(table.get("name", "") or "").strip()
        qualified_table = ".".join(part for part in (schema_name, table_name) if part)
        if not qualified_table:
            continue
        by_table[qualified_table] = {
            str(column.get("name", "") or "").strip()
            for column in table.get("columns", [])
            if str(column.get("name", "") or "").strip()
        }
    return by_table


def _schema_foreign_keys(schema: dict) -> list[dict[str, str]]:
    foreign_keys = []
    for table in schema.get("tables", []):
        schema_name = str(table.get("schema", "") or "").strip()
        table_name = str(table.get("name", "") or "").strip()
        source_table = ".".join(part for part in (schema_name, table_name) if part)
        if not source_table:
            continue
        for fk in table.get("foreign_keys", []):
            source_column = str(fk.get("column", "") or "").strip()
            target_table = ".".join(
                part
                for part in (
                    str(fk.get("references_schema", "") or "").strip(),
                    str(fk.get("references_table", "") or "").strip(),
                )
                if part
            )
            target_column = str(fk.get("references_column", "") or "").strip()
            if source_column and target_table and target_column:
                foreign_keys.append(
                    {
                        "source_table": source_table,
                        "source_column": source_column,
                        "target_table": target_table,
                        "target_column": target_column,
                    }
                )
    return foreign_keys


def _schema_table_metadata(schema: dict) -> dict[str, dict[str, Any]]:
    metadata = {}
    for table in schema.get("tables", []):
        schema_name = str(table.get("schema", "") or "").strip()
        table_name = str(table.get("name", "") or "").strip()
        qualified = ".".join(part for part in (schema_name, table_name) if part)
        if not qualified:
            continue
        metadata[qualified] = table
    return metadata


def _column_metadata(table_name: str, column_name: str, schema: dict) -> dict[str, Any] | None:
    metadata = _schema_table_metadata(schema).get(table_name)
    if not metadata:
        return None
    for column in metadata.get("columns", []):
        candidate = str(column.get("name", "") or "").strip()
        if candidate == column_name or candidate.lower() == column_name.lower():
            return column
    return None


def _is_numeric_column_type(column_type: str) -> bool:
    type_lc = (column_type or "").lower()
    return any(token in type_lc for token in ("int", "numeric", "decimal", "double", "real", "float"))


def _is_key_like_column(column_name: str) -> bool:
    name_lc = (column_name or "").lower()
    return name_lc.endswith("id") or name_lc.endswith("uid") or "instanceuid" in name_lc


def _best_identifier_column(table_name: str, schema: dict) -> str | None:
    metadata = _schema_table_metadata(schema).get(table_name)
    if not metadata:
        return None
    columns = [
        str(column.get("name", "") or "").strip()
        for column in metadata.get("columns", [])
        if str(column.get("name", "") or "").strip()
    ]
    preferred = [column for column in columns if _is_key_like_column(column)]
    if preferred:
        preferred.sort(key=lambda value: (value.lower().endswith("patientid"), value.lower().endswith("sopinstanceuid"), value.lower()), reverse=True)
        return preferred[0]
    return columns[0] if columns else None


def _schema_relationship_edges(schema: dict) -> list[dict[str, Any]]:
    edges = []
    seen = set()
    for edge in _schema_foreign_keys(schema):
        key = (
            edge["source_table"],
            edge["source_column"],
            edge["target_table"],
            edge["target_column"],
            "fk",
        )
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            {
                "left_table": edge["source_table"],
                "left_column": edge["source_column"],
                "right_table": edge["target_table"],
                "right_column": edge["target_column"],
                "kind": "fk",
                "weight": 1,
            }
        )

    columns_by_table = _schema_columns_by_table(schema)
    table_names = list(columns_by_table.keys())
    for index, left_table in enumerate(table_names):
        left_columns = columns_by_table[left_table]
        for right_table in table_names[index + 1 :]:
            right_columns = columns_by_table[right_table]
            shared = sorted(column for column in left_columns & right_columns if _is_key_like_column(column))
            for column in shared:
                key = tuple(sorted((left_table, right_table)) + [column, column, "shared_key"])
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    {
                        "left_table": left_table,
                        "left_column": column,
                        "right_table": right_table,
                        "right_column": column,
                        "kind": "shared_key",
                        "weight": 3,
                    }
                )

    for table_name, columns in columns_by_table.items():
        for column in columns:
            match = re.match(r"Referenced(.+?)(SOPInstanceUID|UID|ID)$", column)
            if not match:
                continue
            target_name = match.group(1)
            suffix = match.group(2)
            for candidate_table in table_names:
                bare = candidate_table.split(".")[-1]
                if bare.lower() != target_name.lower():
                    continue
                target_candidates = []
                if suffix == "SOPInstanceUID":
                    target_candidates.append("SOPInstanceUID")
                if suffix in {"UID", "SOPInstanceUID"}:
                    target_candidates.extend(["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"])
                if suffix == "ID":
                    target_candidates.extend([f"{bare}ID", "PatientID"])
                for target_column in target_candidates:
                    if target_column in columns_by_table[candidate_table]:
                        key = (table_name, column, candidate_table, target_column, "referenced_uid")
                        if key in seen:
                            continue
                        seen.add(key)
                        edges.append(
                            {
                                "left_table": table_name,
                                "left_column": column,
                                "right_table": candidate_table,
                                "right_column": target_column,
                                "kind": "referenced_uid",
                                "weight": 2,
                            }
                        )
                        break
    return edges


def _shortest_join_path(start_table: str, target_table: str, schema: dict) -> list[dict[str, Any]]:
    if start_table == target_table:
        return []
    edges = _schema_relationship_edges(schema)
    adjacency: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for edge in edges:
        adjacency.setdefault(edge["left_table"], []).append((edge["right_table"], edge))
        adjacency.setdefault(edge["right_table"], []).append((edge["left_table"], edge))

    heap: list[tuple[int, str, list[dict[str, Any]]]] = [(0, start_table, [])]
    best_cost = {start_table: 0}
    while heap:
        cost, table_name, path = heapq.heappop(heap)
        if table_name == target_table:
            return path
        if cost > best_cost.get(table_name, cost):
            continue
        for neighbor, edge in adjacency.get(table_name, []):
            next_cost = cost + int(edge.get("weight", 1))
            if next_cost >= best_cost.get(neighbor, 10**9):
                continue
            best_cost[neighbor] = next_cost
            heapq.heappush(heap, (next_cost, neighbor, path + [edge]))
    return []


def _extract_table_references(sql: str) -> list[str]:
    refs = []
    for match in TABLE_REFERENCE_PATTERN.finditer(sql or ""):
        candidate = _normalize_identifier(match.group(1))
        if not candidate:
            continue
        refs.append(candidate)
    return refs


def _extract_table_aliases(sql: str, schema: dict) -> dict[str, str]:
    table_lookup = _schema_table_lookup(schema)
    aliases = {}
    for match in TABLE_ALIAS_PATTERN.finditer(sql or ""):
        raw_ref = match.group(1)
        alias = match.group(2)
        normalized = _normalize_identifier(raw_ref)
        canonical = table_lookup.get(normalized.lower(), normalized)
        aliases[canonical] = canonical
        if alias and alias.lower() not in SQL_KEYWORDS:
            aliases[alias] = canonical
    return aliases


def _replace_outside_single_quotes(text: str, replacer) -> str:
    parts = re.split(r"('(?:''|[^'])*')", text)
    rewritten = []
    for index, part in enumerate(parts):
        if index % 2 == 1:
            rewritten.append(part)
            continue
        rewritten.append(replacer(part))
    return "".join(rewritten)


def _canonicalize_table_references(sql: str, schema: dict, dialect: str) -> str:
    table_lookup = _schema_table_lookup(schema)

    def repl(match: re.Match[str]) -> str:
        raw_ref = match.group(1)
        normalized = _normalize_identifier(raw_ref)
        canonical = table_lookup.get(normalized.lower())
        if not canonical:
            return match.group(0)
        return match.group(0).replace(raw_ref, _quote_qualified_identifier(canonical, dialect), 1)

    return _replace_outside_single_quotes(sql, lambda segment: TABLE_REFERENCE_PATTERN.sub(repl, segment))


def _canonicalize_column_identifiers(sql: str, schema: dict, dialect: str) -> str:
    column_lookup = _schema_column_lookup(schema)

    def replace_qualified(segment: str) -> str:
        def repl(match: re.Match[str]) -> str:
            left = match.group(1)
            right = match.group(2)
            canonical = column_lookup.get(right.lower())
            if not canonical:
                return match.group(0)
            canonical_column = canonical.split(".")[-1]
            return f"{left}.{_quote_identifier(canonical_column, dialect)}"

        return UNQUOTED_QUALIFIED_IDENTIFIER_PATTERN.sub(repl, segment)

    sql = _replace_outside_single_quotes(sql, replace_qualified)

    def replace_bare(segment: str) -> str:
        def repl(match: re.Match[str]) -> str:
            token = match.group(1)
            token_lc = token.lower()
            if token_lc in SQL_KEYWORDS:
                return token
            canonical = column_lookup.get(token_lc)
            if not canonical:
                return token
            canonical_column = canonical.split(".")[-1]
            return _quote_identifier(canonical_column, dialect)

        return UNQUOTED_IDENTIFIER_PATTERN.sub(repl, segment)

    return _replace_outside_single_quotes(sql, replace_bare)


def _canonicalize_sql_identifiers(sql: str, schema: dict, dialect: str) -> str:
    rewritten = _canonicalize_table_references(sql, schema, dialect)
    rewritten = _canonicalize_column_identifiers(rewritten, schema, dialect)
    return rewritten


def _schema_repair_hints(schema: dict, task: str, failing_sql: str = "", error_text: str = "") -> str:
    table_names = _schema_table_names(schema)
    column_names = _schema_column_names(schema)
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", " ".join([task or "", failing_sql or "", error_text or ""]))
        if len(token) >= 3
    }
    related_tables = []
    related_columns = []
    for name in table_names:
        name_lc = name.lower()
        if any(token in name_lc or name_lc.endswith(f".{token}") for token in tokens):
            related_tables.append(name)
    for name in column_names:
        name_lc = name.lower()
        if any(token in name_lc for token in tokens):
            related_columns.append(name)

    missing_identifier = None
    match = MISSING_RELATION_PATTERN.search(error_text or "")
    if match:
        missing_identifier = _normalize_identifier(match.group(1))
    if not missing_identifier:
        refs = _extract_table_references(failing_sql or "")
        for ref in refs:
            if ref not in table_names:
                missing_identifier = ref
                break
    close_tables = get_close_matches(missing_identifier or "", table_names, n=8, cutoff=0.2) if missing_identifier else []

    lines = [
        "Available identifiers to use exactly as written:",
        "Tables: " + (", ".join(table_names[:120]) if table_names else "(none)"),
    ]
    if related_tables:
        lines.append("Tables related to request terms: " + ", ".join(related_tables[:20]))
    if close_tables:
        lines.append("Closest table matches to the missing reference: " + ", ".join(close_tables))
    if related_columns:
        lines.append("Columns related to request terms: " + ", ".join(related_columns[:40]))
    alias_map = _extract_table_aliases(failing_sql, schema)
    columns_by_table = _schema_columns_by_table(schema)
    if alias_map:
        alias_lines = []
        for alias, table_name in alias_map.items():
            cols = sorted(columns_by_table.get(table_name, []))
            if cols:
                alias_lines.append(f"{alias} -> {table_name} columns: {', '.join(cols[:20])}")
        if alias_lines:
            lines.append("Alias-visible columns: " + " | ".join(alias_lines[:12]))
    lines.append(
        "If the schema does not contain a separate entity table for a concept in the request, answer using the actual available tables and columns instead of inventing a new table."
    )
    return "\n".join(lines)


def _validate_sql_against_schema(sql: str, schema: dict):
    table_names = set(_schema_table_names(schema))
    unknown = [ref for ref in _extract_table_references(sql) if ref not in table_names]
    if not unknown:
        return
    suggestions = []
    for ref in unknown:
        matches = get_close_matches(ref, list(table_names), n=5, cutoff=0.2)
        if matches:
            suggestions.append(f"{ref} -> {', '.join(matches)}")
    available_preview = ", ".join(sorted(table_names)[:30])
    parts = [f"SQL references unknown table(s): {', '.join(unknown)}."]
    if suggestions:
        parts.append("Possible matches: " + "; ".join(suggestions) + ".")
    if available_preview:
        parts.append("Available tables: " + available_preview + ".")
    raise RuntimeError(" ".join(parts))


def _validate_sql_columns_against_schema(sql: str, schema: dict):
    aliases = _extract_table_aliases(sql, schema)
    columns_by_table = _schema_columns_by_table(schema)
    unknown = []
    suggestions = []
    for match in QUALIFIED_IDENTIFIER_REFERENCE_PATTERN.finditer(sql or ""):
        qualifier = match.group(1)
        column = match.group(2) or match.group(3) or match.group(4) or ""
        table_name = aliases.get(qualifier)
        if not table_name:
            continue
        available = columns_by_table.get(table_name, set())
        if column in available:
            continue
        column_match = next((candidate for candidate in available if candidate.lower() == column.lower()), None)
        if column_match:
            continue
        unknown.append(f"{qualifier}.{column}")
        close = get_close_matches(column, sorted(available), n=5, cutoff=0.2)
        if close:
            suggestions.append(f"{qualifier}.{column} -> {table_name}: {', '.join(close)}")
    if not unknown:
        return
    detail = [f"SQL references unknown column(s): {', '.join(unknown)}."]
    if suggestions:
        detail.append("Possible matches: " + "; ".join(suggestions) + ".")
    raise RuntimeError(" ".join(detail))


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


def _llm_client_config():
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
    return api_key, base_url, model, timeout_seconds


def _llm_json_response(prompt: str, raw_log_label: str) -> list[dict[str, Any]]:
    api_key, base_url, model, timeout_seconds = _llm_client_config()
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
    log_raw(raw_log_label, content)
    return _extract_json_values(content)


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


def _task_tokens(task: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", task or "")
        if len(token) >= 2
    }


def _score_table_for_task(table: dict, task_tokens: set[str]) -> int:
    schema_name = str(table.get("schema", "") or "").strip().lower()
    table_name = str(table.get("name", "") or "").strip().lower()
    score = 0
    joined_name = f"{schema_name}.{table_name}".strip(".")
    for token in task_tokens:
        if token == table_name or token == joined_name:
            score += 10
        elif token in table_name:
            score += 6
        elif token in joined_name:
            score += 4
    for column in table.get("columns", []):
        column_name = str(column.get("name", "") or "").strip().lower()
        for token in task_tokens:
            if token == column_name:
                score += 7
            elif token in column_name:
                score += 3
    return score


def _relevant_schema_slice(task: str, schema: dict, limit: int = 12) -> dict:
    task_tokens = _task_tokens(task)
    scored = []
    for table in schema.get("tables", []):
        score = _score_table_for_task(table, task_tokens)
        scored.append((score, table))
    scored.sort(key=lambda item: (item[0], str(item[1].get("schema", "")), str(item[1].get("name", ""))), reverse=True)
    selected = [table for score, table in scored if score > 0][:limit]
    if not selected:
        selected = list(schema.get("tables", []))[: min(limit, len(schema.get("tables", [])))]
    selected_names = {
        ".".join(
            part
            for part in (
                str(table.get("schema", "") or "").strip(),
                str(table.get("name", "") or "").strip(),
            )
            if part
        )
        for table in selected
    }
    selected_edges = []
    for edge in _schema_foreign_keys(schema):
        if edge["source_table"] in selected_names or edge["target_table"] in selected_names:
            selected_edges.append(edge)
    return {"dialect": schema.get("dialect", "unknown"), "tables": selected, "foreign_keys": selected_edges}


def _schema_slice_text(schema_slice: dict, dialect: str) -> str:
    lines = [f"Dialect: {dialect}", "Relevant schema slice:"]
    for table in schema_slice.get("tables", []):
        schema_name = str(table.get("schema", "") or "").strip()
        table_name = str(table.get("name", "") or "").strip()
        qualified = ".".join(part for part in (schema_name, table_name) if part)
        columns = ", ".join(str(col.get("name", "") or "").strip() for col in table.get("columns", []) if str(col.get("name", "") or "").strip())
        lines.append(f"- {qualified}: {columns}")
    if schema_slice.get("foreign_keys"):
        lines.append("Relevant foreign keys:")
        for edge in schema_slice["foreign_keys"]:
            lines.append(
                f"- {edge['source_table']}.{edge['source_column']} -> {edge['target_table']}.{edge['target_column']}"
            )
    return "\n".join(lines)


def _llm_sql_intent(task: str, schema: dict) -> dict[str, Any] | None:
    schema_slice = _relevant_schema_slice(task, schema)
    prompt = (
        "You plan a read-only SQL query using a provided database schema.\n"
        "Return one JSON object only with keys: "
        '{"summary":"...","tables":["schema.table"],"dimensions":[{"table":"schema.table","column":"ColumnName","alias":"optional"}],'
        '"measures":[{"table":"schema.table","column":"ColumnName or null","aggregate":"count|count_distinct|sum|avg|min|max","alias":"metric_name"}],'
        '"joins":[{"left_table":"schema.table","left_column":"ColumnName","right_table":"schema.table","right_column":"ColumnName","join_type":"inner|left"}],'
        '"filters":[{"table":"schema.table","column":"ColumnName","operator":"=|!=|>|>=|<|<=|like|ilike|in|not_in|is_not_null|is_null","value":"literal or array"}],'
        '"group_by":[{"table":"schema.table","column":"ColumnName"}],'
        '"having":[{"measure_alias":"metric_name","operator":"=|!=|>|>=|<|<=","value":"number"}],'
        '"order_by":[{"type":"dimension|measure","table":"schema.table","column":"ColumnName","measure_alias":"metric_name","direction":"asc|desc"}],'
        '"limit":10,"notes":["optional assumption"]}\n'
        "Rules:\n"
        "- Use only tables and columns that exist in the schema.\n"
        "- Include every table needed in tables.\n"
        "- Each join must use real columns from the schema.\n"
        "- Prefer joins that match the foreign-key relationships shown.\n"
        "- Use dimensions for grouping/output identifiers and measures for aggregations.\n"
        "- If the user asks for counts, averages, maxima, minima, or sums, express them in measures.\n"
        "- For 'most recent' use max on the relevant date column.\n"
        "- If the request implies ranking, populate order_by and limit.\n"
        "- Do not emit SQL.\n"
        f"{_schema_slice_text(schema_slice, schema.get('dialect', 'unknown'))}\n"
        f"User question: {task}"
    )
    values = _llm_json_response(prompt, "SQL_LLM_PLAN_RAW")
    for value in values:
        if isinstance(value, dict) and ("dimensions" in value or "measures" in value or "tables" in value):
            return value
    return None


def _canonical_table_name(name: str, schema: dict) -> str | None:
    return _schema_table_lookup(schema).get((name or "").strip().lower())


def _canonical_column_name(table_name: str, column_name: str, schema: dict) -> str | None:
    metadata = _schema_table_metadata(schema).get(table_name)
    if not metadata:
        return None
    for column in metadata.get("columns", []):
        candidate = str(column.get("name", "") or "").strip()
        if candidate == column_name:
            return candidate
        if candidate.lower() == column_name.lower():
            return candidate
    return None


def _foreign_key_join_exists(left_table: str, left_column: str, right_table: str, right_column: str, schema: dict) -> bool:
    for edge in _schema_relationship_edges(schema):
        if (
            edge["left_table"] == left_table
            and edge["left_column"] == left_column
            and edge["right_table"] == right_table
            and edge["right_column"] == right_column
        ) or (
            edge["left_table"] == right_table
            and edge["left_column"] == right_column
            and edge["right_table"] == left_table
            and edge["right_column"] == left_column
        ):
            return True
    return False


def _resolve_sql_plan(plan: dict[str, Any], schema: dict, task: str = "") -> dict[str, Any]:
    resolved = {
        "summary": str(plan.get("summary") or "").strip(),
        "tables": [],
        "dimensions": [],
        "measures": [],
        "joins": [],
        "filters": [],
        "group_by": [],
        "having": [],
        "order_by": [],
        "limit": None,
        "notes": [str(note) for note in plan.get("notes", []) if isinstance(note, str)],
    }

    requested_tables = []
    seen_tables = set()
    for raw_table in plan.get("tables", []):
        if not isinstance(raw_table, str):
            continue
        canonical = _canonical_table_name(raw_table, schema)
        if canonical and canonical not in seen_tables:
            seen_tables.add(canonical)
            requested_tables.append(canonical)

    for item in plan.get("dimensions", []):
        if not isinstance(item, dict):
            continue
        table_name = _canonical_table_name(str(item.get("table") or ""), schema)
        column_name = _canonical_column_name(table_name or "", str(item.get("column") or ""), schema) if table_name else None
        if not table_name or not column_name:
            continue
        resolved["dimensions"].append(
            {"table": table_name, "column": column_name, "alias": str(item.get("alias") or "").strip()}
        )
        if table_name not in seen_tables:
            seen_tables.add(table_name)
            requested_tables.append(table_name)

    for item in plan.get("measures", []):
        if not isinstance(item, dict):
            continue
        aggregate = str(item.get("aggregate") or "").strip().lower()
        table_name = _canonical_table_name(str(item.get("table") or ""), schema)
        raw_column = None if item.get("column") in (None, "", "*") else str(item.get("column") or "")
        alias = str(item.get("alias") or "").strip() or f"{aggregate}_value"
        table_name, column_name, aggregate = _best_measure_column(
            aggregate,
            table_name,
            raw_column,
            alias,
            task,
            schema,
            allowed_tables=None,
        )
        if not table_name:
            continue
        resolved["measures"].append(
            {"table": table_name, "column": column_name, "aggregate": aggregate, "alias": alias}
        )
        if table_name not in seen_tables:
            seen_tables.add(table_name)
            requested_tables.append(table_name)

    for item in plan.get("filters", []):
        if not isinstance(item, dict):
            continue
        table_name = _canonical_table_name(str(item.get("table") or ""), schema)
        column_name = _canonical_column_name(table_name or "", str(item.get("column") or ""), schema) if table_name else None
        operator = str(item.get("operator") or "").strip().lower()
        if not table_name or not column_name or operator not in {"=", "!=", ">", ">=", "<", "<=", "like", "ilike", "in", "not_in", "is_not_null", "is_null"}:
            continue
        resolved["filters"].append(
            {
                "table": table_name,
                "column": column_name,
                "operator": operator,
                "value": item.get("value"),
            }
        )
        if table_name not in seen_tables:
            seen_tables.add(table_name)
            requested_tables.append(table_name)

    for item in plan.get("group_by", []):
        if not isinstance(item, dict):
            continue
        table_name = _canonical_table_name(str(item.get("table") or ""), schema)
        column_name = _canonical_column_name(table_name or "", str(item.get("column") or ""), schema) if table_name else None
        if table_name and column_name:
            resolved["group_by"].append({"table": table_name, "column": column_name})

    measure_aliases = {item["alias"] for item in resolved["measures"]}
    for item in plan.get("having", []):
        if not isinstance(item, dict):
            continue
        alias = str(item.get("measure_alias") or "").strip()
        operator = str(item.get("operator") or "").strip()
        if alias in measure_aliases and operator in {"=", "!=", ">", ">=", "<", "<="}:
            resolved["having"].append({"measure_alias": alias, "operator": operator, "value": item.get("value")})

    for item in plan.get("order_by", []):
        if not isinstance(item, dict):
            continue
        order_type = str(item.get("type") or "").strip().lower()
        direction = str(item.get("direction") or "asc").strip().lower()
        if direction not in {"asc", "desc"}:
            direction = "asc"
        if order_type == "measure":
            alias = str(item.get("measure_alias") or "").strip()
            if alias in measure_aliases:
                resolved["order_by"].append({"type": "measure", "measure_alias": alias, "direction": direction})
        elif order_type == "dimension":
            table_name = _canonical_table_name(str(item.get("table") or ""), schema)
            column_name = _canonical_column_name(table_name or "", str(item.get("column") or ""), schema) if table_name else None
            if table_name and column_name:
                resolved["order_by"].append(
                    {"type": "dimension", "table": table_name, "column": column_name, "direction": direction}
                )

    limit = plan.get("limit")
    if isinstance(limit, (int, float)):
        resolved["limit"] = max(1, min(int(limit), _row_limit()))

    required_tables = []
    seen_required = set()
    for item in resolved["dimensions"]:
        if item["table"] not in seen_required:
            seen_required.add(item["table"])
            required_tables.append(item["table"])
    for item in resolved["measures"]:
        if item["table"] not in seen_required:
            seen_required.add(item["table"])
            required_tables.append(item["table"])
    for item in resolved["filters"]:
        if item["table"] not in seen_required:
            seen_required.add(item["table"])
            required_tables.append(item["table"])
    for item in resolved["group_by"]:
        if item["table"] not in seen_required:
            seen_required.add(item["table"])
            required_tables.append(item["table"])
    if not required_tables and requested_tables:
        required_tables = requested_tables[:1]

    resolved_tables, inferred_joins = _infer_join_tree(required_tables, schema)
    resolved["tables"] = resolved_tables
    resolved["joins"] = inferred_joins
    return resolved


def _literal_sql(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _best_measure_column(
    aggregate: str,
    preferred_table: str | None,
    requested_column: str | None,
    alias: str,
    task: str,
    schema: dict,
    allowed_tables: set[str] | None = None,
) -> tuple[str | None, str | None, str]:
    aggregate_lc = aggregate.lower()
    if preferred_table and requested_column:
        canonical = _canonical_column_name(preferred_table, requested_column, schema)
        metadata = _column_metadata(preferred_table, canonical or requested_column, schema) if canonical else None
        if canonical and (
            aggregate_lc in {"count", "count_distinct"}
            or (metadata and _is_numeric_column_type(str(metadata.get("type", "") or "")))
            or aggregate_lc in {"min", "max"}
        ):
            return preferred_table, canonical, aggregate_lc

    if preferred_table and aggregate_lc == "count":
        identifier = _best_identifier_column(preferred_table, schema)
        if identifier:
            return preferred_table, identifier, "count_distinct"
        return preferred_table, None, "count"

    search_tokens = _task_tokens(" ".join([task or "", alias or "", requested_column or ""]))
    candidates = []
    for table_name, table in _schema_table_metadata(schema).items():
        if allowed_tables and table_name not in allowed_tables:
            continue
        for column in table.get("columns", []):
            column_name = str(column.get("name", "") or "").strip()
            column_type = str(column.get("type", "") or "").strip()
            if not column_name:
                continue
            if aggregate_lc in {"avg", "sum"} and not _is_numeric_column_type(column_type):
                continue
            score = 0
            column_lc = column_name.lower()
            table_lc = table_name.lower()
            for token in search_tokens:
                if token == column_lc:
                    score += 12
                elif token in column_lc:
                    score += 7
                if token in table_lc:
                    score += 3
            if aggregate_lc == "avg":
                if "dose" in column_lc:
                    score += 12
                if "maximum" in column_lc or "minimum" in column_lc:
                    score -= 4
                if "prescription" in column_lc or "delivery" in column_lc:
                    score += 3
                if "scaling" in column_lc:
                    score -= 8
                if "dosereferencesequence" in table_lc:
                    score += 6
            if aggregate_lc == "count" and _is_key_like_column(column_name):
                score += 5
            if aggregate_lc == "max" and "date" in column_lc:
                score += 6
            if aggregate_lc == "min" and "date" in column_lc:
                score += 4
            if preferred_table and table_name == preferred_table:
                score += 4
            if score > 0:
                candidates.append((score, table_name, column_name))
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        score, table_name, column_name = candidates[0]
        resolved_aggregate = "count_distinct" if aggregate_lc == "count" and _is_key_like_column(column_name) else aggregate_lc
        return table_name, column_name, resolved_aggregate
    if preferred_table and aggregate_lc == "count":
        return preferred_table, None, "count"
    return preferred_table, None, aggregate_lc


def _infer_join_tree(required_tables: list[str], schema: dict) -> tuple[list[str], list[dict[str, Any]]]:
    if not required_tables:
        return [], []
    resolved_tables = [required_tables[0]]
    join_edges: list[dict[str, Any]] = []
    seen_edges = set()
    connected = {required_tables[0]}
    for table_name in required_tables[1:]:
        if table_name in connected:
            continue
        best_path = None
        best_cost = None
        best_anchor = None
        for anchor in connected:
            path = _shortest_join_path(anchor, table_name, schema)
            if not path:
                continue
            cost = sum(int(edge.get("weight", 1)) for edge in path)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_path = path
                best_anchor = anchor
        if not best_path:
            raise RuntimeError(f"Could not infer a join path from {sorted(connected)} to {table_name}.")
        for edge in best_path:
            edge_key = (
                edge["left_table"],
                edge["left_column"],
                edge["right_table"],
                edge["right_column"],
                edge.get("kind"),
            )
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                join_edges.append(
                    {
                        "left_table": edge["left_table"],
                        "left_column": edge["left_column"],
                        "right_table": edge["right_table"],
                        "right_column": edge["right_column"],
                        "join_type": "inner",
                    }
                )
            for candidate in (edge["left_table"], edge["right_table"]):
                if candidate not in connected:
                    connected.add(candidate)
                    resolved_tables.append(candidate)
    return resolved_tables, join_edges


def _compile_filter_sql(filter_spec: dict[str, Any], aliases: dict[str, str], dialect: str) -> str:
    table_alias = aliases[filter_spec["table"]]
    column_sql = f"{table_alias}.{_quote_identifier(filter_spec['column'], dialect)}"
    operator = filter_spec["operator"]
    value = filter_spec.get("value")
    if operator == "is_null":
        return f"{column_sql} IS NULL"
    if operator == "is_not_null":
        return f"{column_sql} IS NOT NULL"
    if operator in {"in", "not_in"}:
        items = value if isinstance(value, list) else [value]
        rendered = ", ".join(_literal_sql(item) for item in items)
        keyword = "IN" if operator == "in" else "NOT IN"
        return f"{column_sql} {keyword} ({rendered})"
    keyword = operator.upper() if operator != "ilike" else "ILIKE"
    return f"{column_sql} {keyword} {_literal_sql(value)}"


def _measure_expression_sql(measure: dict[str, Any], aliases: dict[str, str], dialect: str) -> str:
    aggregate = measure["aggregate"]
    if aggregate == "count" and not measure.get("column"):
        return "COUNT(*)"
    table_alias = aliases[measure["table"]]
    column_sql = f"{table_alias}.{_quote_identifier(measure['column'], dialect)}"
    if aggregate == "count_distinct":
        return f"COUNT(DISTINCT {column_sql})"
    return f"{aggregate.upper()}({column_sql})"


def _compile_measure_sql(measure: dict[str, Any], aliases: dict[str, str], dialect: str) -> str:
    alias = _quote_identifier(measure["alias"], dialect)
    return f"{_measure_expression_sql(measure, aliases, dialect)} AS {alias}"


def _compile_sql_from_plan(plan: dict[str, Any], schema: dict, dialect: str) -> str:
    if not plan.get("dimensions") and not plan.get("measures"):
        raise RuntimeError("Planned query is missing both dimensions and measures.")
    if not plan.get("tables"):
        raise RuntimeError("Planned query is missing tables.")

    aliases = {table_name: f"t{index}" for index, table_name in enumerate(plan["tables"], start=1)}
    select_parts = []
    group_by_parts = []
    for item in plan.get("dimensions", []):
        expr = f"{aliases[item['table']]}.{_quote_identifier(item['column'], dialect)}"
        alias = str(item.get("alias") or "").strip()
        if alias:
            select_parts.append(f"{expr} AS {_quote_identifier(alias, dialect)}")
        else:
            select_parts.append(expr)
    for item in plan.get("group_by", []):
        group_by_parts.append(f"{aliases[item['table']]}.{_quote_identifier(item['column'], dialect)}")
    if not group_by_parts and plan.get("dimensions"):
        for item in plan["dimensions"]:
            group_by_parts.append(f"{aliases[item['table']]}.{_quote_identifier(item['column'], dialect)}")
    measure_lookup = {item["alias"]: item for item in plan.get("measures", [])}
    select_parts.extend(_compile_measure_sql(item, aliases, dialect) for item in plan.get("measures", []))

    root_table = plan["tables"][0]
    from_sql = f"FROM {_quote_qualified_identifier(root_table, dialect)} {aliases[root_table]}"
    joined = {root_table}
    join_clauses = []
    pending = list(plan.get("joins", []))
    while pending:
        progress = False
        next_pending = []
        for join in pending:
            left_table = join["left_table"]
            right_table = join["right_table"]
            if left_table in joined and right_table not in joined:
                join_table = right_table
                source_table = left_table
                source_column = join["left_column"]
                target_column = join["right_column"]
            elif right_table in joined and left_table not in joined:
                join_table = left_table
                source_table = right_table
                source_column = join["right_column"]
                target_column = join["left_column"]
            else:
                next_pending.append(join)
                continue
            join_sql = (
                f"{join['join_type'].upper()} JOIN {_quote_qualified_identifier(join_table, dialect)} {aliases[join_table]} "
                f"ON {aliases[source_table]}.{_quote_identifier(source_column, dialect)} = "
                f"{aliases[join_table]}.{_quote_identifier(target_column, dialect)}"
            )
            join_clauses.append(join_sql)
            joined.add(join_table)
            progress = True
        if not progress:
            raise RuntimeError("Could not connect planned joins into one query graph.")
        pending = next_pending

    referenced_tables = {item["table"] for item in plan.get("dimensions", [])}
    referenced_tables.update(item["table"] for item in plan.get("measures", []) if item.get("table"))
    referenced_tables.update(item["table"] for item in plan.get("filters", []))
    referenced_tables.update(item["table"] for item in plan.get("group_by", []))
    referenced_tables.update(item["table"] for item in plan.get("order_by", []) if item.get("type") == "dimension")
    unjoined_tables = sorted(referenced_tables - joined)
    if unjoined_tables:
        raise RuntimeError(
            "Planned query references table(s) that are not connected by valid joins: " + ", ".join(unjoined_tables)
        )

    where_parts = [_compile_filter_sql(item, aliases, dialect) for item in plan.get("filters", [])]
    having_parts = []
    for item in plan.get("having", []):
        measure = measure_lookup.get(item["measure_alias"])
        if not measure:
            continue
        having_parts.append(
            f"{_measure_expression_sql(measure, aliases, dialect)} {item['operator']} {_literal_sql(item['value'])}"
        )
    order_parts = []
    for item in plan.get("order_by", []):
        if item["type"] == "measure":
            order_parts.append(f"{_quote_identifier(item['measure_alias'], dialect)} {item['direction'].upper()}")
        else:
            order_parts.append(
                f"{aliases[item['table']]}.{_quote_identifier(item['column'], dialect)} {item['direction'].upper()}"
            )

    sql_parts = [
        "SELECT " + ", ".join(select_parts),
        from_sql,
    ]
    sql_parts.extend(join_clauses)
    if where_parts:
        sql_parts.append("WHERE " + " AND ".join(where_parts))
    if group_by_parts:
        sql_parts.append("GROUP BY " + ", ".join(group_by_parts))
    if having_parts:
        sql_parts.append("HAVING " + " AND ".join(having_parts))
    if order_parts:
        sql_parts.append("ORDER BY " + ", ".join(order_parts))
    if plan.get("limit"):
        sql_parts.append(f"LIMIT {int(plan['limit'])}")
    return "\n".join(sql_parts)


def _plan_and_compile_sql(task: str, schema: dict, dialect: str) -> dict[str, Any] | None:
    plan = _llm_sql_intent(task, schema)
    if not plan:
        return None
    resolved_plan = _resolve_sql_plan(plan, schema, task=task)
    sql = _compile_sql_from_plan(resolved_plan, schema, dialect)
    return {
        "plan": resolved_plan,
        "query_specs": [{"label": resolved_plan.get("summary") or "planned query", "sql": sql}],
    }


def _llm_sql_queries(task: str, schema: dict) -> list[dict[str, str]]:
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
        "- If the schema has no separate table matching a requested business entity, use the relevant columns from the real tables in the schema.\n"
        "- Prefer SELECT or WITH queries. Do not generate mutating SQL.\n"
        "- Add a reasonable LIMIT unless the user asks for aggregation only.\n"
        "- Use the SQL dialect from the schema.\n"
        "Schema:\n"
        f"{_schema_summary(schema)}\n"
        f"{_schema_identifier_catalog(schema)}\n"
        f"{_schema_repair_hints(schema, task)}\n"
        f"User question: {task}"
    )
    return _sql_query_specs_from_json(_llm_json_response(prompt, "SQL_LLM_RAW"))


def _repair_sql_query(task: str, schema: dict, failing_sql: str, error_text: str) -> list[dict[str, str]]:
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
        "- If the schema has no separate table matching a requested business entity, use the relevant columns from the real tables in the schema.\n"
        "- Return a single read-only SQL query.\n"
        "Schema:\n"
        f"{_schema_summary(schema)}\n"
        f"{_schema_identifier_catalog(schema)}\n"
        f"{_schema_repair_hints(schema, task, failing_sql, error_text)}\n"
        f"Original user request: {task}\n"
        f"Failing SQL:\n{failing_sql}\n"
        f"Database error:\n{error_text}"
    )
    return _sql_query_specs_from_json(_llm_json_response(prompt, "SQL_LLM_REPAIR_RAW"))


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


def _should_retry_sql_repair(exc: Exception) -> bool:
    return bool(IDENTIFIER_ERROR_PATTERN.search(f"{type(exc).__name__}: {exc}"))


def _prepare_query_specs(query_specs: list[dict[str, str]], schema: dict, dialect: str):
    prepared = []
    for spec in query_specs:
        rewritten = dict(spec)
        sql = rewritten.get("sql")
        if isinstance(sql, str):
            rewritten["sql"] = _canonicalize_sql_identifiers(sql, schema, dialect)
        prepared.append(rewritten)
    return prepared


def _is_generic_query_task(task: str) -> bool:
    task_lc = (task or "").strip().lower()
    generic_phrases = {
        "query the database",
        "query the database for the required information",
        "query to retrieve the required information",
        "query the database for the requested information",
    }
    if task_lc in generic_phrases:
        return True
    return any(phrase in task_lc for phrase in generic_phrases)


def _preferred_query_task(instruction: dict | None, execution_task: str, original_task: str) -> str:
    instruction_question = ""
    if isinstance(instruction, dict) and isinstance(instruction.get("question"), str):
        instruction_question = instruction["question"].strip()
    candidates = [instruction_question, execution_task.strip(), original_task.strip()]
    for candidate in candidates:
        if candidate and not _is_generic_query_task(candidate):
            return candidate
    return instruction_question or execution_task or original_task


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
        original_task = task
        provided_sql = req.payload.get("sql")
        instruction = {"operation": "query_from_request", "question": task}
    elif req.event == "task.plan":
        plan_context = task_plan_context(req.payload)
        classification_task = plan_context.classification_task
        execution_task = plan_context.execution_task
        original_task = plan_context.original_task
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
                compiled_plan = None
                query_task = _preferred_query_task(instruction, execution_task, original_task)
            else:
                started = time.perf_counter()
                query_task = _preferred_query_task(instruction, execution_task, original_task)
                compiled_plan = None
                planner_error = None
                try:
                    compiled_plan = _plan_and_compile_sql(query_task, schema, dialect)
                except Exception as exc:
                    planner_error = f"{type(exc).__name__}: {exc}"
                    _debug_log(f"Structured SQL planning failed, falling back to direct SQL generation: {planner_error}")
                if compiled_plan:
                    query_specs = compiled_plan["query_specs"]
                    stats["sql_planning_ms"] = _elapsed_ms(started)
                    stats["sql_generation_ms"] = 0
                else:
                    query_specs = _llm_sql_queries(query_task, schema)
                    stats["sql_generation_ms"] = _elapsed_ms(started)
                    if planner_error:
                        stats["sql_planning_fallback"] = 1
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
            attempt_specs = _prepare_query_specs(query_specs, schema, dialect)
            total_repair_ms = 0.0
            last_exc = None
            result = None
            for _attempt in range(3):
                failing_sql = "\n\n".join(spec.get("sql", "") for spec in attempt_specs if isinstance(spec, dict))
                try:
                    for spec in attempt_specs:
                        sql_text = spec.get("sql", "")
                        _validate_sql_against_schema(sql_text, schema)
                        _validate_sql_columns_against_schema(sql_text, schema)
                    result = _execute_sql_queries(conn, attempt_specs, _row_limit())
                    query_specs = attempt_specs
                    break
                except Exception as exc:
                    last_exc = exc
                    if not _should_retry_sql_repair(exc):
                        raise
                    repair_started = time.perf_counter()
                    repaired_specs = _repair_sql_query(query_task, schema, failing_sql, f"{type(exc).__name__}: {exc}")
                    total_repair_ms += _elapsed_ms(repair_started)
                    if not repaired_specs:
                        raise
                    attempt_specs = _prepare_query_specs(repaired_specs, schema, dialect)
            if result is None and last_exc is not None:
                raise last_exc
            if total_repair_ms:
                stats["sql_repair_ms"] = round(total_repair_ms, 2)
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
                                "plan": compiled_plan["plan"] if compiled_plan else None,
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
