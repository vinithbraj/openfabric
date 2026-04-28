# SQL Capability Design

The SQL capability is deterministic first and schema-aware. It handles configured SQL databases by loading a catalog of schemas, tables, and columns, then compiling safe read-only plans before the raw planner can invent SQL.

## Schema Catalog

`src/aor_runtime/tools/sql.py` exposes `get_sql_catalog(...)`, `get_all_sql_catalogs(...)`, and `refresh_schema_cache(...)`.

For PostgreSQL, introspection includes all non-system schemas and excludes `pg_catalog`, `information_schema`, `pg_toast*`, and `pg_temp*`. Mixed-case identifiers are preserved exactly. Catalog entries are represented as `SqlSchemaCatalog`, `SqlTableRef`, and `SqlColumnRef`.

The older `get_schema()` API remains as a compatibility wrapper for planner context and existing tests.

## Identifier Quoting

`src/aor_runtime/runtime/sql_safety.py` owns PostgreSQL identifier normalization:

- `quote_pg_identifier(name)`
- `quote_pg_relation(schema, table)`
- `normalize_pg_relation_quoting(sql, catalog)`

The normalizer fixes invalid relation forms such as `"flathr.Patient"` and unquoted mixed-case forms such as `flathr.Patient`. It also normalizes alias-qualified mixed-case columns like `p.PatientID` when the catalog proves the column exists.

## Safety

All SQL execution passes through read-only validation. The validator allows one `SELECT` statement, or one `WITH` query that resolves to `SELECT`, and rejects mutation/admin/procedure forms such as `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `TRUNCATE`, `COPY`, `CALL`, `DO`, `EXECUTE`, `MERGE`, `LOCK`, `GRANT`, and `REVOKE`.

Comments and multi-statements are rejected conservatively.

## Resolver

`src/aor_runtime/runtime/sql_resolver.py` resolves natural-language table and column references against the catalog. It supports exact names, case-insensitive names, schema-qualified names, singular/plural normalization, token matching, and high-confidence fuzzy matching.

Generic semantic aliases are catalog-backed. For example, `age` maps to a birth-date column only if such a column exists.

## Constraint Coverage

`src/aor_runtime/runtime/sql_constraints.py` extracts semantic constraints before SQL is generated. It currently tracks target tables, age comparisons, related-row counts, distinct/group/order/limit requests, and unknown constraints that must not be dropped.

Generated SQL must pass constraint coverage before `sql.query` can run. For example, `above age 70` requires a resolved birth-date predicate, and `with 2 studies` requires a related-table join or equivalent count predicate with `COUNT(...) = 2`. If any extracted constraint is unresolved or missing from the final SQL, deterministic planning stops and the request either goes through schema-aware LLM SQL generation or fails safely with SQL-specific suggestions.

This is the guardrail that prevents constrained prompts from silently degrading to an unfiltered `COUNT(*)`.

## LLM SQL Generation

When `AOR_ENABLE_SQL_LLM_GENERATION=true`, broad SQL questions can use `src/aor_runtime/runtime/sql_llm.py`.

The LLM receives compact schema context only. It must return JSON containing a single PostgreSQL read-only query. The generated SQL is normalized, validated, checked against declared schema references, optionally validated with PostgreSQL `EXPLAIN`, and then executed only through `sql.query`.

The LLM never emits execution plans, shell commands, Python, or raw tool calls.

When a constraint frame exists, the LLM prompt includes it and the response may declare `constraints_addressed`. The runtime still performs its own coverage validation; the declaration is telemetry, not trust.

## Repair

On SQL execution retry, the SQL capability can repair prior generated SQL using the failure context from `sql.query`. Repair is bounded to two attempts and uses the same schema context, normalization, and read-only validation.

## Configuration

`AOR_ENABLE_SQL_LLM_GENERATION` defaults to false. With it disabled, deterministic schema-aware SQL still works, while broad questions fail safely with SQL-specific suggestions.
