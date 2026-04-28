from __future__ import annotations

import re
from typing import Any

from aor_runtime.llm.client import LLMClient
from aor_runtime.runtime.capabilities.base import CapabilityPack, ClassificationContext, CompileContext, CompiledIntentPlan
from aor_runtime.runtime.intent_classifier import classify_single_intent
from aor_runtime.runtime.intent_compiler import compile_intent_to_plan
from aor_runtime.runtime.intents import (
    IntentResult,
    SqlCatalogReturnIntent,
    SqlCountIntent,
    SqlFailureIntent,
    SqlGeneratedQueryIntent,
    SqlSelectIntent,
)
from aor_runtime.runtime.output_contract import build_output_contract
from aor_runtime.runtime.sql_catalog import SqlSchemaCatalog, SqlTableRef
from aor_runtime.runtime.sql_constraints import (
    SqlConstraint,
    SqlConstraintCoverageResult,
    SqlConstraintFrame,
    SqlProjection,
    constraint_telemetry,
    extract_sql_constraints,
    resolve_sql_constraints,
    validate_sql_constraint_coverage,
)
from aor_runtime.runtime.sql_llm import generate_sql_for_goal, repair_sql_for_error
from aor_runtime.runtime.sql_resolver import resolve_sql_references, resolve_table_name
from aor_runtime.runtime.sql_safety import normalize_pg_relation_quoting, quote_pg_identifier, quote_pg_relation, validate_read_only_sql
from aor_runtime.tools.sql import explain_sql_query, get_sql_catalog, resolve_database_selection


SQL_MUTATION_RE = re.compile(
    r"\b(?:insert|update|delete|drop|alter|create|truncate|copy|call|do|execute|merge|grant|revoke|vacuum|analyze|lock)\b",
    re.IGNORECASE,
)
SQL_HINT_RE = re.compile(
    r"\b(?:sql|database|table|tables|schema|schemas|column|columns|rows?|select|query|queries|join|group(?:ed)?|distinct|count|describe)\b",
    re.IGNORECASE,
)
LIST_TABLES_RE = re.compile(r"\b(?:list|show|return)\s+(?:all\s+)?tables\b", re.IGNORECASE)
LIST_SCHEMAS_RE = re.compile(r"\b(?:list|show|return)\s+(?:all\s+)?schemas\b", re.IGNORECASE)
DESCRIBE_TABLE_RE = re.compile(r"\b(?:describe|show\s+schema\s+for|show\s+columns\s+for|list\s+columns\s+(?:for|in))\s+(?:table\s+)?(?P<table>[\w\".]+)", re.IGNORECASE)
COUNT_RE = re.compile(r"\b(?:count|how\s+many|number\s+of)\b", re.IGNORECASE)
LIMIT_RE = re.compile(r"\b(?:top|first|limit|list|show|latest|oldest)\s+(?P<limit>\d+)\b", re.IGNORECASE)
SLURM_SQL_EXCLUSION_RE = re.compile(
    r"\b(?:slurm|squeue|sacct|sinfo|scontrol|slurmdbd|scheduler|queue|gpu|gpus|gres)\b",
    re.IGNORECASE,
)


class SqlCapabilityPack(CapabilityPack):
    name = "sql"
    intent_types = (SqlCountIntent, SqlSelectIntent, SqlGeneratedQueryIntent, SqlCatalogReturnIntent, SqlFailureIntent)

    def classify(self, goal: str, context: ClassificationContext) -> IntentResult:
        if SLURM_SQL_EXCLUSION_RE.search(str(goal or "")):
            return IntentResult(matched=False, reason=f"{self.name}_slurm_domain")
        legacy = classify_single_intent(goal, schema_payload=context.schema_payload)
        if legacy.matched and isinstance(legacy.intent, (SqlCountIntent, SqlSelectIntent)):
            return legacy

        if "sql.query" not in context.allowed_tools and "runtime.return" not in context.allowed_tools:
            return IntentResult(matched=False, reason=f"{self.name}_unavailable")
        if not _looks_like_sql_goal(goal, context):
            return IntentResult(matched=False, reason=f"{self.name}_no_match")

        if SQL_MUTATION_RE.search(goal):
            return _failure(
                "SQL mutation requests are not supported. This SQL capability only runs read-only SELECT queries.",
                error_type="sql_readonly_validation_failed",
            )

        try:
            database_name, _ = resolve_database_selection(context.settings, _database_hint(goal, context))
            catalog = get_sql_catalog(context.settings, database_name)
        except Exception as exc:  # noqa: BLE001
            return _failure(
                f"SQL schema is unavailable: {exc}",
                error_type="sql_schema_unavailable",
                database=_database_hint(goal, context),
            )

        constraint_frame = resolve_sql_constraints(extract_sql_constraints(goal), catalog)
        deterministic = _classify_deterministic_sql(goal, catalog, constraint_frame=constraint_frame)
        if deterministic.matched:
            return deterministic

        if not context.settings.enable_sql_llm_generation:
            if constraint_frame.unresolved_projections:
                return _failure(
                    "That SQL question requested projections that could not be resolved against the schema catalog.",
                    error_type="sql_projection_unresolved",
                    database=database_name,
                    catalog=catalog,
                    constraint_frame=constraint_frame,
                )
            if constraint_frame.unresolved_constraints:
                return _failure(
                    "That SQL question contains constraints that could not be resolved against the schema catalog.",
                    error_type="sql_constraint_unresolved",
                    database=database_name,
                    catalog=catalog,
                    constraint_frame=constraint_frame,
                )
            if constraint_frame.projections:
                return _failure(
                    "That SQL question requested projections that deterministic SQL could not cover safely.",
                    error_type="sql_projection_uncovered",
                    database=database_name,
                    catalog=catalog,
                    constraint_frame=constraint_frame,
                )
            if constraint_frame.non_target_constraints:
                return _failure(
                    "That SQL question contains constraints that deterministic SQL could not cover safely.",
                    error_type="sql_constraint_uncovered",
                    database=database_name,
                    catalog=catalog,
                    constraint_frame=constraint_frame,
                )
            return _failure(
                "That SQL question needs schema-aware SQL generation, but AOR_ENABLE_SQL_LLM_GENERATION is disabled.",
                error_type="sql_generation_failed",
                database=database_name,
                catalog=catalog,
                constraint_frame=constraint_frame,
            )

        return _classify_llm_sql(goal, context, catalog, constraint_frame=constraint_frame)

    def compile(self, intent: Any, context: CompileContext) -> CompiledIntentPlan | None:
        if isinstance(intent, (SqlCountIntent, SqlSelectIntent)):
            return CompiledIntentPlan(
                plan=compile_intent_to_plan(intent, context.allowed_tools, context.settings),
                metadata={"capability_pack": self.name, "intent_type": type(intent).__name__},
            )
        if isinstance(intent, SqlGeneratedQueryIntent):
            value: Any = {"$ref": "sql_result", "path": "rows"}
            mode = intent.output_mode
            json_shape = "rows" if mode in {"json", "text", "csv"} else None
            if mode == "count" and intent.scalar_key:
                value = {"$ref": "sql_result", "path": f"rows.0.{intent.scalar_key}"}
                json_shape = None
            plan = {
                "steps": [
                    {
                        "id": 1,
                        "action": "sql.query",
                        "args": {"database": intent.database, "query": intent.query},
                        "output": "sql_result",
                    },
                    {
                        "id": 2,
                        "action": "runtime.return",
                        "input": ["sql_result"],
                        "args": {
                            "value": value,
                            "mode": mode,
                            "output_contract": build_output_contract(mode=mode, json_shape=json_shape),
                        },
                    },
                ]
            }
            from aor_runtime.core.contracts import ExecutionPlan

            return CompiledIntentPlan(
                plan=ExecutionPlan.model_validate(plan),
                planning_mode=str(intent.metadata.get("planning_mode") or "sql_llm_generation"),
                metadata={"capability_pack": self.name, "intent_type": type(intent).__name__, **intent.metadata},
            )
        if isinstance(intent, SqlCatalogReturnIntent):
            from aor_runtime.core.contracts import ExecutionPlan

            plan = ExecutionPlan.model_validate(
                {
                    "steps": [
                        {
                            "id": 1,
                            "action": "runtime.return",
                            "args": {
                                "value": intent.value,
                                "mode": intent.output_mode,
                                "output_contract": build_output_contract(
                                    mode=intent.output_mode,
                                    json_shape="rows" if intent.output_mode == "json" and isinstance(intent.value, list) else None,
                                ),
                            },
                        }
                    ]
                }
            )
            return CompiledIntentPlan(
                plan=plan,
                planning_mode="sql_schema",
                metadata={"capability_pack": self.name, "intent_type": type(intent).__name__, **intent.metadata},
            )
        if isinstance(intent, SqlFailureIntent):
            from aor_runtime.core.contracts import ExecutionPlan

            message = intent.message
            if intent.suggestions:
                message = f"{message}\n\nSuggested prompts:\n" + "\n".join(f"{index}. {item}" for index, item in enumerate(intent.suggestions, start=1))
            plan = ExecutionPlan.model_validate(
                {"steps": [{"id": 1, "action": "runtime.return", "args": {"value": message, "mode": "text"}}]}
            )
            return CompiledIntentPlan(
                plan=plan,
                planning_mode="sql_safe_failure",
                metadata={"capability_pack": self.name, "intent_type": type(intent).__name__, **intent.metadata},
            )
        return None


def _classify_deterministic_sql(goal: str, catalog: SqlSchemaCatalog, *, constraint_frame: SqlConstraintFrame) -> IntentResult:
    goal_text = str(goal or "")
    database = catalog.database
    if LIST_SCHEMAS_RE.search(goal_text):
        schemas = sorted({table.schema_name for table in catalog.tables})
        return IntentResult(
            matched=True,
            intent=SqlCatalogReturnIntent(
                database=database,
                value=schemas,
                output_mode="text",
                metadata={
                    "sql_generation_mode": "deterministic_schema",
                    "sql_tables_used": [],
                    "sql_columns_used": [],
                    **constraint_telemetry(constraint_frame),
                },
            ),
        )
    if LIST_TABLES_RE.search(goal_text):
        names = catalog.table_names()
        return IntentResult(
            matched=True,
            intent=SqlCatalogReturnIntent(
                database=database,
                value=names,
                output_mode="text",
                metadata={
                    "sql_generation_mode": "deterministic_schema",
                    "sql_tables_used": names,
                    "sql_columns_used": [],
                    **constraint_telemetry(constraint_frame),
                },
            ),
        )

    describe = DESCRIBE_TABLE_RE.search(goal_text)
    if describe is not None:
        table = resolve_table_name(describe.group("table"), catalog) or resolve_table_name(goal_text, catalog)
        if table is None:
            return _failure("Could not identify the table to describe.", error_type="sql_table_not_found", database=database, catalog=catalog)
        rows = [
            {
                "schema_name": column.schema_name,
                "table_name": column.table_name,
                "column_name": column.column_name,
                "data_type": column.data_type,
                "nullable": column.nullable,
                "primary_key": column.primary_key,
                "foreign_key": column.foreign_key,
            }
            for column in table.columns
        ]
        return IntentResult(
            matched=True,
            intent=SqlCatalogReturnIntent(
                database=database,
                value=rows if _wants_json(goal_text) else _describe_table_text(table),
                output_mode="json" if _wants_json(goal_text) else "text",
                metadata={
                    "sql_generation_mode": "deterministic_schema",
                    "sql_tables_used": [table.qualified_name],
                    "sql_columns_used": [column.qualified_name for column in table.columns],
                    **constraint_telemetry(constraint_frame),
                },
            ),
        )

    if COUNT_RE.search(goal_text):
        count_result = _classify_constraint_count(catalog, constraint_frame)
        if count_result.matched:
            return count_result

    if constraint_frame.query_type == "select" or re.search(r"\b(?:list|show|return|select|get)\b", goal_text, re.IGNORECASE):
        select_result = _classify_constraint_select(catalog, constraint_frame)
        if select_result.matched:
            return select_result

    return IntentResult(matched=False, reason="sql_deterministic_no_match")


def _classify_llm_sql(goal: str, context: ClassificationContext, catalog: SqlSchemaCatalog, *, constraint_frame: SqlConstraintFrame) -> IntentResult:
    resolved = resolve_sql_references(goal, catalog)
    llm = LLMClient(context.settings)
    repair_context = _sql_repair_context(context.failure_context)
    attempts = 0
    if repair_context is not None:
        generation = repair_sql_for_error(
            llm=llm,
            goal=goal,
            catalog=catalog,
            original_sql=repair_context["query"],
            error_message=repair_context["error"],
            resolved_context=resolved,
            constraint_frame=constraint_frame,
        )
        attempts = 1
    else:
        generation = generate_sql_for_goal(llm=llm, goal=goal, catalog=catalog, resolved_context=resolved, constraint_frame=constraint_frame)

    while generation.matched and attempts < 2:
        try:
            explain_sql_query(context.settings, generation.normalized_sql or generation.sql or "", catalog.database)
            break
        except Exception as exc:  # noqa: BLE001
            attempts += 1
            generation = repair_sql_for_error(
                llm=llm,
                goal=goal,
                catalog=catalog,
                original_sql=generation.normalized_sql or generation.sql or "",
                error_message=str(exc),
                resolved_context=resolved,
                constraint_frame=constraint_frame,
            )

    if not generation.matched and generation.validation_failure and "constraint coverage" in generation.validation_failure.lower() and attempts < 1:
        attempts += 1
        generation = repair_sql_for_error(
            llm=llm,
            goal=goal,
            catalog=catalog,
            original_sql=generation.normalized_sql or generation.sql or "",
            error_message=generation.validation_failure,
            resolved_context=resolved,
            constraint_frame=constraint_frame,
        )

    if not generation.matched or not generation.normalized_sql:
        return _failure(
            generation.reason or "Could not generate safe SQL for that request.",
            error_type="sql_generation_failed",
            database=catalog.database,
            catalog=catalog,
            validation_failure=generation.validation_failure,
            llm_calls=max(1, attempts + 1),
            constraint_frame=constraint_frame,
        )

    output_mode = "count" if COUNT_RE.search(goal) and not re.search(r"\b(?:by|group(?:ed)?|per)\b", goal, re.IGNORECASE) else ("json" if _wants_json(goal) else "text")
    scalar_key = "count_value" if output_mode == "count" else None
    coverage = validate_sql_constraint_coverage(constraint_frame, generation.normalized_sql)
    return IntentResult(
        matched=True,
        intent=SqlGeneratedQueryIntent(
            database=catalog.database,
            query=generation.normalized_sql,
            output_mode=output_mode,
            scalar_key=scalar_key,
            metadata={
                "planning_mode": "sql_llm_generation",
                "sql_generation_mode": "llm_sql" if repair_context is None else "llm_sql_repair",
                "sql_repair_attempts": attempts,
                "sql_original": generation.sql,
                "sql_normalized": generation.normalized_sql,
                "sql_final": generation.normalized_sql,
                "sql_tables_used": generation.tables_used,
                "sql_columns_used": generation.columns_used,
                "sql_validation_failure": generation.validation_failure,
                "llm_calls": max(1, attempts + 1),
                "raw_planner_llm_calls": 0,
                **constraint_telemetry(constraint_frame, coverage),
            },
        ),
        metadata={"planning_mode": "sql_llm_generation", "llm_calls": max(1, attempts + 1), "raw_planner_llm_calls": 0},
    )


def _deterministic_select(
    database: str,
    query: str,
    tables_used: list[str],
    columns_used: list[str],
    *,
    constraint_frame: SqlConstraintFrame,
) -> IntentResult:
    validation = _validate_generated_sql(query, constraint_frame)
    if validation is None:
        return IntentResult(matched=False, reason="sql_constraint_coverage_failed")
    normalized_query, coverage = validation
    return IntentResult(
        matched=True,
        intent=SqlGeneratedQueryIntent(
            database=database,
            query=normalized_query,
            output_mode="json",
            metadata={
                "planning_mode": "deterministic_intent",
                "sql_generation_mode": "deterministic_sql",
                "sql_original": query,
                "sql_normalized": normalized_query,
                "sql_final": normalized_query,
                "sql_tables_used": tables_used,
                "sql_columns_used": columns_used,
                "sql_repair_attempts": 0,
                **constraint_telemetry(constraint_frame, coverage),
            },
        ),
    )


def _failure(
    message: str,
    *,
    error_type: str,
    database: str | None = None,
    catalog: SqlSchemaCatalog | None = None,
    validation_failure: str | None = None,
    llm_calls: int = 0,
    constraint_frame: SqlConstraintFrame | None = None,
) -> IntentResult:
    suggestions = _sql_suggestions(database=database, catalog=catalog)
    return IntentResult(
        matched=True,
        intent=SqlFailureIntent(
            message=message,
            suggestions=suggestions,
            metadata={
                "sql_generation_mode": "safe_failure",
                "sql_error_class": error_type,
                "sql_validation_failure": validation_failure,
                "llm_calls": llm_calls,
                "raw_planner_llm_calls": 0,
                **(constraint_telemetry(constraint_frame) if constraint_frame is not None else {}),
            },
        ),
        metadata={"planning_mode": "sql_safe_failure", "llm_calls": llm_calls, "raw_planner_llm_calls": 0},
    )


def _looks_like_sql_goal(goal: str, context: ClassificationContext) -> bool:
    if not context.settings.sql_databases and not context.settings.sql_database_url:
        return False
    text = str(goal or "")
    lowered = text.lower()
    try:
        database_name, _ = resolve_database_selection(context.settings, None)
        if database_name and re.search(rf"\b{re.escape(database_name.lower())}\b", lowered):
            return True
    except Exception:  # noqa: BLE001
        pass
    for name in context.settings.sql_databases:
        if re.search(rf"\b{re.escape(str(name).lower())}\b", lowered):
            return True
    if SQL_HINT_RE.search(text) and not re.search(r"\b(?:file|files|folder|directory|\.[a-z0-9]{1,8})\b", lowered):
        return True
    return False


def _requires_broad_sql(goal: str) -> bool:
    return bool(
        re.search(
            r"\b(?:join|group(?:ed)?|per|by|having|with\s+more\s+than|with\s+less\s+than|more\s+than\s+\d+\s+\w+|less\s+than\s+\d+\s+\w+)\b",
            str(goal or ""),
            re.IGNORECASE,
        )
    )


def _database_hint(goal: str, context: ClassificationContext) -> str | None:
    lowered = str(goal or "").lower()
    for name in sorted(context.settings.sql_databases, key=len, reverse=True):
        if re.search(rf"\b{re.escape(str(name).lower())}\b", lowered):
            return str(name)
    return context.settings.sql_default_database


def _classify_constraint_count(catalog: SqlSchemaCatalog, frame: SqlConstraintFrame) -> IntentResult:
    target = _table_from_qualified(frame.target_entity, catalog)
    if target is None:
        return IntentResult(matched=False, reason="sql_count_no_target")
    if frame.unresolved_projections:
        return IntentResult(matched=False, reason="sql_count_unresolved_projection")
    count_distinct_projection = [projection for projection in frame.projections if projection.aggregate == "count_distinct"]
    unsupported_projections = [projection for projection in frame.projections if projection.aggregate != "count_distinct"]
    if unsupported_projections or len(count_distinct_projection) > 1:
        return IntentResult(matched=False, reason="sql_count_unsupported_projection")
    unsupported = [constraint for constraint in frame.non_target_constraints if constraint.kind not in {"age_comparison", "related_row_count"}]
    if unsupported or frame.unresolved_constraints:
        return IntentResult(matched=False, reason="sql_count_uncovered_constraints")

    age_constraints = [constraint for constraint in frame.constraints if constraint.kind == "age_comparison"]
    related_constraints = [constraint for constraint in frame.constraints if constraint.kind == "related_row_count"]
    if len(related_constraints) > 1:
        return IntentResult(matched=False, reason="sql_count_multiple_related_constraints")

    where_clauses = [
        _age_predicate_sql(constraint, dialect=catalog.dialect, table_alias="p" if related_constraints else None)
        for constraint in age_constraints
    ]
    if any(clause is None for clause in where_clauses):
        return IntentResult(matched=False, reason="sql_count_unresolved_age")
    where = " AND ".join(str(clause) for clause in where_clauses if clause)

    tables_used = [target.qualified_name]
    columns_used = [str(constraint.resolved_column) for constraint in age_constraints if constraint.resolved_column]

    if count_distinct_projection and related_constraints:
        return IntentResult(matched=False, reason="sql_count_distinct_related_not_supported")

    if count_distinct_projection:
        projection = count_distinct_projection[0]
        if not projection.resolved_column:
            return IntentResult(matched=False, reason="sql_count_distinct_projection_unresolved")
        column_name = projection.resolved_column.split(".")[-1]
        query = (
            f"SELECT COUNT(DISTINCT {quote_pg_identifier(column_name)}) AS count_value "
            f"FROM {quote_pg_relation(target.schema_name, target.table_name)}"
        )
        if where:
            query = f"{query} WHERE {where}"
        columns_used.append(projection.resolved_column)
    elif related_constraints:
        related = related_constraints[0]
        related_table = _table_from_qualified(related.resolved_table, catalog)
        primary_column = str(related.metadata.get("primary_column") or "")
        related_column = str(related.metadata.get("related_column") or "")
        if related_table is None or not primary_column or not related_column:
            return IntentResult(matched=False, reason="sql_count_unresolved_related")
        operator = _sql_count_operator(str(related.operator or ""))
        if operator is None:
            return IntentResult(matched=False, reason="sql_count_unknown_related_operator")
        join_type = "LEFT JOIN" if related.operator == "eq" and int(related.value or 0) == 0 else "JOIN"
        where_sql = f"\n    WHERE {where}" if where else ""
        query = (
            "SELECT COUNT(*) AS count_value\n"
            "FROM (\n"
            f"    SELECT p.{quote_pg_identifier(primary_column)}\n"
            f"    FROM {quote_pg_relation(target.schema_name, target.table_name)} p\n"
            f"    {join_type} {quote_pg_relation(related_table.schema_name, related_table.table_name)} s "
            f"ON s.{quote_pg_identifier(related_column)} = p.{quote_pg_identifier(primary_column)}"
            f"{where_sql}\n"
            f"    GROUP BY p.{quote_pg_identifier(primary_column)}\n"
            f"    HAVING COUNT(s.{quote_pg_identifier(related_column)}) {operator} {int(related.value or 0)}\n"
            ") matched_rows"
        )
        tables_used.append(related_table.qualified_name)
        columns_used.extend([f"{target.qualified_name}.{primary_column}", f"{related_table.qualified_name}.{related_column}"])
    else:
        query = f"SELECT COUNT(*) AS count_value FROM {quote_pg_relation(target.schema_name, target.table_name)}"
        if where:
            query = f"{query} WHERE {where}"

    validation = _validate_generated_sql(query, frame, catalog)
    if validation is None:
        return IntentResult(matched=False, reason="sql_count_constraint_coverage_failed")
    normalized_query, coverage = validation
    return IntentResult(
        matched=True,
        intent=SqlGeneratedQueryIntent(
            database=catalog.database,
            query=normalized_query,
            output_mode="count",
            scalar_key="count_value",
            metadata={
                "planning_mode": "deterministic_intent",
                "sql_generation_mode": "deterministic_sql",
                "sql_original": query,
                "sql_normalized": normalized_query,
                "sql_final": normalized_query,
                "sql_tables_used": tables_used,
                "sql_columns_used": sorted(set(columns_used)),
                "sql_repair_attempts": 0,
                **constraint_telemetry(frame, coverage),
            },
        ),
    )


def _classify_constraint_select(catalog: SqlSchemaCatalog, frame: SqlConstraintFrame) -> IntentResult:
    target = _table_from_qualified(frame.target_entity, catalog)
    if target is None:
        return IntentResult(matched=False, reason="sql_select_no_target")
    if frame.unresolved_constraints or frame.unresolved_projections:
        return IntentResult(matched=False, reason="sql_select_unresolved_semantics")
    if any(projection.aggregate != "none" for projection in frame.projections):
        return IntentResult(matched=False, reason="sql_select_unsupported_projection_aggregate")
    unsupported = [
        constraint
        for constraint in frame.non_target_constraints
        if constraint.kind not in {"age_comparison", "related_row_count", "limit", "order_by"}
    ]
    if unsupported:
        return IntentResult(matched=False, reason="sql_select_unsupported_constraints")

    age_constraints = [constraint for constraint in frame.constraints if constraint.kind == "age_comparison"]
    related_constraints = [constraint for constraint in frame.constraints if constraint.kind == "related_row_count"]
    if len(related_constraints) > 1:
        return IntentResult(matched=False, reason="sql_select_multiple_related_constraints")
    if any(projection.resolved_table and projection.resolved_table != target.qualified_name for projection in frame.projections):
        return IntentResult(matched=False, reason="sql_select_projection_cross_table")

    limit = _limit_value(frame) or _extract_limit(frame.goal)
    tables_used = [target.qualified_name]
    projection_columns = [projection.resolved_column for projection in frame.projections if projection.resolved_column]
    columns_used = [str(column) for column in projection_columns]
    columns_used.extend(str(constraint.resolved_column) for constraint in age_constraints if constraint.resolved_column)

    if related_constraints:
        query_info = _projection_related_select_sql(
            catalog=catalog,
            target=target,
            related=related_constraints[0],
            projections=frame.projections,
            age_constraints=age_constraints,
            dialect=catalog.dialect,
        )
        if query_info is None:
            return IntentResult(matched=False, reason="sql_select_related_unresolved")
        query, related_table, related_columns = query_info
        tables_used.append(related_table.qualified_name)
        columns_used.extend(related_columns)
    else:
        select_clause = _projection_select_clause(frame.projections)
        if select_clause is None:
            select_clause = "*"
            columns_used.extend(column.qualified_name for column in target.columns)
        query = f"SELECT {select_clause} FROM {quote_pg_relation(target.schema_name, target.table_name)}"
        where_clauses = [_age_predicate_sql(constraint, dialect=catalog.dialect) for constraint in age_constraints]
        if any(clause is None for clause in where_clauses):
            return IntentResult(matched=False, reason="sql_select_unresolved_age")
        where = " AND ".join(str(clause) for clause in where_clauses if clause)
        if where:
            query = f"{query} WHERE {where}"

    order_by = _order_by_clause(frame, table_alias="p" if related_constraints else None)
    if order_by:
        query = f"{query} {order_by}"
    if any(projection.distinct for projection in frame.projections) and not order_by and not related_constraints:
        first_projection = frame.projections[0] if frame.projections else None
        if first_projection and first_projection.resolved_column:
            query = f"{query} ORDER BY {quote_pg_identifier(first_projection.resolved_column.split('.')[-1])}"
    if limit:
        query = f"{query} LIMIT {limit}"

    return _deterministic_select(
        catalog.database,
        query,
        tables_used,
        sorted(set(columns_used)),
        constraint_frame=frame,
    )


def _projection_related_select_sql(
    *,
    catalog: SqlSchemaCatalog,
    target: SqlTableRef,
    related: SqlConstraint,
    projections: list[SqlProjection],
    age_constraints: list[SqlConstraint],
    dialect: str | None,
) -> tuple[str, SqlTableRef, list[str]] | None:
    related_table = _table_from_qualified(related.resolved_table, catalog)
    primary_column = str(related.metadata.get("primary_column") or "")
    related_column = str(related.metadata.get("related_column") or "")
    if related_table is None:
        return None
    if not primary_column or not related_column:
        return None
    operator = _sql_count_operator(str(related.operator or ""))
    if operator is None:
        return None
    select_clause = _projection_select_clause(projections, table_alias="p")
    if select_clause is None:
        return None
    where_clauses = [_age_predicate_sql(constraint, dialect=dialect, table_alias="p") for constraint in age_constraints]
    if any(clause is None for clause in where_clauses):
        return None
    where = " AND ".join(str(clause) for clause in where_clauses if clause)
    where_sql = f"\nWHERE {where}" if where else ""
    join_type = "LEFT JOIN" if related.operator == "eq" and int(related.value or 0) == 0 else "JOIN"
    group_columns = [f'p.{quote_pg_identifier(primary_column)}']
    for projection in projections:
        if projection.resolved_column:
            group_column = f'p.{quote_pg_identifier(projection.resolved_column.split(".")[-1])}'
            if group_column not in group_columns:
                group_columns.append(group_column)
    query = (
        f"SELECT {select_clause}\n"
        f"FROM {quote_pg_relation(target.schema_name, target.table_name)} p\n"
        f"{join_type} {quote_pg_relation(related_table.schema_name, related_table.table_name)} s "
        f"ON s.{quote_pg_identifier(related_column)} = p.{quote_pg_identifier(primary_column)}"
        f"{where_sql}\n"
        f"GROUP BY {', '.join(group_columns)}\n"
        f"HAVING COUNT(s.{quote_pg_identifier(related_column)}) {operator} {int(related.value or 0)}"
    )
    return (
        query,
        related_table,
        [f"{target.qualified_name}.{primary_column}", f"{related_table.qualified_name}.{related_column}"],
    )


def _projection_select_clause(projections: list[SqlProjection], *, table_alias: str | None = None) -> str | None:
    if not projections:
        return None
    prefix = f"{table_alias}." if table_alias else ""
    expressions: list[str] = []
    for projection in projections:
        if not projection.resolved_column:
            return None
        expressions.append(f"{prefix}{quote_pg_identifier(projection.resolved_column.split('.')[-1])}")
    distinct = "DISTINCT " if any(projection.distinct for projection in projections) else ""
    return f"{distinct}{', '.join(expressions)}"


def _age_predicate_sql(constraint: SqlConstraint, *, dialect: str | None, table_alias: str | None = None) -> str | None:
    if not constraint.resolved_column:
        return None
    column_name = constraint.resolved_column.split(".")[-1]
    prefix = f"{table_alias}." if table_alias else ""
    column_sql = f"{prefix}{quote_pg_identifier(column_name)}"
    if constraint.operator == "between" and isinstance(constraint.value, dict):
        lower = int(constraint.value.get("lower") or 0)
        upper = int(constraint.value.get("upper") or 0)
        return (
            f"{column_sql} <= {_age_cutoff_sql(lower, dialect)} "
            f"AND {column_sql} >= {_age_cutoff_sql(upper, dialect)}"
        )
    operator = {
        "gt": "<",
        "gte": "<=",
        "lt": ">",
        "lte": ">=",
    }.get(str(constraint.operator or ""))
    if operator is None:
        return None
    return f"{column_sql} {operator} {_age_cutoff_sql(int(constraint.value or 0), dialect)}"


def _age_cutoff_sql(age: int, dialect: str | None) -> str:
    if str(dialect or "").lower() == "sqlite":
        return f"date('now', '-{age} years')"
    return f"CURRENT_DATE - INTERVAL '{age} years'"


def _sql_count_operator(operator: str) -> str | None:
    return {
        "eq": "=",
        "neq": "<>",
        "gt": ">",
        "gte": ">=",
        "lt": "<",
        "lte": "<=",
    }.get(operator)


def _validate_generated_sql(query: str, frame: SqlConstraintFrame, catalog: SqlSchemaCatalog | None = None) -> tuple[str, SqlConstraintCoverageResult] | None:
    normalized_query = normalize_pg_relation_quoting(query, catalog)
    validation = validate_read_only_sql(normalized_query)
    if not validation.valid:
        return None
    normalized_query = str(validation.normalized_sql or normalized_query)
    coverage = validate_sql_constraint_coverage(frame, normalized_query)
    if not coverage.valid:
        return None
    return normalized_query, coverage


def _table_from_qualified(qualified_name: str | None, catalog: SqlSchemaCatalog) -> SqlTableRef | None:
    if not qualified_name or "." not in qualified_name:
        return None
    schema, table = qualified_name.split(".", 1)
    return catalog.table_by_name(schema, table)


def _order_by_clause(frame: SqlConstraintFrame, *, table_alias: str | None = None) -> str | None:
    for constraint in frame.constraints:
        if constraint.kind != "order_by" or not constraint.resolved_column:
            continue
        column_name = constraint.resolved_column.split(".")[-1]
        prefix = f"{table_alias}." if table_alias else ""
        direction = "DESC" if str(constraint.value or "").lower() == "desc" else "ASC"
        return f"ORDER BY {prefix}{quote_pg_identifier(column_name)} {direction}"
    return None


def _limit_value(frame: SqlConstraintFrame) -> int | None:
    for constraint in frame.constraints:
        if constraint.kind == "limit" and constraint.value is not None:
            return min(max(int(constraint.value), 1), 500)
    return None


def _extract_limit(goal: str) -> int | None:
    match = LIMIT_RE.search(goal)
    if match is None:
        return None
    value = int(match.group("limit"))
    return min(max(value, 1), 500)


def _wants_json(goal: str) -> bool:
    return bool(re.search(r"\bjson\b", str(goal or ""), re.IGNORECASE))


def _describe_table_text(table: SqlTableRef) -> str:
    lines = [table.qualified_name]
    for column in table.columns:
        flags = []
        if column.primary_key:
            flags.append("primary key")
        if column.nullable is False:
            flags.append("not null")
        if column.foreign_key:
            flags.append(f"references {column.foreign_key}")
        suffix = f" ({', '.join(flags)})" if flags else ""
        lines.append(f"{column.column_name}: {column.data_type or 'unknown'}{suffix}")
    return "\n".join(lines)


def _sql_suggestions(*, database: str | None, catalog: SqlSchemaCatalog | None) -> list[str]:
    db = database or (catalog.database if catalog is not None else "database")
    suggestions = [f"List all tables in {db}."]
    table = catalog.tables[0] if catalog is not None and catalog.tables else None
    if table is not None:
        relation = quote_pg_relation(table.schema_name, table.table_name)
        suggestions.extend([f"Describe table {relation} in {db}.", f"Count rows in {relation} from {db}."])
    else:
        suggestions.append(f"List schemas in {db}.")
    return suggestions[:3]


def _sql_repair_context(failure_context: dict[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(failure_context, dict):
        return None
    if str(failure_context.get("failed_step") or "").strip().lower() != "sql.query":
        return None
    step = failure_context.get("step")
    if not isinstance(step, dict):
        return None
    args = step.get("args")
    if not isinstance(args, dict):
        return None
    query = str(args.get("query") or "").strip()
    error = str(failure_context.get("error") or "").strip()
    if not query or not error:
        return None
    return {"query": query, "error": error}
