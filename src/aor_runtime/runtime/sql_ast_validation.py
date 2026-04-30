"""OpenFABRIC Runtime Module: aor_runtime.runtime.sql_ast_validation

Purpose:
    Validate SQL syntax and schema references using AST-aware analysis.

Responsibilities:
    Check read-only structure, alias scopes, identifier quoting, ambiguous columns, wrong-table columns, and repair hints.

Data flow / Interfaces:
    Consumes SQL text plus catalog metadata before sql.query or sql.validate execution.

Boundaries:
    Prevents PostgreSQL runtime errors and unsafe SQL from becoming user-visible failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from aor_runtime.runtime.sql_catalog import SqlSchemaCatalog, SqlTableRef
from aor_runtime.runtime.sql_safety import normalize_pg_relation_quoting, quote_pg_identifier


@dataclass(frozen=True)
class SqlScopeIssue:
    """Represent sql scope issue within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SqlScopeIssue.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.sql_ast_validation.SqlScopeIssue and related tests.
    """
    issue_type: str
    column: str | None = None
    alias: str | None = None
    table: str | None = None
    scope_tables: tuple[str, ...] = ()
    candidate_columns: tuple[str, ...] = ()
    suggested_reference: str | None = None

    def message(self) -> str:
        """Message for SqlScopeIssue instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SqlScopeIssue.message calls and related tests.
        """
        target = f"{self.alias}.{self.column}" if self.alias and self.column else self.column or self.table or "SQL"
        if self.issue_type == "ambiguous_column":
            suffix = f" Scope tables: {', '.join(self.scope_tables)}." if self.scope_tables else ""
            suggestion = f" Qualify it, for example {self.suggested_reference}." if self.suggested_reference else ""
            return f"SQL references ambiguous column: {target}.{suffix}{suggestion}"
        if self.issue_type == "unknown_table":
            candidates = f" Candidate tables: {', '.join(self.candidate_columns[:8])}." if self.candidate_columns else ""
            return f"SQL references unknown table: {target}.{candidates}"
        if self.issue_type == "unknown_column":
            candidates = f" Candidate columns: {', '.join(self.candidate_columns[:8])}." if self.candidate_columns else ""
            suggestion = f" Suggested reference: {self.suggested_reference}." if self.suggested_reference else ""
            if self.alias and self.table and self.column:
                return (
                    f"SQL references unknown column: {target}. "
                    f"Column {self.column} is not on table {self.table}.{candidates}{suggestion}"
                )
            return f"SQL references unknown column: {target}.{candidates}{suggestion}"
        return f"SQL scope validation failed: {target}."


@dataclass(frozen=True)
class SqlAstValidationResult:
    """Represent sql ast validation result within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SqlAstValidationResult.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.sql_ast_validation.SqlAstValidationResult and related tests.
    """
    valid: bool
    normalized_sql: str
    issues: tuple[SqlScopeIssue, ...] = ()
    parse_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def messages(self) -> list[str]:
        """Messages for SqlAstValidationResult instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed property value for callers that need this runtime fact.

        Used by:
            Used by planning, execution, validation, and presentation through SqlAstValidationResult.messages calls and related tests.
        """
        if self.parse_error:
            return [self.parse_error]
        return [issue.message() for issue in self.issues]


def normalize_and_validate_sql_ast(sql: str, catalog: SqlSchemaCatalog) -> SqlAstValidationResult:
    """Normalize and validate sql ast for the surrounding runtime workflow.

    Inputs:
        Receives sql, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_ast_validation.normalize_and_validate_sql_ast.
    """
    normalized = normalize_pg_relation_quoting(sql, catalog)
    result = validate_sql_ast_scope(normalized, catalog)
    if not result.valid:
        return result
    return SqlAstValidationResult(True, normalized_sql=normalized, metadata=result.metadata)


def validate_sql_ast_scope(sql: str, catalog: SqlSchemaCatalog) -> SqlAstValidationResult:
    """Validate sql ast scope for the surrounding runtime workflow.

    Inputs:
        Receives sql, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_ast_validation.validate_sql_ast_scope.
    """
    dialect = _sqlglot_dialect(catalog)
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception as exc:  # noqa: BLE001
        return SqlAstValidationResult(False, normalized_sql=str(sql or ""), parse_error=f"SQL parse failed: {exc}")

    issues: list[SqlScopeIssue] = []
    cte_names = {str(cte.alias or "").lower() for cte in tree.find_all(exp.CTE) if str(cte.alias or "").strip()}
    for table in tree.find_all(exp.Table):
        table_ref = _table_ref_for_expression(table, catalog)
        if table_ref is None and not _is_pseudo_relation(table) and str(table.name or "").lower() not in cte_names:
            issues.append(_unknown_table_issue(table, catalog))

    for scope in traverse_scope(tree):
        alias_tables = _scope_alias_tables(scope, catalog)
        if not alias_tables:
            continue
        scope_tables = tuple(sorted({table.qualified_name for table in alias_tables.values()}))
        output_aliases = _scope_output_aliases(scope.expression)

        for column in scope.columns:
            alias = str(column.table or "").strip()
            if not alias or alias.lower() not in alias_tables:
                continue
            issue = _validate_qualified_column(column, alias_tables[alias.lower()], alias=alias, catalog=catalog, scope_tables=scope_tables)
            if issue is not None:
                issues.append(issue)

        for column in scope.unqualified_columns:
            name = str(column.name or "").strip()
            if not name or name in output_aliases:
                continue
            owners = _owners_for_column(name, alias_tables)
            if len(owners) == 1:
                continue
            if len(owners) > 1:
                suggested = _suggest_reference(owners[0][0], _canonical_column_name(name, owners[0][1]) or name)
                issues.append(
                    SqlScopeIssue(
                        issue_type="ambiguous_column",
                        column=name,
                        scope_tables=scope_tables,
                        candidate_columns=tuple(_scope_column_names(alias_tables)),
                        suggested_reference=suggested,
                    )
                )
            else:
                owners_elsewhere = _catalog_owners_for_column(name, catalog)
                suggested = _suggest_reference(owners_elsewhere[0][0], owners_elsewhere[0][2]) if owners_elsewhere else None
                candidates = tuple(_scope_column_names(alias_tables) or _catalog_column_names(catalog))
                issues.append(
                    SqlScopeIssue(
                        issue_type="unknown_column",
                        column=name,
                        scope_tables=scope_tables,
                        candidate_columns=candidates,
                        suggested_reference=suggested,
                    )
                )

    unique: dict[str, SqlScopeIssue] = {}
    for issue in issues:
        unique.setdefault(issue.message(), issue)
    clean_issues = tuple(unique.values())
    return SqlAstValidationResult(
        valid=not clean_issues,
        normalized_sql=str(sql or "").strip(),
        issues=clean_issues,
        metadata={"sql_ast_scope_issues": [issue.__dict__ for issue in clean_issues]} if clean_issues else {},
    )


def _validate_qualified_column(
    column: exp.Column,
    table: SqlTableRef,
    *,
    alias: str,
    catalog: SqlSchemaCatalog,
    scope_tables: tuple[str, ...],
) -> SqlScopeIssue | None:
    """Handle the internal validate qualified column helper path for this module.

    Inputs:
        Receives column, table, alias, catalog, scope_tables for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_ast_validation._validate_qualified_column.
    """
    name = str(column.name or "").strip()
    if not name:
        return None
    if table.column_by_name(name) is not None:
        return None
    table_columns = tuple(sorted(column_ref.column_name for column_ref in table.columns))
    owners_elsewhere = _catalog_owners_for_column(name, catalog)
    suggested = _suggest_reference(owners_elsewhere[0][0], owners_elsewhere[0][2]) if owners_elsewhere else None
    return SqlScopeIssue(
        issue_type="unknown_column",
        alias=alias,
        column=name,
        table=table.qualified_name,
        scope_tables=scope_tables,
        candidate_columns=table_columns,
        suggested_reference=suggested,
    )


def _scope_alias_tables(scope: Any, catalog: SqlSchemaCatalog) -> dict[str, SqlTableRef]:
    """Handle the internal scope alias tables helper path for this module.

    Inputs:
        Receives scope, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_ast_validation._scope_alias_tables.
    """
    alias_tables: dict[str, SqlTableRef] = {}
    for alias, source_pair in dict(scope.selected_sources).items():
        source = source_pair[1] if isinstance(source_pair, tuple) and len(source_pair) >= 2 else source_pair
        if not isinstance(source, exp.Table):
            continue
        table_ref = _table_ref_for_expression(source, catalog)
        if table_ref is not None:
            alias_tables[str(alias).lower()] = table_ref
            alias_tables.setdefault(table_ref.table_name.lower(), table_ref)
    return alias_tables


def _table_ref_for_expression(table: exp.Table, catalog: SqlSchemaCatalog) -> SqlTableRef | None:
    """Handle the internal table ref for expression helper path for this module.

    Inputs:
        Receives table, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_ast_validation._table_ref_for_expression.
    """
    name = str(table.name or "").strip()
    schema = str(table.db or "").strip() or None
    if "." in name and not schema:
        schema, name = name.split(".", 1)
    return catalog.table_by_name(schema, name)


def _unknown_table_issue(table: exp.Table, catalog: SqlSchemaCatalog) -> SqlScopeIssue:
    """Handle the internal unknown table issue helper path for this module.

    Inputs:
        Receives table, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_ast_validation._unknown_table_issue.
    """
    name = str(table.sql(dialect=_sqlglot_dialect(catalog)) or table.name)
    return SqlScopeIssue(
        issue_type="unknown_table",
        table=name,
        candidate_columns=tuple(catalog.table_names()),
    )


def _owners_for_column(column: str, alias_tables: dict[str, SqlTableRef]) -> list[tuple[str, SqlTableRef]]:
    """Handle the internal owners for column helper path for this module.

    Inputs:
        Receives column, alias_tables for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_ast_validation._owners_for_column.
    """
    owners: list[tuple[str, SqlTableRef]] = []
    seen_tables: set[str] = set()
    for alias, table in alias_tables.items():
        if alias == table.table_name.lower() and any(existing[1].qualified_name == table.qualified_name for existing in owners):
            continue
        if table.column_by_name(column) is None:
            continue
        key = f"{alias}:{table.qualified_name}"
        if key in seen_tables:
            continue
        seen_tables.add(key)
        owners.append((alias, table))
    return owners


def _catalog_owners_for_column(column: str, catalog: SqlSchemaCatalog) -> list[tuple[str, SqlTableRef, str]]:
    """Handle the internal catalog owners for column helper path for this module.

    Inputs:
        Receives column, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_ast_validation._catalog_owners_for_column.
    """
    owners: list[tuple[str, SqlTableRef, str]] = []
    for table in catalog.tables:
        column_ref = table.column_by_name(column)
        if column_ref is not None:
            owners.append((table.table_name.lower(), table, column_ref.column_name))
    return owners


def _canonical_column_name(column: str, table: SqlTableRef) -> str | None:
    """Handle the internal canonical column name helper path for this module.

    Inputs:
        Receives column, table for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_ast_validation._canonical_column_name.
    """
    column_ref = table.column_by_name(column)
    return column_ref.column_name if column_ref is not None else None


def _scope_column_names(alias_tables: dict[str, SqlTableRef]) -> list[str]:
    """Handle the internal scope column names helper path for this module.

    Inputs:
        Receives alias_tables for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_ast_validation._scope_column_names.
    """
    names = sorted({column.column_name for table in alias_tables.values() for column in table.columns})
    return names


def _catalog_column_names(catalog: SqlSchemaCatalog) -> list[str]:
    """Handle the internal catalog column names helper path for this module.

    Inputs:
        Receives catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_ast_validation._catalog_column_names.
    """
    return sorted({column.column_name for table in catalog.tables for column in table.columns})


def _scope_output_aliases(expression: exp.Expression) -> set[str]:
    """Handle the internal scope output aliases helper path for this module.

    Inputs:
        Receives expression for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_ast_validation._scope_output_aliases.
    """
    if not isinstance(expression, exp.Select):
        return set()
    aliases: set[str] = set()
    for item in expression.expressions:
        alias = str(item.alias or "").strip()
        if alias:
            aliases.add(alias)
    return aliases


def _suggest_reference(alias: str, column: str) -> str:
    """Handle the internal suggest reference helper path for this module.

    Inputs:
        Receives alias, column for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_ast_validation._suggest_reference.
    """
    return f"{alias}.{quote_pg_identifier(column)}"


def _is_pseudo_relation(table: exp.Table) -> bool:
    """Handle the internal is pseudo relation helper path for this module.

    Inputs:
        Receives table for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_ast_validation._is_pseudo_relation.
    """
    return str(table.name or "").lower() in {"unnest"}


def _sqlglot_dialect(catalog: SqlSchemaCatalog) -> str:
    """Handle the internal sqlglot dialect helper path for this module.

    Inputs:
        Receives catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_ast_validation._sqlglot_dialect.
    """
    dialect = str(catalog.dialect or "").lower()
    return "postgres" if dialect == "postgresql" else dialect or "postgres"
