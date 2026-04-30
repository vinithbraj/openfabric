"""OpenFABRIC Runtime Module: aor_runtime.runtime.sql_resolver

Purpose:
    Resolve user SQL concepts to schema-backed tables, columns, and relationships.

Responsibilities:
    Map concepts such as patient, study, series, instance, modality, age, and DICOM relationships to catalog entities.

Data flow / Interfaces:
    Used by SQL repair, validation, and schema-grounded planning helpers.

Boundaries:
    Must return clean unsupported-concept answers instead of speculative invalid SQL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from difflib import SequenceMatcher

from aor_runtime.runtime.sql_catalog import SqlColumnRef, SqlSchemaCatalog, SqlTableRef


TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class ResolvedSqlContext:
    """Represent resolved sql context within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ResolvedSqlContext.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.sql_resolver.ResolvedSqlContext and related tests.
    """
    goal: str
    catalog: SqlSchemaCatalog
    tables: list[SqlTableRef] = field(default_factory=list)
    columns: list[SqlColumnRef] = field(default_factory=list)
    ambiguous_tables: list[SqlTableRef] = field(default_factory=list)
    ambiguous_columns: list[SqlColumnRef] = field(default_factory=list)
    error: str | None = None

    @property
    def has_ambiguity(self) -> bool:
        """Has ambiguity for ResolvedSqlContext instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed property value for callers that need this runtime fact.

        Used by:
            Used by planning, execution, validation, and presentation through ResolvedSqlContext.has_ambiguity calls and related tests.
        """
        return bool(self.ambiguous_tables or self.ambiguous_columns)


def resolve_table_name(user_phrase: str, catalog: SqlSchemaCatalog) -> SqlTableRef | None:
    """Resolve table name for the surrounding runtime workflow.

    Inputs:
        Receives user_phrase, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_resolver.resolve_table_name.
    """
    ambiguous_exact = _ambiguous_exact_table_matches(user_phrase, catalog)
    if ambiguous_exact:
        return None
    candidates = _rank_tables(user_phrase, catalog)
    if not candidates:
        return None
    best_score, best_table = candidates[0]
    if best_score < 6:
        return None
    tied = [table for score, table in candidates if score == best_score]
    return best_table if len(tied) == 1 else None


def resolve_column_name(user_phrase: str, table: SqlTableRef) -> SqlColumnRef | None:
    """Resolve column name for the surrounding runtime workflow.

    Inputs:
        Receives user_phrase, table for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_resolver.resolve_column_name.
    """
    candidates = _rank_columns(user_phrase, table)
    if not candidates:
        return None
    best_score, best_column = candidates[0]
    if best_score < 5:
        return None
    tied = [column for score, column in candidates if score == best_score]
    return best_column if len(tied) == 1 else None


def resolve_sql_references(goal: str, catalog: SqlSchemaCatalog) -> ResolvedSqlContext:
    """Resolve sql references for the surrounding runtime workflow.

    Inputs:
        Receives goal, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_resolver.resolve_sql_references.
    """
    ambiguous_exact = _ambiguous_exact_table_matches(goal, catalog)
    if ambiguous_exact:
        return ResolvedSqlContext(goal=goal, catalog=catalog, ambiguous_tables=ambiguous_exact)
    table_scores = _rank_tables(goal, catalog)
    if not table_scores:
        return ResolvedSqlContext(goal=goal, catalog=catalog, error="No matching table found.")

    selected: list[SqlTableRef] = []
    ambiguous: list[SqlTableRef] = []
    if table_scores[0][0] >= 6:
        best_score = table_scores[0][0]
        selected = [table for score, table in table_scores if score == best_score]
        if len(selected) > 1:
            ambiguous = selected
            selected = []
        else:
            for score, table in table_scores[1:4]:
                if score >= 6 and best_score - score <= 2:
                    selected.append(table)

    if not selected and not ambiguous:
        return ResolvedSqlContext(
            goal=goal,
            catalog=catalog,
            error="No matching table found.",
        )

    selected_columns: list[SqlColumnRef] = []
    ambiguous_columns: list[SqlColumnRef] = []
    for table in selected:
        column_scores = _rank_columns(goal, table)
        if not column_scores or column_scores[0][0] < 5:
            continue
        best_score = column_scores[0][0]
        best_columns = [column for score, column in column_scores if score == best_score]
        if len(best_columns) > 1:
            ambiguous_columns.extend(best_columns)
        else:
            selected_columns.append(best_columns[0])

    return ResolvedSqlContext(
        goal=goal,
        catalog=catalog,
        tables=selected,
        columns=selected_columns,
        ambiguous_tables=ambiguous,
        ambiguous_columns=ambiguous_columns,
    )


def _rank_tables(user_phrase: str, catalog: SqlSchemaCatalog) -> list[tuple[int, SqlTableRef]]:
    """Handle the internal rank tables helper path for this module.

    Inputs:
        Receives user_phrase, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_resolver._rank_tables.
    """
    phrase = str(user_phrase or "")
    phrase_tokens = _tokens(phrase)
    normalized_phrase = _normalize(phrase)
    scored: list[tuple[int, SqlTableRef]] = []
    for table in catalog.tables:
        score = 0
        table_name = table.table_name
        schema_name = table.schema_name
        table_norm = _normalize(table_name)
        singular_table = _singular(table_norm)
        schema_norm = _normalize(schema_name)
        qualified_norm = _normalize(f"{schema_name} {table_name}")
        qualified_dot = f"{schema_norm}.{table_norm}"

        if normalized_phrase == table_norm or normalized_phrase == singular_table:
            score += 25
        if table_norm in phrase_tokens or singular_table in phrase_tokens:
            score += 18
        if table_name.lower() in phrase.lower():
            score += 12
        if qualified_dot in phrase.lower().replace('"', ""):
            score += 30
        if schema_norm in phrase_tokens and (table_norm in phrase_tokens or singular_table in phrase_tokens):
            score += 16
        score += len(phrase_tokens & _tokens(qualified_norm)) * 4
        for column in table.columns:
            score += len(phrase_tokens & _tokens(column.column_name)) * 1
        if score == 0:
            ratio = SequenceMatcher(None, normalized_phrase, table_norm).ratio()
            if ratio >= 0.88:
                score += 8
        if score > 0:
            scored.append((score, table))
    scored.sort(key=lambda item: (-item[0], item[1].qualified_name.lower()))
    return scored


def _ambiguous_exact_table_matches(user_phrase: str, catalog: SqlSchemaCatalog) -> list[SqlTableRef]:
    """Handle the internal ambiguous exact table matches helper path for this module.

    Inputs:
        Receives user_phrase, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_resolver._ambiguous_exact_table_matches.
    """
    phrase_tokens = _tokens(user_phrase)
    matches: dict[str, list[SqlTableRef]] = {}
    for table in catalog.tables:
        table_norm = _normalize(table.table_name)
        if table_norm in phrase_tokens or _singular(table_norm) in phrase_tokens:
            matches.setdefault(table_norm, []).append(table)
    for candidates in matches.values():
        if len(candidates) > 1:
            return sorted(candidates, key=lambda table: table.qualified_name.lower())
    return []


def _rank_columns(user_phrase: str, table: SqlTableRef) -> list[tuple[int, SqlColumnRef]]:
    """Handle the internal rank columns helper path for this module.

    Inputs:
        Receives user_phrase, table for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_resolver._rank_columns.
    """
    phrase = str(user_phrase or "")
    phrase_tokens = _tokens(phrase)
    normalized_phrase = _normalize(phrase)
    singular_phrase = _normalize_singular(phrase)
    compact_singular_phrase = singular_phrase.replace(" ", "")
    scored: list[tuple[int, SqlColumnRef]] = []
    for column in table.columns:
        score = 0
        column_norm = _normalize(column.column_name)
        singular_column = _normalize_singular(column.column_name)
        column_tokens = _tokens(column.column_name)
        if singular_phrase == singular_column:
            score += 30
        if compact_singular_phrase and compact_singular_phrase == singular_column.replace(" ", ""):
            score += 20
        if normalized_phrase == column_norm:
            score += 20
        if column_norm in phrase_tokens:
            score += 14
        if column.column_name.lower() in phrase.lower():
            score += 12
        score += len(phrase_tokens & column_tokens) * 5
        score += _semantic_column_score(phrase_tokens, column)
        if score == 0:
            ratio = SequenceMatcher(None, normalized_phrase, column_norm).ratio()
            if ratio >= 0.9:
                score += 7
        if score > 0:
            scored.append((score, column))
    scored.sort(key=lambda item: (-item[0], item[1].column_name.lower()))
    return scored


def _semantic_column_score(phrase_tokens: set[str], column: SqlColumnRef) -> int:
    """Handle the internal semantic column score helper path for this module.

    Inputs:
        Receives phrase_tokens, column for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_resolver._semantic_column_score.
    """
    column_tokens = _tokens(column.column_name)
    score = 0
    if "age" in phrase_tokens and {"birth", "date"} <= column_tokens:
        score += 16
    if {"birth", "date"} <= phrase_tokens and {"birth", "date"} <= column_tokens:
        score += 12
    if "name" in phrase_tokens and "name" in column_tokens:
        score += 8
    if "id" in phrase_tokens and "id" in column_tokens:
        score += 8
    return score


def _tokens(value: str) -> set[str]:
    """Handle the internal tokens helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_resolver._tokens.
    """
    normalized = _normalize(value)
    raw_tokens = set(TOKEN_RE.findall(normalized))
    expanded: set[str] = set()
    for token in raw_tokens:
        if token == "ids":
            expanded.add("id")
            continue
        expanded.add(token)
        expanded.add(_singular(token))
    return {token for token in expanded if token}


def _normalize(value: str) -> str:
    """Handle the internal normalize helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_resolver._normalize.
    """
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
    return " ".join(TOKEN_RE.findall(text.lower()))


def _normalize_singular(value: str) -> str:
    """Handle the internal normalize singular helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_resolver._normalize_singular.
    """
    tokens = []
    for token in TOKEN_RE.findall(_normalize(value)):
        tokens.append(_singular(token))
    return " ".join(token for token in tokens if token)


def _singular(token: str) -> str:
    """Handle the internal singular helper path for this module.

    Inputs:
        Receives token for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_resolver._singular.
    """
    if token == "ids":
        return "id"
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token
