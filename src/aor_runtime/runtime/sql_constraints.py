"""OpenFABRIC Runtime Module: aor_runtime.runtime.sql_constraints

Purpose:
    Extract and validate SQL semantic constraints from user prompts and generated SQL.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
from typing import Any, Literal

from aor_runtime.runtime.sql_catalog import SqlColumnRef, SqlSchemaCatalog, SqlTableRef
from aor_runtime.runtime.sql_resolver import resolve_column_name, resolve_table_name


SqlConstraintKind = Literal[
    "target_table",
    "age_comparison",
    "related_row_count",
    "projection",
    "group_by",
    "order_by",
    "limit",
    "distinct",
    "unknown",
]
SqlConstraintOperator = Literal["eq", "neq", "gt", "gte", "lt", "lte", "between"]
SqlQueryType = Literal["count", "select", "aggregate", "describe", "list_tables", "unknown"]
SqlProjectionAggregate = Literal["count", "count_distinct", "none"]


COUNT_RE = re.compile(r"\b(?:count|how\s+many|number\s+of)\b", re.IGNORECASE)
LIST_TABLES_RE = re.compile(r"\b(?:list|show|return)\s+(?:all\s+)?tables\b", re.IGNORECASE)
DESCRIBE_RE = re.compile(r"\b(?:describe|show\s+schema\s+for|show\s+columns\s+for|list\s+columns)\b", re.IGNORECASE)
AGE_STRICT_PREFIX_RE = re.compile(
    r"\b(?P<raw>(?P<direction>above|over|older\s+than|greater\s+than|below|under|younger\s+than|less\s+than)\s+"
    r"(?:age\s+)?(?P<age>\d{1,3})(?:\s*(?:years?|yrs?)?(?:\s+old|\s+of\s+age)?)?)\b",
    re.IGNORECASE,
)
AGE_STRICT_AGE_PREFIX_RE = re.compile(
    r"\b(?P<raw>age\s+(?P<direction>above|over|greater\s+than|below|under|less\s+than)\s+(?P<age>\d{1,3}))\b",
    re.IGNORECASE,
)
AGE_INCLUSIVE_PREFIX_RE = re.compile(
    r"\b(?P<raw>(?:at\s+least|minimum(?:\s+age)?|age\s+at\s+least)\s+(?P<age>\d{1,3})"
    r"(?:\s*(?:years?|yrs?)?(?:\s+old)?)?)(?=\s*(?:in|from|where|and|$|[?.!,;:]))",
    re.IGNORECASE,
)
AGE_INCLUSIVE_SUFFIX_RE = re.compile(
    r"\b(?P<raw>(?P<age>\d{1,3})\s+(?:and\s+above|or\s+above|and\s+older|or\s+older)(?:\s*(?:years?|yrs?)?)?)\b",
    re.IGNORECASE,
)
AGE_BETWEEN_RE = re.compile(
    r"\b(?P<raw>between\s+(?P<lower>\d{1,3})\s+and\s+(?P<upper>\d{1,3})(?:\s*(?:years?|yrs?)?(?:\s+old)?)?)\b",
    re.IGNORECASE,
)
RELATED_ROW_COUNT_RE = re.compile(
    r"\b(?P<raw>with\s+(?P<quantity>no|exactly\s+\d+|more\s+than\s+\d+|at\s+least\s+\d+|"
    r"fewer\s+than\s+\d+|less\s+than\s+\d+|at\s+most\s+\d+|\d+)\s+"
    r"(?P<subject>[A-Za-z_][\w ]*?)(?=\s+(?:in|from|where|and)\b|[?.!,;:]|$))",
    re.IGNORECASE,
)
WITH_SEGMENT_RE = re.compile(r"\bwith\s+([^?.!,;]+)", re.IGNORECASE)
DISTINCT_RE = re.compile(r"\b(?P<raw>distinct\s+(?:values\s+(?:of|from)\s+)?(?P<subject>[A-Za-z_][\w ]*?)(?=\s+(?:in|from|by)\b|$))", re.IGNORECASE)
GROUP_BY_RE = re.compile(
    r"\b(?P<raw>(?:group(?:ed)?\s+by|counts?(?:\s+[A-Za-z_][\w]*){0,4}\s+by|per)\s+"
    r"(?P<subject>[A-Za-z_][\w ]*?)(?=\s+(?:in|from|where)\b|$))",
    re.IGNORECASE,
)
LIMIT_RE = re.compile(r"\b(?P<raw>(?:top|first|limit|latest|oldest)\s+(?P<limit>\d{1,4}))\b", re.IGNORECASE)
COUNT_DISTINCT_RE = re.compile(
    r"\b(?P<raw>count\s+distinct\s+(?P<subject>.+?))(?=\s+(?:above|over|older|below|under|with|in|from|where|group|by)\b|[?.!,;:]|$)",
    re.IGNORECASE,
)
DISTINCT_PROJECTION_RE = re.compile(
    r"\b(?P<raw>(?:list|show|return|get|select)?\s*(?:the\s+)?distinct\s+"
    r"(?:values\s+(?:of|from)\s+)?(?P<subject>.+?))(?=\s+(?:above|over|older|below|under|with|in|from|where|group|by)\b|[?.!,;:]|$)",
    re.IGNORECASE,
)
TOP_PROJECTION_RE = re.compile(
    r"\b(?P<raw>(?:top|first|latest|oldest)\s+\d{1,4}\s+(?P<subject>.+?))"
    r"(?=\s+(?:above|over|older|below|under|with|in|from|where|group|by)\b|[?.!,;:]|$)",
    re.IGNORECASE,
)
SELECT_PROJECTION_RE = re.compile(
    r"\b(?P<verb>list|show|return|get|select)\s+(?P<subject>.+?)"
    r"(?=\s+(?:above|over|older|below|under|younger|with|in|from|where|group|by|as\s+(?:json|csv|text))\b|[?.!,;:]|$)",
    re.IGNORECASE,
)

PROJECTION_LEADING_STOPWORDS_RE = re.compile(r"^(?:the|all|of|values?|rows?|columns?)\s+", re.IGNORECASE)


@dataclass(frozen=True)
class SqlConstraint:
    """Represent sql constraint within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SqlConstraint.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.sql_constraints.SqlConstraint and related tests.
    """
    id: str
    kind: SqlConstraintKind
    raw_text: str
    subject: str | None = None
    operator: SqlConstraintOperator | None = None
    value: Any | None = None
    unit: str | None = None
    resolved_table: str | None = None
    resolved_column: str | None = None
    requires_join: bool = False
    covered: bool = False
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """To dict for SqlConstraint instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SqlConstraint.to_dict calls and related tests.
        """
        return {
            "id": self.id,
            "kind": self.kind,
            "raw_text": self.raw_text,
            "subject": self.subject,
            "operator": self.operator,
            "value": self.value,
            "unit": self.unit,
            "resolved_table": self.resolved_table,
            "resolved_column": self.resolved_column,
            "requires_join": self.requires_join,
            "covered": self.covered,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SqlProjection:
    """Represent sql projection within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SqlProjection.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.sql_constraints.SqlProjection and related tests.
    """
    id: str
    raw_text: str
    subject: str
    resolved_table: str | None = None
    resolved_column: str | None = None
    distinct: bool = False
    aggregate: SqlProjectionAggregate = "none"
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """To dict for SqlProjection instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SqlProjection.to_dict calls and related tests.
        """
        return {
            "id": self.id,
            "raw_text": self.raw_text,
            "subject": self.subject,
            "resolved_table": self.resolved_table,
            "resolved_column": self.resolved_column,
            "distinct": self.distinct,
            "aggregate": self.aggregate,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class SqlConstraintFrame:
    """Represent sql constraint frame within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SqlConstraintFrame.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.sql_constraints.SqlConstraintFrame and related tests.
    """
    query_type: SqlQueryType
    target_entity: str | None = None
    constraints: list[SqlConstraint] = field(default_factory=list)
    unresolved_constraints: list[SqlConstraint] = field(default_factory=list)
    projections: list[SqlProjection] = field(default_factory=list)
    unresolved_projections: list[SqlProjection] = field(default_factory=list)
    covered_constraint_ids: list[str] = field(default_factory=list)
    missing_constraint_ids: list[str] = field(default_factory=list)
    goal: str = ""

    @property
    def unresolved_constraint_ids(self) -> list[str]:
        """Unresolved constraint ids for SqlConstraintFrame instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed property value for callers that need this runtime fact.

        Used by:
            Used by planning, execution, validation, and presentation through SqlConstraintFrame.unresolved_constraint_ids calls and related tests.
        """
        return [constraint.id for constraint in self.unresolved_constraints]

    @property
    def non_target_constraints(self) -> list[SqlConstraint]:
        """Non target constraints for SqlConstraintFrame instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed property value for callers that need this runtime fact.

        Used by:
            Used by planning, execution, validation, and presentation through SqlConstraintFrame.non_target_constraints calls and related tests.
        """
        return [constraint for constraint in self.constraints if constraint.kind != "target_table"]

    def constraint_by_id(self, constraint_id: str) -> SqlConstraint | None:
        """Constraint by id for SqlConstraintFrame instances.

        Inputs:
            Receives constraint_id for this SqlConstraintFrame method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SqlConstraintFrame.constraint_by_id calls and related tests.
        """
        for constraint in self.constraints:
            if constraint.id == constraint_id:
                return constraint
        return None

    def projection_by_id(self, projection_id: str) -> SqlProjection | None:
        """Projection by id for SqlConstraintFrame instances.

        Inputs:
            Receives projection_id for this SqlConstraintFrame method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SqlConstraintFrame.projection_by_id calls and related tests.
        """
        for projection in self.projections:
            if projection.id == projection_id:
                return projection
        return None

    def to_dict(self) -> dict[str, Any]:
        """To dict for SqlConstraintFrame instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SqlConstraintFrame.to_dict calls and related tests.
        """
        return {
            "query_type": self.query_type,
            "target_entity": self.target_entity,
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "unresolved_constraints": [constraint.to_dict() for constraint in self.unresolved_constraints],
            "projections": [projection.to_dict() for projection in self.projections],
            "unresolved_projections": [projection.to_dict() for projection in self.unresolved_projections],
            "covered_constraint_ids": list(self.covered_constraint_ids),
            "missing_constraint_ids": list(self.missing_constraint_ids),
        }


@dataclass(frozen=True)
class SqlConstraintCoverageResult:
    """Represent sql constraint coverage result within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SqlConstraintCoverageResult.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.sql_constraints.SqlConstraintCoverageResult and related tests.
    """
    valid: bool
    covered_constraint_ids: list[str] = field(default_factory=list)
    missing_constraint_ids: list[str] = field(default_factory=list)
    covered_projection_ids: list[str] = field(default_factory=list)
    missing_projection_ids: list[str] = field(default_factory=list)
    reason: str | None = None


@dataclass(frozen=True)
class SqlJoinResolution:
    """Represent sql join resolution within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SqlJoinResolution.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.sql_constraints.SqlJoinResolution and related tests.
    """
    primary_table: SqlTableRef
    related_table: SqlTableRef
    primary_column: str
    related_column: str


def extract_sql_constraints(goal: str) -> SqlConstraintFrame:
    """Extract sql constraints for the surrounding runtime workflow.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints.extract_sql_constraints.
    """
    text = str(goal or "")
    constraints: list[SqlConstraint] = []
    projections: list[SqlProjection] = []
    used_spans: list[tuple[int, int]] = []

    def add(kind: SqlConstraintKind, raw_text: str, **kwargs: Any) -> None:
        """Add for the surrounding runtime workflow.

        Inputs:
            Receives kind, raw_text for this function; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints.add.
        """
        constraints.append(
            SqlConstraint(
                id=f"c{len(constraints) + 1}",
                kind=kind,
                raw_text=raw_text.strip(),
                **kwargs,
            )
        )

    def add_projection(raw_text: str, subject: str, **kwargs: Any) -> None:
        """Add projection for the surrounding runtime workflow.

        Inputs:
            Receives raw_text, subject for this function; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints.add_projection.
        """
        if _is_explicit_all_entity_subject(subject):
            return
        clean_subject = _clean_projection_subject(subject)
        if not clean_subject or _is_row_projection_subject(clean_subject):
            return
        projection = SqlProjection(
            id=f"p{len(projections) + 1}",
            raw_text=raw_text.strip(),
            subject=clean_subject,
            **kwargs,
        )
        duplicate = any(
            _normalize_name(existing.subject) == _normalize_name(projection.subject)
            and existing.distinct == projection.distinct
            and existing.aggregate == projection.aggregate
            for existing in projections
        )
        if not duplicate:
            projections.append(projection)

    for regex in (AGE_BETWEEN_RE, AGE_STRICT_AGE_PREFIX_RE, AGE_STRICT_PREFIX_RE, AGE_INCLUSIVE_PREFIX_RE, AGE_INCLUSIVE_SUFFIX_RE):
        for match in regex.finditer(text):
            if _overlaps(match.span(), used_spans):
                continue
            used_spans.append(match.span())
            if regex is AGE_BETWEEN_RE:
                lower = int(match.group("lower"))
                upper = int(match.group("upper"))
                if lower > upper:
                    lower, upper = upper, lower
                add(
                    "age_comparison",
                    match.group("raw"),
                    subject="age",
                    operator="between",
                    value={"lower": lower, "upper": upper},
                    unit="years",
                )
                continue
            direction = str(match.group("direction") if "direction" in match.groupdict() else "").lower()
            if regex in {AGE_INCLUSIVE_PREFIX_RE, AGE_INCLUSIVE_SUFFIX_RE}:
                operator: SqlConstraintOperator = "gte"
            elif direction in {"above", "over", "older than", "greater than"}:
                operator = "gt"
            else:
                operator = "lt"
            add(
                "age_comparison",
                match.group("raw"),
                subject="age",
                operator=operator,
                value=int(match.group("age")),
                unit="years",
            )

    for match in RELATED_ROW_COUNT_RE.finditer(text):
        if _overlaps(match.span(), used_spans):
            continue
        used_spans.append(match.span())
        operator, value = _parse_related_quantity(match.group("quantity"))
        add(
            "related_row_count",
            match.group("raw"),
            subject=_clean_subject(match.group("subject")),
            operator=operator,
            value=value,
            unit="rows",
            requires_join=True,
        )

    for match in GROUP_BY_RE.finditer(text):
        if _overlaps(match.span(), used_spans):
            continue
        used_spans.append(match.span())
        add("group_by", match.group("raw"), subject=_clean_subject(match.group("subject")))

    for match in LIMIT_RE.finditer(text):
        if _overlaps(match.span(), used_spans):
            continue
        used_spans.append(match.span())
        kind: SqlConstraintKind = "limit"
        raw = match.group("raw")
        add(kind, raw, subject=raw.split()[0].lower(), operator="eq", value=int(match.group("limit")), unit="rows")
        keyword = raw.split()[0].lower()
        if keyword in {"oldest", "latest"}:
            add(
                "order_by",
                raw,
                subject="age" if keyword == "oldest" else "date",
                operator="eq",
                value="asc" if keyword == "oldest" else "desc",
            )

    for match in COUNT_DISTINCT_RE.finditer(text):
        add_projection(
            match.group("raw"),
            match.group("subject"),
            distinct=True,
            aggregate="count_distinct",
        )

    if not COUNT_RE.search(text):
        for regex in (DISTINCT_PROJECTION_RE, TOP_PROJECTION_RE, SELECT_PROJECTION_RE):
            for match in regex.finditer(text):
                if regex is SELECT_PROJECTION_RE and _projection_subject_starts_with_count(match.group("subject")):
                    continue
                if regex is SELECT_PROJECTION_RE and re.match(r"\s*(?:the\s+)?distinct\b", match.group("subject"), re.IGNORECASE):
                    continue
                add_projection(
                    match.group("raw") if "raw" in match.groupdict() else match.group(0),
                    match.group("subject"),
                    distinct=regex is DISTINCT_PROJECTION_RE,
                )

    for match in WITH_SEGMENT_RE.finditer(text):
        if _overlaps(match.span(), used_spans):
            continue
        raw = match.group(0).strip()
        if raw:
            add("unknown", raw, subject=_clean_subject(match.group(1)), confidence=0.6)

    return SqlConstraintFrame(
        query_type=_infer_query_type(text),
        target_entity=None,
        constraints=constraints,
        unresolved_constraints=[constraint for constraint in constraints if constraint.kind == "unknown"],
        projections=projections,
        goal=text,
    )


def resolve_sql_constraints(frame: SqlConstraintFrame, catalog: SqlSchemaCatalog) -> SqlConstraintFrame:
    """Resolve sql constraints for the surrounding runtime workflow.

    Inputs:
        Receives frame, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints.resolve_sql_constraints.
    """
    target_table = resolve_table_name(frame.goal, catalog)
    resolved_projections, unresolved_projections = _resolve_sql_projections(frame.projections, target_table, catalog)
    resolved_projection_tables = {
        projection.resolved_table for projection in resolved_projections if projection.resolved_table
    }
    if len(resolved_projection_tables) == 1:
        projection_table = _table_from_qualified(next(iter(resolved_projection_tables)), catalog)
        if target_table is None or all(
            projection.resolved_table != target_table.qualified_name
            for projection in resolved_projections
            if projection.resolved_table
        ):
            target_table = projection_table

    resolved_constraints: list[SqlConstraint] = []
    unresolved: list[SqlConstraint] = []

    target_constraint: SqlConstraint | None = None
    if target_table is not None and frame.query_type in {"count", "select", "aggregate"}:
        target_constraint = SqlConstraint(
            id="target_table",
            kind="target_table",
            raw_text=target_table.table_name,
            subject=target_table.table_name,
            resolved_table=target_table.qualified_name,
        )
        resolved_constraints.append(target_constraint)

    for constraint in frame.constraints:
        resolved = constraint
        if constraint.kind == "unknown":
            unresolved.append(constraint)
            resolved_constraints.append(constraint)
            continue
        if constraint.kind == "age_comparison":
            if target_table is None:
                unresolved.append(constraint)
            else:
                birth_date = resolve_birth_date_column(target_table)
                if birth_date is None:
                    unresolved.append(constraint)
                else:
                    resolved = replace(
                        constraint,
                        resolved_table=target_table.qualified_name,
                        resolved_column=birth_date.qualified_name,
                    )
            resolved_constraints.append(resolved)
            continue
        if constraint.kind == "related_row_count":
            related_table = resolve_table_name(str(constraint.subject or ""), catalog)
            join = resolve_join_relationship(target_table, related_table) if target_table is not None and related_table is not None else None
            if target_table is None or related_table is None or join is None or related_table.qualified_name == target_table.qualified_name:
                unresolved.append(constraint)
            else:
                resolved = replace(
                    constraint,
                    resolved_table=related_table.qualified_name,
                    metadata={
                        **constraint.metadata,
                        "primary_table": target_table.qualified_name,
                        "primary_column": join.primary_column,
                        "related_table": related_table.qualified_name,
                        "related_column": join.related_column,
                    },
                )
            resolved_constraints.append(resolved)
            continue
        if constraint.kind in {"distinct", "group_by", "projection", "order_by"}:
            if target_table is None:
                unresolved.append(constraint)
            else:
                column = resolve_column_name(str(constraint.subject or ""), target_table)
                if column is None:
                    unresolved.append(constraint)
                else:
                    resolved = replace(
                        constraint,
                        resolved_table=target_table.qualified_name,
                        resolved_column=column.qualified_name,
                    )
            resolved_constraints.append(resolved)
            continue
        resolved_constraints.append(resolved)

    return SqlConstraintFrame(
        query_type=frame.query_type,
        target_entity=target_table.qualified_name if target_table is not None else frame.target_entity,
        constraints=resolved_constraints,
        unresolved_constraints=unresolved,
        projections=resolved_projections,
        unresolved_projections=unresolved_projections,
        covered_constraint_ids=list(frame.covered_constraint_ids),
        missing_constraint_ids=list(frame.missing_constraint_ids),
        goal=frame.goal,
    )


def resolve_birth_date_column(table: SqlTableRef) -> SqlColumnRef | None:
    """Resolve birth date column for the surrounding runtime workflow.

    Inputs:
        Receives table for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints.resolve_birth_date_column.
    """
    exact_priority = ["PatientBirthDate", "BirthDate", "DOB", "DateOfBirth"]
    for expected in exact_priority:
        column = table.column_by_name(expected)
        if column is not None:
            return column
    candidates: list[SqlColumnRef] = []
    for column in table.columns:
        normalized = _normalize_name(column.column_name)
        tokens = set(normalized.split())
        compact = normalized.replace(" ", "")
        if compact == "dob" or {"birth", "date"} <= tokens or "birthdate" in compact or "dateofbirth" in compact:
            candidates.append(column)
    return candidates[0] if len(candidates) == 1 else None


def resolve_join_relationship(primary_table: SqlTableRef | None, related_table: SqlTableRef | None) -> SqlJoinResolution | None:
    """Resolve join relationship for the surrounding runtime workflow.

    Inputs:
        Receives primary_table, related_table for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints.resolve_join_relationship.
    """
    if primary_table is None or related_table is None:
        return None
    for foreign_key in related_table.foreign_keys:
        if _fk_points_to(foreign_key, primary_table):
            local = _first_string(foreign_key.get("columns"))
            remote = _first_string(foreign_key.get("referred_columns"))
            if local and remote:
                return SqlJoinResolution(primary_table, related_table, remote, local)
    for foreign_key in primary_table.foreign_keys:
        if _fk_points_to(foreign_key, related_table):
            local = _first_string(foreign_key.get("columns"))
            remote = _first_string(foreign_key.get("referred_columns"))
            if local and remote:
                return SqlJoinResolution(primary_table, related_table, local, remote)

    primary_names = {column.column_name for column in primary_table.columns}
    related_names = {column.column_name for column in related_table.columns}
    shared = sorted(primary_names & related_names)
    if not shared:
        return None
    for column in primary_table.primary_key_columns:
        if column in shared:
            return SqlJoinResolution(primary_table, related_table, column, column)
    id_columns = [column for column in shared if column.lower().endswith("id")]
    if id_columns:
        return SqlJoinResolution(primary_table, related_table, id_columns[0], id_columns[0])
    return None


def validate_sql_constraint_coverage(frame: SqlConstraintFrame, sql: str) -> SqlConstraintCoverageResult:
    """Validate sql constraint coverage for the surrounding runtime workflow.

    Inputs:
        Receives frame, sql for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints.validate_sql_constraint_coverage.
    """
    constraints = list(frame.constraints)
    projections = list(frame.projections)
    if not constraints and not projections and not frame.unresolved_projections:
        return SqlConstraintCoverageResult(True, [], [], [], [])
    sql_text = str(sql or "")
    normalized = _coverage_normalize(sql_text)
    non_target = [constraint for constraint in constraints if constraint.kind != "target_table"]
    if (non_target or projections) and _is_plain_unfiltered_count(normalized):
        return SqlConstraintCoverageResult(
            False,
            [],
            [constraint.id for constraint in constraints],
            [],
            [projection.id for projection in projections],
            "Constrained prompt produced a plain unfiltered COUNT(*).",
        )

    covered: list[str] = []
    missing: list[str] = []
    for constraint in constraints:
        if constraint.kind == "target_table":
            ok = _covers_table(constraint, normalized)
        elif constraint.kind == "age_comparison":
            ok = _covers_age_constraint(constraint, normalized)
        elif constraint.kind == "related_row_count":
            ok = _covers_related_row_count(constraint, normalized)
        elif constraint.kind == "distinct":
            ok = "distinct" in normalized and _covers_column(constraint, normalized)
        elif constraint.kind == "group_by":
            ok = "group by" in normalized and _covers_column(constraint, normalized)
        elif constraint.kind == "order_by":
            ok = "order by" in normalized and _covers_column(constraint, normalized)
        elif constraint.kind == "limit":
            ok = re.search(rf"\blimit\s+{re.escape(str(constraint.value))}\b", normalized) is not None
        elif constraint.kind == "projection":
            ok = _covers_column(constraint, normalized)
        else:
            ok = False
        if ok:
            covered.append(constraint.id)
        else:
            missing.append(constraint.id)

    projection_covered: list[str] = []
    projection_missing: list[str] = [projection.id for projection in frame.unresolved_projections]
    for projection in projections:
        if projection in frame.unresolved_projections:
            continue
        ok = _covers_projection(projection, normalized)
        if ok:
            projection_covered.append(projection.id)
        else:
            projection_missing.append(projection.id)

    return SqlConstraintCoverageResult(
        valid=not missing and not projection_missing,
        covered_constraint_ids=covered,
        missing_constraint_ids=missing,
        covered_projection_ids=projection_covered,
        missing_projection_ids=projection_missing,
        reason=None
        if not missing and not projection_missing
        else "SQL is missing "
        + ", ".join(
            part
            for part in (
                f"constraints: {', '.join(missing)}" if missing else "",
                f"projections: {', '.join(projection_missing)}" if projection_missing else "",
            )
            if part
        )
        + ".",
    )


def constraint_telemetry(frame: SqlConstraintFrame, coverage: SqlConstraintCoverageResult | None = None) -> dict[str, Any]:
    """Constraint telemetry for the surrounding runtime workflow.

    Inputs:
        Receives frame, coverage for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints.constraint_telemetry.
    """
    covered = list(coverage.covered_constraint_ids if coverage is not None else frame.covered_constraint_ids)
    missing = list(coverage.missing_constraint_ids if coverage is not None else frame.missing_constraint_ids)
    projection_covered = list(coverage.covered_projection_ids if coverage is not None else [])
    projection_missing = list(coverage.missing_projection_ids if coverage is not None else [])
    unresolved = [constraint.id for constraint in frame.unresolved_constraints]
    unresolved_projection_ids = [projection.id for projection in frame.unresolved_projections]
    return {
        "sql_constraints_extracted": [constraint.to_dict() for constraint in frame.constraints],
        "sql_constraints_resolved": [
            constraint.to_dict()
            for constraint in frame.constraints
            if constraint.id not in set(unresolved) and constraint.kind != "unknown"
        ],
        "sql_constraints_covered": covered,
        "sql_constraints_missing": missing or unresolved,
        "sql_constraint_coverage_passed": bool(coverage.valid) if coverage is not None else False,
        "sql_projections_extracted": [projection.to_dict() for projection in frame.projections],
        "sql_projections_resolved": [
            projection.to_dict()
            for projection in frame.projections
            if projection.id not in set(unresolved_projection_ids) and projection.resolved_column
        ],
        "sql_projections_covered": projection_covered,
        "sql_projections_missing": projection_missing or unresolved_projection_ids,
        "sql_projection_coverage_passed": bool(coverage.valid and not (coverage.missing_projection_ids)) if coverage is not None else False,
    }


def _resolve_sql_projections(
    projections: list[SqlProjection],
    target_table: SqlTableRef | None,
    catalog: SqlSchemaCatalog,
) -> tuple[list[SqlProjection], list[SqlProjection]]:
    """Handle the internal resolve sql projections helper path for this module.

    Inputs:
        Receives projections, target_table, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._resolve_sql_projections.
    """
    resolved: list[SqlProjection] = []
    unresolved: list[SqlProjection] = []
    for projection in projections:
        column: SqlColumnRef | None = None
        if target_table is not None:
            column = resolve_column_name(projection.subject, target_table)
        if column is None:
            column = _resolve_unique_catalog_column(projection.subject, catalog)
        if column is None:
            unresolved.append(projection)
            resolved.append(projection)
            continue
        resolved_projection = replace(
            projection,
            resolved_table=f"{column.schema_name}.{column.table_name}",
            resolved_column=column.qualified_name,
        )
        resolved.append(resolved_projection)
    return resolved, unresolved


def _resolve_unique_catalog_column(subject: str, catalog: SqlSchemaCatalog) -> SqlColumnRef | None:
    """Handle the internal resolve unique catalog column helper path for this module.

    Inputs:
        Receives subject, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._resolve_unique_catalog_column.
    """
    matches: list[SqlColumnRef] = []
    for table in catalog.tables:
        column = resolve_column_name(subject, table)
        if column is not None:
            matches.append(column)
    unique = {(column.schema_name, column.table_name, column.column_name): column for column in matches}
    return next(iter(unique.values())) if len(unique) == 1 else None


def _parse_related_quantity(value: str) -> tuple[SqlConstraintOperator, int]:
    """Handle the internal parse related quantity helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._parse_related_quantity.
    """
    text = str(value or "").strip().lower()
    if text == "no":
        return "eq", 0
    number_match = re.search(r"\d+", text)
    number = int(number_match.group(0)) if number_match else 0
    if text.startswith("exactly"):
        return "eq", number
    if text.startswith("more than"):
        return "gt", number
    if text.startswith("at least"):
        return "gte", number
    if text.startswith(("fewer than", "less than")):
        return "lt", number
    if text.startswith("at most"):
        return "lte", number
    return "eq", number


def _infer_query_type(goal: str) -> SqlQueryType:
    """Handle the internal infer query type helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._infer_query_type.
    """
    if LIST_TABLES_RE.search(goal):
        return "list_tables"
    if DESCRIBE_RE.search(goal):
        return "describe"
    if GROUP_BY_RE.search(goal) or re.search(r"\b(?:group|grouped|per)\b", goal, re.IGNORECASE):
        return "aggregate"
    if COUNT_RE.search(goal):
        return "count"
    if TOP_PROJECTION_RE.search(goal):
        return "select"
    if re.search(r"\b(?:list|show|select|return|get)\b", goal, re.IGNORECASE):
        return "select"
    return "unknown"


def _table_from_qualified(qualified_name: str | None, catalog: SqlSchemaCatalog) -> SqlTableRef | None:
    """Handle the internal table from qualified helper path for this module.

    Inputs:
        Receives qualified_name, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._table_from_qualified.
    """
    if not qualified_name or "." not in qualified_name:
        return None
    schema, table = qualified_name.split(".", 1)
    return catalog.table_by_name(schema, table)


def _covers_table(constraint: SqlConstraint, normalized_sql: str) -> bool:
    """Handle the internal covers table helper path for this module.

    Inputs:
        Receives constraint, normalized_sql for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._covers_table.
    """
    if not constraint.resolved_table:
        return False
    return _normalized_relation(constraint.resolved_table) in normalized_sql


def _covers_column(constraint: SqlConstraint, normalized_sql: str) -> bool:
    """Handle the internal covers column helper path for this module.

    Inputs:
        Receives constraint, normalized_sql for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._covers_column.
    """
    if not constraint.resolved_column:
        return False
    column = constraint.resolved_column.split(".")[-1]
    return _column_matches_sql(column, normalized_sql)


def _covers_projection(projection: SqlProjection, normalized_sql: str) -> bool:
    """Handle the internal covers projection helper path for this module.

    Inputs:
        Receives projection, normalized_sql for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._covers_projection.
    """
    if not projection.resolved_column:
        return False
    column = projection.resolved_column.split(".")[-1]
    select_list = _select_list(normalized_sql)
    if projection.aggregate == "count_distinct":
        return (
            "count" in select_list
            and "distinct" in select_list
            and _column_matches_sql(column, select_list)
        )
    if projection.distinct and "select distinct" not in normalized_sql:
        return False
    return _column_matches_sql(column, select_list)


def _covers_age_constraint(constraint: SqlConstraint, normalized_sql: str) -> bool:
    """Handle the internal covers age constraint helper path for this module.

    Inputs:
        Receives constraint, normalized_sql for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._covers_age_constraint.
    """
    if not constraint.resolved_column:
        return False
    column = constraint.resolved_column.split(".")[-1]
    has_supported_date_cutoff = ("current_date" in normalized_sql and "interval" in normalized_sql) or (
        "date(" in normalized_sql and "now" in normalized_sql
    )
    if not _column_matches_sql(column, normalized_sql) or not has_supported_date_cutoff:
        return False
    if constraint.operator == "between" and isinstance(constraint.value, dict):
        lower = str(constraint.value.get("lower"))
        upper = str(constraint.value.get("upper"))
        return lower in normalized_sql and upper in normalized_sql and "<=" in normalized_sql and ">=" in normalized_sql
    expected = _age_birthdate_operator(str(constraint.operator or ""))
    return expected is not None and expected in normalized_sql and str(constraint.value) in normalized_sql


def _covers_related_row_count(constraint: SqlConstraint, normalized_sql: str) -> bool:
    """Handle the internal covers related row count helper path for this module.

    Inputs:
        Receives constraint, normalized_sql for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._covers_related_row_count.
    """
    if not constraint.resolved_table:
        return False
    if _normalized_relation(constraint.resolved_table) not in normalized_sql:
        return False
    if "count(" not in normalized_sql and "count (" not in normalized_sql:
        return False
    if "having" not in normalized_sql and not (constraint.operator == "eq" and constraint.value == 0 and "not exists" in normalized_sql):
        return False
    expected_operator = _sql_operator(str(constraint.operator or ""))
    if expected_operator is None:
        return False
    value = str(constraint.value)
    if expected_operator == "=":
        operator_present = re.search(rf"(?:=|is)\s+{re.escape(value)}\b", normalized_sql) is not None
    else:
        operator_present = re.search(rf"{re.escape(expected_operator)}\s*{re.escape(value)}\b", normalized_sql) is not None
    join_present = " join " in f" {normalized_sql} " or " exists " in f" {normalized_sql} "
    return bool(operator_present and join_present)


def _is_plain_unfiltered_count(normalized_sql: str) -> bool:
    """Handle the internal is plain unfiltered count helper path for this module.

    Inputs:
        Receives normalized_sql for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._is_plain_unfiltered_count.
    """
    return (
        re.fullmatch(r"select\s+count\s*\(\s*\*\s*\)(?:\s+as\s+\w+)?\s+from\s+[\w.]+", normalized_sql) is not None
        and " where " not in f" {normalized_sql} "
        and " join " not in f" {normalized_sql} "
        and " having " not in f" {normalized_sql} "
    )


def _coverage_normalize(sql: str) -> str:
    """Handle the internal coverage normalize helper path for this module.

    Inputs:
        Receives sql for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._coverage_normalize.
    """
    return re.sub(r"\s+", " ", str(sql or "").replace('"', "").strip().lower())


def _select_list(normalized_sql: str) -> str:
    """Handle the internal select list helper path for this module.

    Inputs:
        Receives normalized_sql for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._select_list.
    """
    match = re.search(r"\bselect\s+(?:distinct\s+)?(?P<select>.*?)\s+from\b", normalized_sql)
    return match.group("select") if match is not None else ""


def _coverage_compact(value: str) -> str:
    """Handle the internal coverage compact helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._coverage_compact.
    """
    return re.sub(r"[^a-z0-9]+", "", str(value or "").replace('"', "").lower())


def _column_matches_sql(column: str, normalized_sql: str) -> bool:
    """Handle the internal column matches sql helper path for this module.

    Inputs:
        Receives column, normalized_sql for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._column_matches_sql.
    """
    compact_sql = _coverage_compact(normalized_sql)
    compact_column = _coverage_compact(column)
    return bool(compact_column and (compact_column in compact_sql or _normalize_name(column) in normalized_sql))


def _normalized_relation(qualified_name: str) -> str:
    """Handle the internal normalized relation helper path for this module.

    Inputs:
        Receives qualified_name for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._normalized_relation.
    """
    return _coverage_normalize(qualified_name)


def _age_birthdate_operator(operator: str) -> str | None:
    """Handle the internal age birthdate operator helper path for this module.

    Inputs:
        Receives operator for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._age_birthdate_operator.
    """
    return {
        "gt": "<",
        "gte": "<=",
        "lt": ">",
        "lte": ">=",
    }.get(operator)


def _sql_operator(operator: str) -> str | None:
    """Handle the internal sql operator helper path for this module.

    Inputs:
        Receives operator for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._sql_operator.
    """
    return {
        "eq": "=",
        "neq": "<>",
        "gt": ">",
        "gte": ">=",
        "lt": "<",
        "lte": "<=",
    }.get(operator)


def _fk_points_to(foreign_key: dict[str, Any], table: SqlTableRef) -> bool:
    """Handle the internal fk points to helper path for this module.

    Inputs:
        Receives foreign_key, table for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._fk_points_to.
    """
    return (
        str(foreign_key.get("referred_schema") or "").lower() == table.schema_name.lower()
        and str(foreign_key.get("referred_table") or "").lower() == table.table_name.lower()
    )


def _first_string(value: Any) -> str | None:
    """Handle the internal first string helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._first_string.
    """
    if not isinstance(value, list) or not value:
        return None
    text = str(value[0] or "").strip()
    return text or None


def _clean_subject(value: str) -> str:
    """Handle the internal clean subject helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._clean_subject.
    """
    return re.sub(r"\s+", " ", str(value or "").strip())


def _clean_projection_subject(value: str) -> str:
    """Handle the internal clean projection subject helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._clean_projection_subject.
    """
    text = _clean_subject(value)
    text = re.sub(r"\b(?:in|from)\s+[A-Za-z_][\w.-]*\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bas\s+(?:json|csv|text)\s*$", "", text, flags=re.IGNORECASE)
    previous = None
    while previous != text:
        previous = text
        text = PROJECTION_LEADING_STOPWORDS_RE.sub("", text).strip()
    return _clean_subject(text)


def _is_row_projection_subject(subject: str) -> bool:
    """Handle the internal is row projection subject helper path for this module.

    Inputs:
        Receives subject for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._is_row_projection_subject.
    """
    normalized = _normalize_name(subject)
    if not normalized:
        return True
    return normalized in {"all", "all rows", "all row", "all columns", "all column", "rows", "columns", "records", "record"}


def _is_explicit_all_entity_subject(subject: str) -> bool:
    """Handle the internal is explicit all entity subject helper path for this module.

    Inputs:
        Receives subject for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._is_explicit_all_entity_subject.
    """
    tokens = _normalize_name(subject).split()
    return len(tokens) == 2 and tokens[0] == "all"


def _projection_subject_starts_with_count(subject: str) -> bool:
    """Handle the internal projection subject starts with count helper path for this module.

    Inputs:
        Receives subject for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._projection_subject_starts_with_count.
    """
    return bool(re.match(r"\s*(?:the\s+)?(?:count|number\s+of|how\s+many)\b", str(subject or ""), re.IGNORECASE))


def _normalize_name(value: str) -> str:
    """Handle the internal normalize name helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._normalize_name.
    """
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    """Handle the internal overlaps helper path for this module.

    Inputs:
        Receives span, spans for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.sql_constraints._overlaps.
    """
    return any(span[0] < existing[1] and existing[0] < span[1] for existing in spans)
