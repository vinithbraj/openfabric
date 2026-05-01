"""OpenFABRIC Runtime Module: aor_runtime.runtime.semantic.action_matrix

Purpose:
    Declare the canonical semantic action matrix that bridges typed LLM meaning to safe runtime tools.

Responsibilities:
    Store read-only and disabled mutating semantic action rows, match canonical frames to rows, and expose testable matrix facts.

Data flow / Interfaces:
    Consumed by semantic extraction prompts, semantic compilers, matrix tests, and coverage validation.

Boundaries:
    Contains semantic capability metadata only; domain compilers still own tool-argument lowering and safety validators own execution approval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from aor_runtime.runtime.semantic.ontology import (
    normalize_semantic_dimension,
    normalize_semantic_entity,
    normalize_semantic_filter_field,
    normalize_semantic_token,
)

SemanticActionStatus = Literal["active_read_only", "disabled_mutating"]
SemanticSafetyTier = Literal["read_only", "mutating_disabled"]
SemanticOutputKind = Literal["scalar", "table", "file", "text", "json", "status", "unknown"]
SemanticOutputCardinality = Literal["single", "multi_scalar", "grouped", "collection", "sectioned", "unknown"]
SemanticRenderStyle = Literal["scalar", "metric_table", "record_table", "key_value", "bullets", "sectioned", "unknown"]


@dataclass(frozen=True)
class SemanticActionOutput:
    """Describe the output contract attached to one semantic action row.

    Inputs:
        Built as static matrix metadata.

    Returns:
        A compact output shape used by compilers and tests.

    Used by:
        SemanticActionRow and semantic matrix validation tests.
    """

    kind: SemanticOutputKind
    cardinality: SemanticOutputCardinality
    render_style: SemanticRenderStyle


@dataclass(frozen=True)
class SemanticActionRow:
    """Represent one user-meaning action that the runtime can or cannot lower.

    Inputs:
        Built from domain semantic facts and registered tool names.

    Returns:
        Immutable matrix metadata for matching and compiler dispatch.

    Used by:
        SemanticFrameCompiler, semantic-frame prompts, and capability matrix tests.
    """

    semantic_action_id: str
    domain: str
    intent: str
    entities: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    filters: tuple[str, ...] = ()
    target_dimensions: tuple[str, ...] = ()
    grouping_dimensions: tuple[str, ...] = ()
    output: SemanticActionOutput = field(default_factory=lambda: SemanticActionOutput("unknown", "unknown", "unknown"))
    strategy: str = "unsupported"
    tools: tuple[str, ...] = ()
    status: SemanticActionStatus = "active_read_only"
    safety_tier: SemanticSafetyTier = "read_only"
    compiler_owner: str = ""

    @property
    def active(self) -> bool:
        """Return whether this matrix row is active for semantic lowering.

        Inputs:
            Uses the row status.

        Returns:
            True for active read-only rows.

        Used by:
            Compiler dispatch and matrix tests.
        """
        return self.status == "active_read_only"


@dataclass(frozen=True)
class SemanticActionSelection:
    """Describe an explainable semantic matrix selection decision.

    Inputs:
        Built from a semantic frame and candidate matrix rows.

    Returns:
        Selected row, score, matched facts, and rejected candidate reasons.

    Used by:
        select_semantic_action, compiler dispatch, coverage validation, and tests.
    """

    row: SemanticActionRow | None
    score: int | None = None
    matched_entities: tuple[str, ...] = ()
    matched_filters: tuple[str, ...] = ()
    matched_dimensions: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    expected_output: SemanticActionOutput | None = None
    rejected_candidates: tuple[str, ...] = ()

    @property
    def selected(self) -> bool:
        """Return whether the decision selected an active matrix row.

        Inputs:
            Uses the decision's row field.

        Returns:
            True when a row was selected.

        Used by:
            Tests and compiler diagnostics.
        """
        return self.row is not None


def active_semantic_actions() -> tuple[SemanticActionRow, ...]:
    """Return all active read-only matrix rows.

    Inputs:
        Uses module-level matrix metadata.

    Returns:
        Tuple of active semantic action rows.

    Used by:
        Semantic-frame prompt construction and matrix tests.
    """
    return tuple(row for row in SEMANTIC_ACTION_MATRIX if row.active)


def disabled_semantic_actions() -> tuple[SemanticActionRow, ...]:
    """Return matrix rows that are documented but unavailable to semantic lowering.

    Inputs:
        Uses module-level matrix metadata.

    Returns:
        Tuple of disabled mutating semantic action rows.

    Used by:
        Matrix tests and future policy work.
    """
    return tuple(row for row in SEMANTIC_ACTION_MATRIX if not row.active)


def semantic_action_prompt_metadata() -> list[dict[str, Any]]:
    """Return safe semantic action metadata for LLM prompt guidance.

    Inputs:
        Uses active matrix rows only.

    Returns:
        JSON-safe facts without tools args, rows, PHI, or operational payloads.

    Used by:
        build_semantic_frame_prompt.
    """
    return [
        {
            "semantic_action_id": row.semantic_action_id,
            "domain": row.domain,
            "intent": row.intent,
            "entities": list(row.entities),
            "metrics": list(row.metrics),
            "filters": list(row.filters),
            "target_dimensions": list(row.target_dimensions),
            "grouping_dimensions": list(row.grouping_dimensions),
            "output": {
                "kind": row.output.kind,
                "cardinality": row.output.cardinality,
                "render_style": row.output.render_style,
            },
            "strategy": row.strategy,
        }
        for row in active_semantic_actions()
    ]


def select_semantic_action(frame: Any) -> SemanticActionRow | None:
    """Select the best active matrix row for a canonical semantic frame.

    Inputs:
        Receives a SemanticFrame-like object and reads typed domain, intent, entity, metric, filters, targets, and output shape.

    Returns:
        The best matching active row, or None when the matrix does not support the frame.

    Used by:
        SemanticFrameCompiler.compile before domain-specific lowering.
    """
    return select_semantic_action_decision(frame).row


def select_semantic_action_decision(frame: Any) -> SemanticActionSelection:
    """Select a semantic action row with compact rejection metadata.

    Inputs:
        Receives a SemanticFrame-like object.

    Returns:
        Explainable selection facts for compiler and coverage diagnostics.

    Used by:
        select_semantic_action, SemanticFrameCompiler, and matrix tests.
    """
    candidates = [row for row in active_semantic_actions() if row.domain == _frame_domain(frame) and row.intent == _frame_intent(frame)]
    scored: list[tuple[int, SemanticActionRow]] = []
    rejected: list[str] = []
    for row in candidates:
        score = _row_match_score(row, frame)
        if score >= 0:
            scored.append((score, row))
        else:
            rejected.append(row.semantic_action_id)
    if not scored:
        return SemanticActionSelection(row=None, rejected_candidates=tuple(rejected))
    scored.sort(key=lambda item: item[0], reverse=True)
    score, row = scored[0]
    entities = tuple(value for value in _frame_entities(frame) if value in {_canonical_entity(row.domain, item) for item in row.entities})
    filters = tuple(value for value in _frame_filters(frame) if value in {_canonical_filter(item) for item in row.filters})
    dimensions = tuple(
        value
        for value in (*_frame_dimensions(frame), *_frame_target_dimensions(frame))
        if value in {_canonical_dimension(item) for item in (*row.grouping_dimensions, *row.target_dimensions)}
    )
    return SemanticActionSelection(
        row=row,
        score=score,
        matched_entities=entities,
        matched_filters=filters,
        matched_dimensions=tuple(dict.fromkeys(dimensions)),
        required_tools=row.tools,
        expected_output=row.output,
        rejected_candidates=tuple(rejected),
    )


def _row_match_score(row: SemanticActionRow, frame: Any) -> int:
    """Score whether one matrix row covers a frame.

    Inputs:
        Receives one matrix row and a SemanticFrame-like object.

    Returns:
        Non-negative score for a match, or -1 for no match.

    Used by:
        select_semantic_action.
    """
    domain = row.domain
    entities = set(_frame_entities(frame))
    row_entities = {_canonical_entity(domain, value) for value in row.entities}
    filters = set(_frame_filters(frame))
    row_filters = {_canonical_filter(value) for value in row.filters}
    dimensions = set(_frame_dimensions(frame))
    targets = set(_frame_target_dimensions(frame))
    row_grouping_dimensions = {_canonical_dimension(value) for value in row.grouping_dimensions}
    row_target_dimensions = {_canonical_dimension(value) for value in row.target_dimensions}
    metric = _frame_metric(frame)
    output_cardinality = _frame_output_cardinality(frame)
    action_id = row.semantic_action_id
    score = 10

    if action_id.endswith("search_content"):
        return 100 if filters.intersection({"needle", "query", "term"}) or entities.intersection({"content", "references", "matches"}) else -1
    if action_id.endswith("read_file"):
        return 100 if filters.intersection({"file", "path"}) or entities.intersection({"file_content", "read_file"}) else -1
    if action_id.endswith("find_files"):
        return 90 if filters.intersection({"pattern", "path"}) or entities.intersection({"file", "files"}) else -1
    if action_id.endswith("glob_files"):
        return 90 if filters.intersection({"extension", "pattern", "type"}) or entities.intersection({"file", "files"}) else -1
    if action_id.endswith("list_directory"):
        return 90 if entities.intersection({"directory", "folder"}) or filters.intersection({"path"}) else -1
    if action_id.endswith("discover_configs"):
        return 100 if entities.intersection({"config", "configs"}) or filters.intersection({"pattern"}) else -1
    if action_id.endswith("summarize_docs"):
        return 100 if entities.intersection({"docs", "markdown"}) else -1
    if row.entities and not entities:
        return -1
    if row.entities and entities and not entities.intersection(row_entities):
        if not _entity_family_match(tuple(row_entities), entities):
            return -1
    if row.metrics and metric and metric not in row.metrics:
        return -1
    if row.grouping_dimensions and not (dimensions.intersection(row_grouping_dimensions) or targets.intersection(row_grouping_dimensions)):
        if output_cardinality == "grouped":
            return -1
    if action_id.endswith("schema_list_tables"):
        return 100 if entities.intersection({"tables"}) else (80 if entities.intersection({"schema"}) else -1)
    if action_id.endswith("schema_join_path"):
        return 100 if entities.intersection({"relationships", "join_path"}) else -1
    if action_id.endswith("schema_describe_entity"):
        return 95 if entities.intersection({"patients", "studies", "series", "instances", "schema", "columns"}) else -1
    if action_id.endswith("top_n"):
        return 98 if "limit" in filters or "related_count" in filters else -1
    if action_id.endswith("count_multi_entity"):
        return 100 if len(entities.intersection(row_entities)) > 1 else -1
    if action_id.endswith("co_occurrence_count"):
        return 98 if {"concept_term", "match_policy"}.intersection(filters) else -1
    if action_id.endswith("missing_value_count"):
        return 98 if "missing_value" in filters else -1
    if action_id.endswith("count_filtered_entity"):
        semantic_filters = filters.intersection(row_filters - {"database"})
        return 95 if semantic_filters else -1
    if action_id.endswith("count_grouped") or action_id.endswith("queue_grouped_count"):
        return 90 if dimensions.intersection(row_grouping_dimensions) or output_cardinality == "grouped" else -1
    if action_id.endswith("accounting_grouped_aggregate"):
        return 100 if metric and (dimensions.intersection(row_grouping_dimensions) or targets.intersection(row_grouping_dimensions)) else -1
    if action_id.endswith("accounting_elapsed_aggregate"):
        return 85 if metric else -1
    if action_id.endswith("list_partitions"):
        return 100 if entities.intersection({"partitions"}) else -1
    if action_id.endswith("list_nodes"):
        return 100 if entities.intersection({"nodes"}) else -1
    if action_id.endswith("count_partitions"):
        return 100 if entities.intersection({"partitions"}) else -1
    if action_id.endswith("count_nodes"):
        return 100 if entities.intersection({"nodes"}) else -1
    if action_id.endswith("node_detail"):
        return 95 if entities.intersection({"node_detail"}) or "node" in filters else -1
    if action_id.endswith("queue_list_jobs") or action_id.endswith("queue_count_jobs"):
        return 92 if entities.intersection({"jobs"}) else -1
    if action_id.endswith("cluster_status"):
        return 80
    if action_id.endswith("accounting_health"):
        return 100 if entities.intersection({"accounting", "slurmdbd", "slurmdbd_health"}) else -1
    if action_id.endswith("group_count_by_extension"):
        return 100 if dimensions.intersection({"extension", "type"}) or targets.intersection({"extension", "type"}) else -1
    if action_id.endswith("search_content"):
        return 100 if filters.intersection({"needle", "query", "term"}) else -1
    if action_id.endswith("read_file"):
        return 100 if entities.intersection({"file_content", "read_file"}) or "file" in filters else -1
    if action_id.endswith("docker_list_containers_readonly"):
        return 100 if entities.intersection({"docker", "containers", "container"}) else -1
    if row.filters and filters.intersection(row_filters):
        score += 10
    if row.target_dimensions and targets.intersection(row_target_dimensions):
        score += 10
    return score


def _frame_domain(frame: Any) -> str:
    """Read a normalized frame domain.

    Inputs:
        Receives a frame-like object.

    Returns:
        Normalized domain string.

    Used by:
        select_semantic_action.
    """
    return normalize_semantic_token(getattr(frame, "domain", ""))


def _frame_intent(frame: Any) -> str:
    """Read a normalized frame intent.

    Inputs:
        Receives a frame-like object.

    Returns:
        Normalized intent string.

    Used by:
        select_semantic_action.
    """
    return normalize_semantic_token(getattr(frame, "intent", ""))


def _frame_metric(frame: Any) -> str:
    """Read a normalized frame metric name.

    Inputs:
        Receives a frame-like object.

    Returns:
        Metric name or empty string.

    Used by:
        select_semantic_action.
    """
    metric = getattr(frame, "metric", None)
    if metric is None:
        return ""
    return normalize_semantic_token(getattr(metric, "name", metric))


def _frame_output_cardinality(frame: Any) -> str:
    """Read the requested output cardinality from a frame.

    Inputs:
        Receives a frame-like object.

    Returns:
        Normalized cardinality string.

    Used by:
        select_semantic_action.
    """
    output = getattr(frame, "output", None)
    return normalize_semantic_token(getattr(output, "cardinality", ""))


def _frame_entities(frame: Any) -> tuple[str, ...]:
    """Collect entity-like labels from a semantic frame.

    Inputs:
        Receives a frame-like object.

    Returns:
        Normalized entity labels from entity, output result entities, and entity targets.

    Used by:
        select_semantic_action.
    """
    values: list[str] = []
    domain = _frame_domain(frame)
    entity = _canonical_entity(domain, getattr(frame, "entity", ""))
    if entity:
        values.append(entity)
    output = getattr(frame, "output", None)
    values.extend(_canonical_entity(domain, value) for value in getattr(output, "result_entities", []) or [])
    targets = getattr(frame, "targets", {}) or {}
    entity_target = targets.get("entity") if isinstance(targets, dict) else None
    values.extend(_canonical_entity(domain, value) for value in _target_values(entity_target))
    return tuple(dict.fromkeys(value for value in values if value))


def _frame_filters(frame: Any) -> tuple[str, ...]:
    """Collect filter field names from a semantic frame.

    Inputs:
        Receives a frame-like object.

    Returns:
        Normalized filter names.

    Used by:
        select_semantic_action.
    """
    return tuple(
        dict.fromkeys(
            _canonical_filter(getattr(item, "field", ""))
            for item in getattr(frame, "filters", []) or []
            if _canonical_filter(getattr(item, "field", ""))
        )
    )


def _frame_dimensions(frame: Any) -> tuple[str, ...]:
    """Collect grouping dimensions from a semantic frame.

    Inputs:
        Receives a frame-like object.

    Returns:
        Normalized dimensions from frame and output contract.

    Used by:
        select_semantic_action.
    """
    values = [_canonical_dimension(value) for value in getattr(frame, "dimensions", []) or []]
    output = getattr(frame, "output", None)
    values.extend(_canonical_dimension(value) for value in getattr(output, "group_by", []) or [])
    return tuple(dict.fromkeys(value for value in values if value))


def _frame_target_dimensions(frame: Any) -> tuple[str, ...]:
    """Collect target dimension names from a semantic frame.

    Inputs:
        Receives a frame-like object.

    Returns:
        Normalized target dimension names.

    Used by:
        select_semantic_action.
    """
    targets = getattr(frame, "targets", {}) or {}
    if not isinstance(targets, dict):
        return ()
    return tuple(dict.fromkeys(_canonical_dimension(key) for key in targets if _canonical_dimension(key)))


def _entity_family_match(row_entities: tuple[str, ...], frame_entities: set[str]) -> bool:
    """Return whether entity aliases belong to the same semantic family.

    Inputs:
        Receives row entity labels and frame entity labels.

    Returns:
        True for broad families such as DICOM entities or SLURM jobs.

    Used by:
        _row_match_score.
    """
    row_set = set(row_entities)
    if row_set.intersection({"patients", "studies", "series", "instances", "rtplan", "rtdose", "rtstruct", "ct", "mr", "pt"}):
        return bool(frame_entities.intersection(row_set))
    if row_set.intersection({"jobs", "queue"}) and frame_entities.intersection({"job", "jobs", "queue"}):
        return True
    return False


def _target_values(target: Any) -> list[str]:
    """Read target values without depending on the SemanticTargetSet class.

    Inputs:
        Receives a target-like object.

    Returns:
        Normalized target values.

    Used by:
        _frame_entities.
    """
    if target is None:
        return []
    if hasattr(target, "model_dump"):
        raw = target.model_dump().get("values")
    elif isinstance(target, dict):
        raw = target.get("values")
    else:
        raw = target
    values = raw if isinstance(raw, list) else [raw]
    return [normalize_semantic_token(value) for value in values if normalize_semantic_token(value)]


def _normalize_token(value: Any) -> str:
    """Normalize labels used in matrix matching.

    Inputs:
        Receives arbitrary label text.

    Returns:
        Lowercase underscore token.

    Used by:
        Semantic action matrix helpers.
    """
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def _canonical_entity(domain: str, value: Any) -> str:
    """Normalize an entity for matrix matching.

    Inputs:
        Receives a matrix row domain and raw entity label.

    Returns:
        Canonical entity token.

    Used by:
        _row_match_score and selection metadata helpers.
    """
    return normalize_semantic_entity(domain, value)


def _canonical_dimension(value: Any) -> str:
    """Normalize a dimension for matrix matching.

    Inputs:
        Receives a raw dimension label.

    Returns:
        Canonical dimension token.

    Used by:
        _row_match_score and frame readers.
    """
    return normalize_semantic_dimension(value)


def _canonical_filter(value: Any) -> str:
    """Normalize a filter field for matrix matching.

    Inputs:
        Receives a raw filter field label.

    Returns:
        Canonical filter token.

    Used by:
        _row_match_score and frame readers.
    """
    return normalize_semantic_filter_field(value)


def _out(kind: SemanticOutputKind, cardinality: SemanticOutputCardinality, style: SemanticRenderStyle) -> SemanticActionOutput:
    """Build a compact output-contract row.

    Inputs:
        Receives output kind, cardinality, and render style.

    Returns:
        SemanticActionOutput metadata.

    Used by:
        SEMANTIC_ACTION_MATRIX construction.
    """
    return SemanticActionOutput(kind, cardinality, style)


SEMANTIC_ACTION_MATRIX: tuple[SemanticActionRow, ...] = (
    # SQL / DICOM read-only data and metadata actions.
    SemanticActionRow("sql.count_multi_entity", "sql", "count", entities=("patients", "studies", "series", "instances", "rtplan", "rtdose", "rtstruct", "ct", "mr", "pt"), filters=("database",), target_dimensions=("entity",), output=_out("table", "multi_scalar", "metric_table"), strategy="sql_template", tools=("sql.query",), compiler_owner="sql"),
    SemanticActionRow("sql.count_filtered_entity", "sql", "count", entities=("patients", "studies", "series", "instances"), filters=("database", "age", "patient_age", "concept_term", "modality", "missing_value"), target_dimensions=("modality",), output=_out("scalar", "single", "scalar"), strategy="schema_grounded_builder", tools=("sql.query",), compiler_owner="sql"),
    SemanticActionRow("sql.count_grouped", "sql", "count", entities=("patients", "studies", "series", "instances", "modalities"), filters=("database", "modality"), target_dimensions=("modality", "year", "month", "decade"), grouping_dimensions=("modality", "year", "month", "decade", "patient", "study", "series"), output=_out("table", "grouped", "metric_table"), strategy="grouped_pushdown", tools=("sql.query",), compiler_owner="sql"),
    SemanticActionRow("sql.count_entity", "sql", "count", entities=("patients", "studies", "series", "instances", "rtplan", "rtdose", "rtstruct", "ct", "mr", "pt", "modalities"), filters=("database",), output=_out("scalar", "single", "scalar"), strategy="sql_template", tools=("sql.query",), compiler_owner="sql"),
    SemanticActionRow("sql.aggregate_metric", "sql", "aggregate_metric", entities=("patients", "studies", "series", "instances", "modalities"), metrics=("count", "average", "min", "max"), filters=("database", "modality", "age"), grouping_dimensions=("patient", "study", "series", "modality", "decade"), output=_out("table", "multi_scalar", "metric_table"), strategy="schema_grounded_builder", tools=("sql.query",), compiler_owner="sql"),
    SemanticActionRow("sql.list_entity", "sql", "list", entities=("patients", "studies", "series", "instances", "modalities", "tables", "columns"), filters=("database", "modality", "age"), output=_out("table", "collection", "record_table"), strategy="schema_grounded_builder", tools=("sql.query",), compiler_owner="sql"),
    SemanticActionRow("sql.schema_list_tables", "sql", "schema_answer", entities=("tables", "schema"), filters=("database",), output=_out("table", "collection", "record_table"), strategy="schema_metadata", tools=("sql.schema",), compiler_owner="sql"),
    SemanticActionRow("sql.schema_describe_entity", "sql", "schema_answer", entities=("patients", "studies", "series", "instances", "schema"), filters=("database",), output=_out("text", "sectioned", "sectioned"), strategy="schema_metadata", tools=("sql.schema",), compiler_owner="sql"),
    SemanticActionRow("sql.schema_join_path", "sql", "schema_answer", entities=("join_path", "relationships", "schema"), filters=("database",), output=_out("text", "sectioned", "sectioned"), strategy="schema_metadata", tools=("sql.schema",), compiler_owner="sql"),
    SemanticActionRow("sql.validate_sql", "sql", "validate_or_explain", entities=("sql", "query"), filters=("database", "query"), output=_out("text", "single", "bullets"), strategy="sql_validate", tools=("sql.validate",), compiler_owner="sql"),
    SemanticActionRow("sql.explain_sql", "sql", "validate_or_explain", entities=("sql", "query"), filters=("database", "query"), output=_out("text", "single", "bullets"), strategy="sql_validate", tools=("sql.validate",), compiler_owner="sql"),
    SemanticActionRow("sql.top_n", "sql", "list", entities=("patients", "studies", "series", "instances", "modalities"), filters=("database", "modality", "limit", "related_count"), grouping_dimensions=("patient", "study", "series", "modality"), output=_out("table", "collection", "record_table"), strategy="schema_grounded_builder", tools=("sql.query",), compiler_owner="sql"),
    SemanticActionRow("sql.missing_value_count", "sql", "count", entities=("patients", "studies", "series", "instances"), filters=("database", "missing_value"), output=_out("scalar", "single", "scalar"), strategy="schema_grounded_builder", tools=("sql.query",), compiler_owner="sql"),
    SemanticActionRow("sql.co_occurrence_count", "sql", "count", entities=("patients", "studies"), filters=("database", "concept_term", "match_policy", "modality"), output=_out("scalar", "single", "scalar"), strategy="schema_grounded_builder", tools=("sql.query",), compiler_owner="sql"),
    # SLURM read-only actions.
    SemanticActionRow("slurm.list_partitions", "slurm", "list", entities=("partition", "partitions"), filters=("partition",), output=_out("table", "collection", "record_table"), strategy="single_target_pushdown", tools=("slurm.partitions",), compiler_owner="slurm"),
    SemanticActionRow("slurm.list_nodes", "slurm", "list", entities=("node", "nodes", "unique_nodes"), filters=("partition", "state", "state_group"), output=_out("table", "collection", "record_table"), strategy="single_target_pushdown", tools=("slurm.nodes",), compiler_owner="slurm"),
    SemanticActionRow("slurm.count_partitions", "slurm", "count", entities=("partition", "partitions"), filters=("partition",), output=_out("scalar", "single", "scalar"), strategy="single_target_pushdown", tools=("slurm.partitions",), compiler_owner="slurm"),
    SemanticActionRow("slurm.count_nodes", "slurm", "count", entities=("node", "nodes", "unique_nodes"), filters=("partition", "state", "state_group"), output=_out("scalar", "single", "scalar"), strategy="single_target_pushdown", tools=("slurm.nodes",), compiler_owner="slurm"),
    SemanticActionRow("slurm.node_detail", "slurm", "list", entities=("node_detail",), filters=("node",), output=_out("table", "collection", "record_table"), strategy="single_target_pushdown", tools=("slurm.node_detail",), compiler_owner="slurm"),
    SemanticActionRow("slurm.queue_list_jobs", "slurm", "list", entities=("job", "jobs", "queue"), filters=("partition", "state", "user"), output=_out("table", "collection", "record_table"), strategy="single_target_pushdown", tools=("slurm.queue",), compiler_owner="slurm"),
    SemanticActionRow("slurm.queue_grouped_count", "slurm", "count", entities=("job", "jobs", "queue"), filters=("partition", "state", "user"), target_dimensions=("partition", "state", "user", "node"), grouping_dimensions=("partition", "state", "user", "node"), output=_out("table", "grouped", "metric_table"), strategy="grouped_pushdown", tools=("slurm.queue",), compiler_owner="slurm"),
    SemanticActionRow("slurm.queue_count_jobs", "slurm", "count", entities=("job", "jobs", "queue"), filters=("partition", "state", "user"), output=_out("scalar", "single", "scalar"), strategy="single_target_pushdown", tools=("slurm.queue",), compiler_owner="slurm"),
    SemanticActionRow("slurm.accounting_list_jobs", "slurm", "list", entities=("accounting_jobs", "jobs"), filters=("partition", "state", "user", "start", "end"), grouping_dimensions=("partition", "state", "user", "job_name"), output=_out("table", "collection", "record_table"), strategy="single_target_pushdown", tools=("slurm.accounting",), compiler_owner="slurm"),
    SemanticActionRow("slurm.accounting_grouped_aggregate", "slurm", "aggregate_metric", entities=("job", "jobs", "accounting_jobs"), metrics=("average_elapsed", "min_elapsed", "max_elapsed", "sum_elapsed", "count", "runtime_summary"), filters=("partition", "state", "user", "start", "end", "include_all_states"), target_dimensions=("partition", "state", "user", "job_name"), grouping_dimensions=("partition", "state", "user", "job_name"), output=_out("table", "grouped", "metric_table"), strategy="grouped_pushdown", tools=("slurm.accounting_aggregate",), compiler_owner="slurm"),
    SemanticActionRow("slurm.accounting_elapsed_aggregate", "slurm", "aggregate_metric", entities=("job", "jobs", "accounting_jobs"), metrics=("average_elapsed", "min_elapsed", "max_elapsed", "sum_elapsed", "count", "runtime_summary"), filters=("partition", "state", "user", "start", "end", "include_all_states"), output=_out("scalar", "single", "scalar"), strategy="single_target_pushdown", tools=("slurm.accounting_aggregate",), compiler_owner="slurm"),
    SemanticActionRow("slurm.cluster_status", "slurm", "status", entities=("cluster", "queue", "nodes", "partitions"), output=_out("status", "single", "key_value"), strategy="single_target_pushdown", tools=("slurm.metrics",), compiler_owner="slurm"),
    SemanticActionRow("slurm.accounting_health", "slurm", "status", entities=("accounting", "slurmdbd", "slurmdbd_health"), output=_out("status", "single", "key_value"), strategy="single_target_pushdown", tools=("slurm.slurmdbd_health",), compiler_owner="slurm"),
    # Filesystem read-only actions.
    SemanticActionRow("filesystem.list_directory", "filesystem", "list", entities=("directory", "folder"), filters=("path",), output=_out("table", "collection", "record_table"), strategy="single_target_pushdown", tools=("fs.list",), compiler_owner="filesystem"),
    SemanticActionRow("filesystem.glob_files", "filesystem", "list", entities=("file", "files"), filters=("path", "extension", "pattern", "type"), target_dimensions=("extension", "type"), output=_out("table", "collection", "record_table"), strategy="single_target_pushdown", tools=("fs.glob",), compiler_owner="filesystem"),
    SemanticActionRow("filesystem.find_files", "filesystem", "list", entities=("file", "files"), filters=("path", "pattern"), output=_out("table", "collection", "record_table"), strategy="single_target_pushdown", tools=("fs.find",), compiler_owner="filesystem"),
    SemanticActionRow("filesystem.search_content", "filesystem", "list", entities=("content", "references", "matches"), filters=("path", "needle", "query", "term", "pattern"), output=_out("table", "collection", "record_table"), strategy="single_target_pushdown", tools=("fs.search_content",), compiler_owner="filesystem"),
    SemanticActionRow("filesystem.read_file", "filesystem", "list", entities=("file_content", "read_file"), filters=("path", "file"), output=_out("text", "single", "bullets"), strategy="single_target_pushdown", tools=("fs.read",), compiler_owner="filesystem"),
    SemanticActionRow("filesystem.group_count_by_extension", "filesystem", "count", entities=("file", "files"), filters=("path", "extension", "pattern", "type"), target_dimensions=("extension", "type"), grouping_dimensions=("extension", "type"), output=_out("table", "multi_scalar", "metric_table"), strategy="fan_out", tools=("fs.aggregate",), compiler_owner="filesystem"),
    SemanticActionRow("filesystem.count_files", "filesystem", "count", entities=("file", "files"), filters=("path", "extension", "pattern", "type"), output=_out("scalar", "single", "scalar"), strategy="single_target_pushdown", tools=("fs.aggregate",), compiler_owner="filesystem"),
    SemanticActionRow("filesystem.size_path", "filesystem", "aggregate_metric", entities=("path", "directory", "file"), metrics=("total_size", "count_and_total_size"), filters=("path",), output=_out("table", "multi_scalar", "metric_table"), strategy="single_target_pushdown", tools=("fs.aggregate",), compiler_owner="filesystem"),
    SemanticActionRow("filesystem.summarize_docs", "filesystem", "list", entities=("docs", "markdown"), filters=("path", "extension"), output=_out("text", "sectioned", "sectioned"), strategy="single_target_pushdown", tools=("fs.glob", "fs.read"), compiler_owner="filesystem"),
    SemanticActionRow("filesystem.discover_configs", "filesystem", "list", entities=("config", "configs"), filters=("path", "pattern"), output=_out("table", "collection", "record_table"), strategy="single_target_pushdown", tools=("fs.glob",), compiler_owner="filesystem"),
    # Shell/system read-only actions.
    SemanticActionRow("shell.current_user", "shell", "status", entities=("user", "current_user"), output=_out("status", "single", "key_value"), strategy="known_read_only_command", tools=("shell.exec",), compiler_owner="shell"),
    SemanticActionRow("shell.hostname", "shell", "status", entities=("hostname",), output=_out("status", "single", "key_value"), strategy="known_read_only_command", tools=("shell.exec",), compiler_owner="shell"),
    SemanticActionRow("shell.uptime", "shell", "status", entities=("uptime",), output=_out("status", "single", "key_value"), strategy="known_read_only_command", tools=("shell.exec",), compiler_owner="shell"),
    SemanticActionRow("shell.pwd", "shell", "status", entities=("pwd", "working_directory"), output=_out("status", "single", "key_value"), strategy="known_read_only_command", tools=("shell.exec",), compiler_owner="shell"),
    SemanticActionRow("shell.disk_usage", "shell", "status", entities=("disk", "filesystem", "filesystems"), output=_out("table", "collection", "record_table"), strategy="known_read_only_command", tools=("shell.exec",), compiler_owner="shell"),
    SemanticActionRow("shell.memory_usage", "shell", "status", entities=("memory", "resources"), output=_out("table", "collection", "record_table"), strategy="known_read_only_command", tools=("shell.exec",), compiler_owner="shell"),
    SemanticActionRow("shell.cpu_info", "shell", "status", entities=("cpu", "resources"), output=_out("text", "single", "bullets"), strategy="known_read_only_command", tools=("shell.exec",), compiler_owner="shell"),
    SemanticActionRow("shell.list_ports", "shell", "list", entities=("port", "ports"), output=_out("table", "collection", "record_table"), strategy="known_read_only_command", tools=("shell.exec",), compiler_owner="shell"),
    SemanticActionRow("shell.list_processes", "shell", "list", entities=("process", "processes"), filters=("process",), output=_out("table", "collection", "record_table"), strategy="known_read_only_command", tools=("shell.exec",), compiler_owner="shell"),
    SemanticActionRow("shell.count_processes", "shell", "count", entities=("process", "processes"), filters=("process",), output=_out("scalar", "single", "scalar"), strategy="known_read_only_command", tools=("shell.exec",), compiler_owner="shell"),
    SemanticActionRow("shell.mounts", "shell", "list", entities=("mount", "mounts", "filesystems"), output=_out("table", "collection", "record_table"), strategy="known_read_only_command", tools=("shell.exec",), compiler_owner="shell"),
    SemanticActionRow("shell.docker_list_containers_readonly", "shell", "list", entities=("docker", "container", "containers"), output=_out("table", "collection", "record_table"), strategy="known_read_only_command", tools=("shell.exec",), compiler_owner="shell"),
    # Text/runtime and diagnostics.
    SemanticActionRow("text.format_table", "text", "transform", entities=("table",), output=_out("table", "collection", "record_table"), strategy="format", tools=("text.format",), compiler_owner="text"),
    SemanticActionRow("text.format_scalar", "text", "transform", entities=("scalar",), output=_out("scalar", "single", "scalar"), strategy="format", tools=("text.format",), compiler_owner="text"),
    SemanticActionRow("text.format_sections", "text", "transform", entities=("sections",), output=_out("text", "sectioned", "sectioned"), strategy="format", tools=("text.format",), compiler_owner="text"),
    SemanticActionRow("runtime.return_answer", "text", "transform", entities=("answer",), output=_out("text", "single", "bullets"), strategy="return", tools=("runtime.return",), compiler_owner="runtime"),
    SemanticActionRow("diagnostic.workspace_dashboard", "diagnostic", "diagnostic", entities=("workspace", "dashboard"), output=_out("text", "sectioned", "sectioned"), strategy="diagnostic_sections", tools=("fs.list", "fs.search_content", "shell.exec"), compiler_owner="diagnostic"),
    SemanticActionRow("diagnostic.runtime_capability_summary", "diagnostic", "diagnostic", entities=("runtime", "capabilities"), output=_out("text", "sectioned", "sectioned"), strategy="diagnostic_sections", tools=("fs.search_content",), compiler_owner="diagnostic"),
    SemanticActionRow("diagnostic.sql_capability_summary", "diagnostic", "diagnostic", entities=("sql", "database"), output=_out("text", "sectioned", "sectioned"), strategy="diagnostic_sections", tools=("sql.schema",), compiler_owner="diagnostic"),
    SemanticActionRow("diagnostic.slurm_capability_summary", "diagnostic", "diagnostic", entities=("slurm",), output=_out("text", "sectioned", "sectioned"), strategy="diagnostic_sections", tools=("slurm.metrics", "slurm.partitions", "slurm.nodes"), compiler_owner="diagnostic"),
    SemanticActionRow("diagnostic.system_diagnostic", "diagnostic", "diagnostic", entities=("system",), output=_out("text", "sectioned", "sectioned"), strategy="diagnostic_sections", tools=("shell.exec",), compiler_owner="diagnostic"),
    SemanticActionRow("diagnostic.multi_domain_sections", "diagnostic", "diagnostic", entities=("multi_domain", "sections"), output=_out("text", "sectioned", "sectioned"), strategy="diagnostic_sections", tools=("fs.list", "fs.search_content", "sql.schema", "shell.exec", "slurm.metrics"), compiler_owner="diagnostic"),
    # Disabled mutating tier: documented but not eligible for semantic lowering.
    SemanticActionRow("filesystem.write_file", "filesystem", "transform", tools=("fs.write",), status="disabled_mutating", safety_tier="mutating_disabled", compiler_owner="filesystem"),
    SemanticActionRow("filesystem.copy_file", "filesystem", "transform", tools=("fs.copy",), status="disabled_mutating", safety_tier="mutating_disabled", compiler_owner="filesystem"),
    SemanticActionRow("filesystem.mkdir", "filesystem", "transform", tools=("fs.mkdir",), status="disabled_mutating", safety_tier="mutating_disabled", compiler_owner="filesystem"),
    SemanticActionRow("filesystem.delete_file", "filesystem", "transform", tools=(), status="disabled_mutating", safety_tier="mutating_disabled", compiler_owner="filesystem"),
    SemanticActionRow("shell.execute_mutating_command", "shell", "transform", tools=("shell.exec",), status="disabled_mutating", safety_tier="mutating_disabled", compiler_owner="shell"),
    SemanticActionRow("python.execute_code", "text", "transform", tools=("python.exec",), status="disabled_mutating", safety_tier="mutating_disabled", compiler_owner="python"),
    SemanticActionRow("sql.mutate", "sql", "transform", tools=("sql.query",), status="disabled_mutating", safety_tier="mutating_disabled", compiler_owner="sql"),
    SemanticActionRow("slurm.cancel_job", "slurm", "transform", tools=(), status="disabled_mutating", safety_tier="mutating_disabled", compiler_owner="slurm"),
    SemanticActionRow("slurm.submit_job", "slurm", "transform", tools=(), status="disabled_mutating", safety_tier="mutating_disabled", compiler_owner="slurm"),
)


__all__ = [
    "SEMANTIC_ACTION_MATRIX",
    "SemanticActionOutput",
    "SemanticActionRow",
    "SemanticActionSelection",
    "active_semantic_actions",
    "disabled_semantic_actions",
    "select_semantic_action",
    "select_semantic_action_decision",
    "semantic_action_prompt_metadata",
]
