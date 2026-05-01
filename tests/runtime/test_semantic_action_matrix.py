from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.runtime.semantic.action_matrix import (
    SEMANTIC_ACTION_MATRIX,
    active_semantic_actions,
    disabled_semantic_actions,
    select_semantic_action,
    select_semantic_action_decision,
    semantic_action_prompt_metadata,
)
from aor_runtime.runtime.semantic.ontology import detect_prompt_entities, normalize_semantic_entity
from aor_runtime.runtime.semantic_frame import SemanticCoverageValidator, SemanticFrame, SemanticFrameCanonicalizer, SemanticFrameCompiler, SemanticOutputContract
from aor_runtime.tools.factory import build_tool_registry


def _settings(tmp_path: Path) -> Settings:
    return Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")


def _registered_tool_names(tmp_path: Path) -> set[str]:
    return set(build_tool_registry(_settings(tmp_path)).names())


def test_active_semantic_action_matrix_rows_have_registered_tools_and_output_contracts(tmp_path: Path) -> None:
    registered = _registered_tool_names(tmp_path)

    assert active_semantic_actions()
    for row in active_semantic_actions():
        assert row.semantic_action_id
        assert row.compiler_owner
        assert row.tools
        assert set(row.tools).issubset(registered)
        assert row.output.kind != "unknown"
        assert row.output.cardinality != "unknown"
        assert row.output.render_style != "unknown"


def test_disabled_mutating_rows_are_documented_but_not_selected(tmp_path: Path) -> None:
    assert disabled_semantic_actions()
    assert {row.status for row in disabled_semantic_actions()} == {"disabled_mutating"}

    frame = SemanticFrame(
        domain="filesystem",
        intent="transform",
        entity="write_file",
        output=SemanticOutputContract(kind="file", cardinality="single", render_style="record_table"),
    )

    assert select_semantic_action(frame) is None
    assert SemanticFrameCompiler(settings=_settings(tmp_path), allowed_tools=["fs.write"]).compile(frame) is None


def test_semantic_action_prompt_metadata_contains_only_safe_matrix_facts() -> None:
    metadata = semantic_action_prompt_metadata()

    assert metadata
    assert all("tools" not in row for row in metadata)
    assert any(row["semantic_action_id"] == "sql.count_entity" for row in metadata)
    assert any(row["semantic_action_id"] == "slurm.list_partitions" for row in metadata)
    assert not any(row["semantic_action_id"] == "filesystem.write_file" for row in metadata)


def test_matrix_selects_sql_multi_entity_and_filtered_count_actions() -> None:
    multi = SemanticFrame(
        domain="sql",
        intent="count",
        entity="dicom_counts",
        targets={"entity": {"values": ["patients", "studies", "series", "rtplan"]}},
        output=SemanticOutputContract(
            kind="table",
            cardinality="multi_scalar",
            render_style="metric_table",
            result_entities=["patients", "studies", "series", "rtplan"],
        ),
    )
    filtered = SemanticFrame(
        domain="sql",
        intent="count",
        entity="patients",
        filters=[{"field": "age", "operator": "gt", "value": 45}],
        output=SemanticOutputContract(kind="scalar", cardinality="single", render_style="scalar", result_entities=["patients"]),
    )

    assert select_semantic_action(multi).semantic_action_id == "sql.count_multi_entity"
    assert select_semantic_action(filtered).semantic_action_id == "sql.count_filtered_entity"


def test_semantic_ontology_normalizes_core_sql_and_slurm_entities() -> None:
    assert normalize_semantic_entity("sql", "patient") == "patients"
    assert normalize_semantic_entity("sql", "RTPLANS") == "rtplan"
    assert normalize_semantic_entity("slurm", "unique nodes") == "nodes"
    assert normalize_semantic_entity("slurm", "partition") == "partitions"
    assert [match.canonical for match in detect_prompt_entities("slurm", "list all the nodes in slurm")] == ["nodes"]


def test_prompt_grounding_corrects_llm_entity_before_matrix_selection(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = SemanticFrame(
        domain="slurm",
        intent="list",
        entity="jobs",
        output=SemanticOutputContract(kind="table", cardinality="collection", render_style="record_table"),
    )

    canonical = SemanticFrameCanonicalizer(settings).canonicalize(frame, goal="list all the nodes in slurm", allow_prompt_inference=False)
    decision = select_semantic_action_decision(canonical)

    assert canonical.entity == "nodes"
    assert canonical.output.result_entities == ["nodes"]
    assert decision.row is not None
    assert decision.row.semantic_action_id == "slurm.list_nodes"
    assert decision.required_tools == ("slurm.nodes",)


def test_matrix_selects_simple_sql_entity_count_after_grounding(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = SemanticFrame(
        domain="sql",
        intent="count",
        entity="dicom",
        filters=[{"field": "database", "operator": "eq", "value": "dicom"}],
        output=SemanticOutputContract(kind="scalar", cardinality="single", render_style="scalar"),
    )

    canonical = SemanticFrameCanonicalizer(settings).canonicalize(frame, goal="count of patients in dicom", allow_prompt_inference=False)
    decision = select_semantic_action_decision(canonical)

    assert canonical.entity == "patients"
    assert canonical.output.result_entities == ["patients"]
    assert decision.row is not None
    assert decision.row.semantic_action_id == "sql.count_entity"


def test_matrix_compiles_core_slurm_inventory_actions(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    compiler = SemanticFrameCompiler(settings=settings, allowed_tools=["slurm.partitions", "slurm.nodes"])
    partitions = SemanticFrame(
        domain="slurm",
        intent="list",
        entity="partitions",
        output=SemanticOutputContract(kind="table", cardinality="collection", render_style="record_table"),
    )
    nodes = SemanticFrame(
        domain="slurm",
        intent="list",
        entity="nodes",
        output=SemanticOutputContract(kind="table", cardinality="collection", render_style="record_table"),
    )
    node_count = SemanticFrame(
        domain="slurm",
        intent="count",
        entity="nodes",
        output=SemanticOutputContract(kind="scalar", cardinality="single", render_style="scalar"),
    )

    compiled_partitions = compiler.compile(partitions)
    compiled_nodes = compiler.compile(nodes)
    compiled_node_count = compiler.compile(node_count)

    assert compiled_partitions is not None
    assert compiled_partitions.metadata["semantic_action"] == "slurm.list_partitions"
    assert compiled_partitions.plan.steps[0].action == "slurm.partitions"
    assert compiled_partitions.plan.steps[1].args["source"]["path"] == "partitions"
    assert compiled_nodes is not None
    assert compiled_nodes.metadata["semantic_action"] == "slurm.list_nodes"
    assert compiled_nodes.plan.steps[0].action == "slurm.nodes"
    assert compiled_node_count is not None
    assert compiled_node_count.metadata["semantic_action"] == "slurm.count_nodes"
    assert compiled_node_count.plan.steps[1].args["source"]["path"] == "unique_count"


def test_matrix_compiles_simple_sql_entity_counts(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    compiler = SemanticFrameCompiler(settings=settings, allowed_tools=["sql.query"])
    cases = [
        ("count of patients in dicom", "patients", 'FROM flathr."Patient"', "sql.count_entity"),
        ("count of studies in dicom", "studies", 'FROM flathr."Study"', "sql.count_entity"),
        ("count of series in dicom", "series", 'FROM flathr."Series"', "sql.count_entity"),
        ("count of patients over age 45 in dicom", "patients", '"PatientBirthDate"', "sql.count_filtered_entity"),
    ]

    for goal, entity, expected_sql, expected_action in cases:
        filters = [{"field": "database", "operator": "eq", "value": "dicom"}]
        if "age 45" in goal:
            filters.append({"field": "age", "operator": "gt", "value": 45})
        frame = SemanticFrame(
            domain="sql",
            intent="count",
            entity="dicom",
            filters=filters,
            output=SemanticOutputContract(kind="scalar", cardinality="single", render_style="scalar"),
        )
        canonical = SemanticFrameCanonicalizer(settings).canonicalize(frame, goal=goal, allow_prompt_inference=False)
        compiled = compiler.compile(canonical)

        assert canonical.entity == entity
        assert compiled is not None
        assert compiled.metadata["semantic_action"] == expected_action
        assert compiled.plan.steps[0].action == "sql.query"
        assert expected_sql in compiled.plan.steps[0].args["query"]


def test_prompt_grounding_fills_filesystem_read_and_search_filters(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    compiler = SemanticFrameCompiler(settings=settings, allowed_tools=["fs.read", "fs.search_content"])
    read_frame = SemanticFrame(
        domain="filesystem",
        intent="list",
        output=SemanticOutputContract(kind="text", cardinality="sectioned", render_style="sectioned"),
    )
    search_frame = SemanticFrame(
        domain="filesystem",
        intent="list",
        entity="content",
        output=SemanticOutputContract(kind="table", cardinality="collection", render_style="record_table"),
    )

    canonical_read = SemanticFrameCanonicalizer(settings).canonicalize(
        read_frame,
        goal="Read README.md if it exists, summarize the project purpose, and list the commands mentioned in it.",
        allow_prompt_inference=False,
    )
    canonical_search = SemanticFrameCanonicalizer(settings).canonicalize(
        search_frame,
        goal="Search this repository for references to sql.query and summarize where it is used.",
        allow_prompt_inference=False,
    )
    compiled_read = compiler.compile(canonical_read)
    compiled_search = compiler.compile(canonical_search)

    assert compiled_read is not None
    assert compiled_read.metadata["semantic_action"] == "filesystem.read_file"
    assert compiled_read.plan.steps[0].action == "fs.read"
    assert compiled_read.plan.steps[0].args["path"] == "README.md"
    assert compiled_search is not None
    assert compiled_search.metadata["semantic_action"] == "filesystem.search_content"
    assert compiled_search.plan.steps[0].action == "fs.search_content"
    assert compiled_search.plan.steps[0].args["needle"] == "sql.query"


def test_matrix_compiles_filesystem_size_path_summary(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    compiler = SemanticFrameCompiler(settings=settings, allowed_tools=["fs.aggregate"])
    frame = SemanticFrame(
        domain="filesystem",
        intent="aggregate_metric",
        entity="directory",
        filters=[{"field": "path", "operator": "eq", "value": "."}],
        output=SemanticOutputContract(kind="table", cardinality="multi_scalar", render_style="metric_table"),
    )

    compiled = compiler.compile(frame)

    assert compiled is not None
    assert compiled.metadata["semantic_action"] == "filesystem.size_path"
    assert compiled.plan.steps[0].action == "fs.aggregate"
    assert compiled.plan.steps[0].args["aggregate"] == "count_and_total_size"
    assert compiled.plan.steps[1].args["source"]["path"] == "summary_text"


def test_prompt_grounding_compiles_sql_schema_answers(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    compiler = SemanticFrameCompiler(settings=settings, allowed_tools=["sql.schema"])
    cases = [
        ("List the DICOM database tables and summarize what each table represents.", "tables", "sql.schema_list_tables"),
        ("Show DICOM columns related to MRN, patient ID, study UID, series UID, and modality.", "schema", "sql.schema_describe_entity"),
        ("Summarize relationships between patient, study, series, and instance tables in dicom.", "relationships", "sql.schema_join_path"),
    ]

    for goal, entity, action_id in cases:
        frame = SemanticFrame(
            domain="sql",
            intent="schema_answer",
            output=SemanticOutputContract(kind="text", cardinality="sectioned", render_style="sectioned"),
        )
        canonical = SemanticFrameCanonicalizer(settings).canonicalize(frame, goal=goal, allow_prompt_inference=False)
        compiled = compiler.compile(canonical)

        assert canonical.entity == entity
        assert compiled is not None
        assert compiled.metadata["semantic_action"] == action_id
        assert compiled.plan.steps[0].action == "sql.schema"


def test_prompt_grounding_compiles_top_patients_by_study_count(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    compiler = SemanticFrameCompiler(settings=settings, allowed_tools=["sql.query"])
    frame = SemanticFrame(
        domain="sql",
        intent="list",
        entity="patients",
        filters=[{"field": "database", "operator": "eq", "value": "dicom"}],
        output=SemanticOutputContract(kind="table", cardinality="collection", render_style="record_table", result_entities=["patients"]),
    )

    canonical = SemanticFrameCanonicalizer(settings).canonicalize(
        frame,
        goal="Show the top 10 patients by number of studies in dicom.",
        allow_prompt_inference=False,
    )
    decision = select_semantic_action_decision(canonical)
    compiled = compiler.compile(canonical)

    assert decision.row is not None
    assert decision.row.semantic_action_id == "sql.top_n"
    assert compiled is not None
    assert compiled.metadata["semantic_action"] == "sql.top_n"
    query = compiled.plan.steps[0].args["query"]
    assert 'COUNT(DISTINCT st."StudyInstanceUID") AS study_count' in query
    assert 'GROUP BY p."PatientID"' in query
    assert "LIMIT 10" in query


def test_prompt_grounding_compiles_top_related_dicom_counts(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    compiler = SemanticFrameCompiler(settings=settings, allowed_tools=["sql.query"])
    cases = [
        ("Show the top 10 studies with the most series in dicom.", "studies", "series_count", 'GROUP BY st."StudyInstanceUID"'),
        ("Show the top 10 series with the most instances in dicom.", "series", "instance_count", 'GROUP BY se."SeriesInstanceUID"'),
        ("Show the top 5 modalities by number of series in dicom.", "modalities", "series_count", 'GROUP BY se."Modality"'),
    ]

    for goal, entity, expected_count, expected_group in cases:
        frame = SemanticFrame(
            domain="sql",
            intent="list",
            filters=[{"field": "database", "operator": "eq", "value": "dicom"}],
            output=SemanticOutputContract(kind="table", cardinality="collection", render_style="record_table"),
        )
        canonical = SemanticFrameCanonicalizer(settings).canonicalize(frame, goal=goal, allow_prompt_inference=False)
        compiled = compiler.compile(canonical)

        assert canonical.entity == entity
        assert compiled is not None
        assert compiled.metadata["semantic_action"] == "sql.top_n"
        query = compiled.plan.steps[0].args["query"]
        assert expected_count in query
        assert expected_group in query


def test_prompt_grounding_compiles_grouped_modality_counts(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    compiler = SemanticFrameCompiler(settings=settings, allowed_tools=["sql.query"])
    frame = SemanticFrame(
        domain="sql",
        intent="count",
        entity="studies",
        dimensions=["modality"],
        filters=[{"field": "database", "operator": "eq", "value": "dicom"}],
        output=SemanticOutputContract(kind="table", cardinality="grouped", render_style="metric_table", group_by=["modality"]),
    )

    compiled = compiler.compile(frame)

    assert compiled is not None
    assert compiled.metadata["semantic_action"] == "sql.count_grouped"
    query = compiled.plan.steps[0].args["query"]
    assert 'se."Modality" AS modality' in query
    assert 'COUNT(DISTINCT st."StudyInstanceUID") AS study_count' in query


def test_prompt_grounding_compiles_age_decade_distribution(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    compiler = SemanticFrameCompiler(settings=settings, allowed_tools=["sql.query"])
    frame = SemanticFrame(
        domain="sql",
        intent="aggregate_metric",
        entity="patients",
        filters=[{"field": "database", "operator": "eq", "value": "dicom"}],
        output=SemanticOutputContract(kind="table", cardinality="grouped", render_style="metric_table"),
    )

    canonical = SemanticFrameCanonicalizer(settings).canonicalize(
        frame,
        goal="Show the age distribution of patients in dicom grouped into decades.",
        allow_prompt_inference=False,
    )
    compiled = compiler.compile(canonical)

    assert canonical.intent == "count"
    assert "decade" in canonical.dimensions
    assert compiled is not None
    assert compiled.metadata["semantic_action"] == "sql.count_grouped"
    query = compiled.plan.steps[0].args["query"]
    assert "age_decade" in query
    assert 'p."PatientBirthDate"' in query


def test_prompt_grounding_compiles_study_modality_set_lists(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    compiler = SemanticFrameCompiler(settings=settings, allowed_tools=["sql.query"])
    cases = [
        ("Show studies that contain both RTPLAN and RTDOSE records in dicom.", "RTPLAN", "RTDOSE", "EXISTS"),
        ("Show studies that contain RTSTRUCT but no RTDOSE in dicom.", "RTSTRUCT", "RTDOSE", "NOT EXISTS"),
    ]

    for goal, included, other, expected_clause in cases:
        frame = SemanticFrame(
            domain="sql",
            intent="list",
            entity="studies",
            filters=[{"field": "database", "operator": "eq", "value": "dicom"}],
            output=SemanticOutputContract(kind="table", cardinality="collection", render_style="record_table", result_entities=["studies"]),
        )
        canonical = SemanticFrameCanonicalizer(settings).canonicalize(frame, goal=goal, allow_prompt_inference=False)
        compiled = compiler.compile(canonical)

        assert compiled is not None
        query = compiled.plan.steps[0].args["query"]
        assert included in query
        assert other in query
        assert expected_clause in query


def test_semantic_coverage_rejects_wrong_tool_for_inventory_frames(tmp_path: Path) -> None:
    frame = SemanticFrame(
        domain="slurm",
        intent="list",
        entity="nodes",
        output=SemanticOutputContract(kind="table", cardinality="collection", render_style="record_table"),
    )
    wrong = SemanticFrameCompiler(settings=_settings(tmp_path), allowed_tools=["slurm.queue"])._compile_single_tool_frame(
        frame,
        action="slurm.queue",
        args={},
        output_alias="wrong_queue",
        source_path="jobs",
        strategy="single_target_pushdown",
    )

    assert wrong is not None
    coverage = SemanticCoverageValidator().validate(frame, wrong.plan)
    assert coverage.covered is False
    assert "slurm.list_nodes" in coverage.errors[0]


def test_matrix_compiles_read_only_docker_inspection(tmp_path: Path) -> None:
    frame = SemanticFrame(
        domain="shell",
        intent="list",
        entity="docker",
        output=SemanticOutputContract(kind="table", cardinality="collection", render_style="record_table"),
    )

    compiled = SemanticFrameCompiler(settings=_settings(tmp_path), allowed_tools=["shell.exec"]).compile(frame)

    assert compiled is not None
    assert compiled.metadata["semantic_action"] == "shell.docker_list_containers_readonly"
    assert compiled.plan.steps[0].action == "shell.exec"
    assert compiled.plan.steps[0].args["command"] == "docker ps"


def test_semantic_action_matrix_has_no_duplicate_ids() -> None:
    ids = [row.semantic_action_id for row in SEMANTIC_ACTION_MATRIX]

    assert len(ids) == len(set(ids))
