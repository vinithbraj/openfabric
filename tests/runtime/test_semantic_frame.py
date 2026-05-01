from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan, ExecutionStep, PlannerConfig
from aor_runtime.runtime.dataflow import resolve_execution_step
from aor_runtime.runtime.llm_recursion import LLMStageRecursionBudget
from aor_runtime.runtime.semantic_frame import (
    LLMSemanticFrameExtractor,
    SemanticCoverageValidator,
    SemanticFilter,
    SemanticFrame,
    SemanticFrameCanonicalizer,
    SemanticFrameCompiler,
    SemanticFramePlanner,
    SemanticOutputContract,
    SemanticTargetSet,
    deterministic_semantic_frame,
    project_semantic_result,
)
from aor_runtime.runtime.semantic.compilers.slurm import compile_slurm_frame
from aor_runtime.runtime.semantic.policies.slurm import slurm_accounting_state_policy, slurm_all_states_phrase
from aor_runtime.runtime.planner import TaskPlanner
from aor_runtime.tools.factory import build_tool_registry


class FakeLLM:
    def __init__(self, responses: list[str | dict[str, Any]] | None = None) -> None:
        self.responses = [json.dumps(item) if isinstance(item, dict) else item for item in (responses or [])]
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    def complete(self, *, system_prompt: str, user_prompt: str, **_: Any) -> str:
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        if not self.responses:
            raise AssertionError("LLM called more times than expected")
        return self.responses.pop(0)


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db", **overrides)


def test_semantic_frame_public_facade_keeps_core_imports_available() -> None:
    assert SemanticFrame(domain="slurm").domain == "slurm"
    assert SemanticFilter(field="State", value="COMPLETED").field == "state"
    assert SemanticFrameCompiler is not None
    assert deterministic_semantic_frame is not None


def test_semantic_filter_accepts_list_values_only_for_in_operator() -> None:
    filter_ = SemanticFilter(field="partition", operator="in", value=["slicer", "totalseg"])

    assert filter_.field == "partition"
    assert filter_.value == ["slicer", "totalseg"]

    try:
        SemanticFilter(field="partition", operator="eq", value=["slicer", "totalseg"])
    except ValueError as exc:
        assert "operator='in'" in str(exc)
    else:
        raise AssertionError("Expected list-valued eq filter to fail")


def test_semantic_frame_normalizes_null_list_fields() -> None:
    frame = SemanticFrame.model_validate(
        {
            "domain": "filesystem",
            "intent": "count",
            "dimensions": None,
            "filters": None,
            "children": None,
            "output": {
                "kind": "table",
                "cardinality": "grouped",
                "render_style": "metric_table",
                "group_by": None,
                "result_entities": None,
            },
        }
    )

    assert frame.dimensions == []
    assert frame.filters == []
    assert frame.children == []
    assert frame.output.group_by == []
    assert frame.output.result_entities == []


def test_slurm_policy_module_exposes_all_state_semantics(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = deterministic_semantic_frame("average runtime for all jobs in totalseg partition", settings).frame
    assert frame is not None

    assert slurm_all_states_phrase("average runtime for all jobs in totalseg partition") == "all jobs"
    policy = slurm_accounting_state_policy(frame)
    assert policy.include_all_states is True
    assert policy.state is None
    assert policy.default_state_applied is False


def test_deterministic_sql_frame_extracts_dicom_text_concepts_as_scalar(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    result = deterministic_semantic_frame(
        "count of patieints that have both brain and breast related studies in dicom",
        settings,
    )

    assert result.matched
    assert result.frame is not None
    assert result.frame.domain == "sql"
    assert result.frame.intent == "count"
    assert result.frame.output.kind == "scalar"
    assert [filter_.value for filter_ in result.frame.filters if filter_.field == "concept_term"] == ["brain", "breast"]
    assert result.frame.filters[-1].field == "match_policy"
    assert result.frame.filters[-1].value == "all_terms"


def test_deterministic_sql_frame_keeps_single_modality_count_scalar(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    result = deterministic_semantic_frame("count of patients that have brain in RTPLAN in dicom", settings)

    assert result.matched
    assert result.frame is not None
    assert result.frame.output.kind == "scalar"
    assert result.frame.targets["modality"].values == ["RTPLAN"]
    assert [filter_.value for filter_ in result.frame.filters if filter_.field == "concept_term"] == ["brain"]


def test_deterministic_sql_frame_keeps_explicit_grouped_modality_table(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    result = deterministic_semantic_frame("count patients by RTPLAN and RTDOSE modality in dicom", settings)

    assert result.matched
    assert result.frame is not None
    assert result.frame.output.kind == "table"
    assert result.frame.dimensions == ["modality"]
    assert result.frame.targets["modality"].values == ["RTPLAN", "RTDOSE"]


def test_deterministic_sql_frame_extracts_multi_entity_counts_as_table(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    result = deterministic_semantic_frame("count of all patients, studies, series, RTPLANS in dicom", settings)

    assert result.matched
    assert result.frame is not None
    assert result.frame.domain == "sql"
    assert result.frame.intent == "count"
    assert result.frame.entity == "dicom_counts"
    assert result.frame.output.kind == "table"
    assert result.frame.output.cardinality == "multi_scalar"
    assert result.frame.output.render_style == "metric_table"
    assert result.frame.output.result_entities == ["patients", "studies", "series", "rtplan"]
    assert result.frame.dimensions == []
    assert result.frame.targets["entity"].values == ["patients", "studies", "series", "rtplan"]


def test_semantic_sql_compiler_builds_patient_concept_cooccurrence_query(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = deterministic_semantic_frame(
        "count of patieints that have both brain and breast related studies in dicom",
        settings,
    ).frame
    assert frame is not None

    compiled = SemanticFrameCompiler(settings=settings, allowed_tools=["sql.query"]).compile(frame)

    assert compiled is not None
    query = compiled.plan.steps[0].args["query"]
    assert [step.action for step in compiled.plan.steps] == ["sql.query", "text.format", "runtime.return"]
    assert 'COUNT(DISTINCT p."PatientID") AS patient_count' in query
    assert query.count("EXISTS") == 2
    assert "brain" in query
    assert "breast" in query
    assert 'p."PatientID"' in query
    assert "GROUP BY" not in query.upper()


def test_semantic_sql_compiler_builds_multi_entity_count_query(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = deterministic_semantic_frame("count of all patients, studies, series, RTPLANS in dicom", settings).frame
    assert frame is not None

    compiled = SemanticFrameCompiler(settings=settings, allowed_tools=["sql.query"]).compile(frame)

    assert compiled is not None
    query = compiled.plan.steps[0].args["query"]
    assert '(SELECT COUNT(*) FROM flathr."Patient") AS patient_count' in query
    assert '(SELECT COUNT(*) FROM flathr."Study") AS study_count' in query
    assert '(SELECT COUNT(*) FROM flathr."Series") AS series_count' in query
    assert "rtplan_count" in query
    assert [step.action for step in compiled.plan.steps] == ["sql.query", "text.format", "runtime.return"]


def test_semantic_sql_compiler_builds_single_patient_count_query(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = SemanticFrame(
        domain="sql",
        intent="count",
        entity="patients",
        metric={"name": "count"},
        filters=[{"field": "database", "value": "dicom"}],
        output=SemanticOutputContract(kind="scalar", cardinality="single", render_style="scalar", result_entities=["patients"]),
    )

    compiled = SemanticFrameCompiler(settings=settings, allowed_tools=["sql.query"]).compile(frame)

    assert compiled is not None
    assert compiled.plan.steps[0].args["query"] == 'SELECT COUNT(*) AS patient_count FROM flathr."Patient"'


def test_semantic_sql_compiler_builds_single_modality_count_query(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = SemanticFrame(
        domain="sql",
        intent="count",
        entity="rtplan",
        metric={"name": "count"},
        filters=[{"field": "database", "value": "dicom"}],
        output=SemanticOutputContract(kind="scalar", cardinality="single", render_style="scalar", result_entities=["rtplan"]),
    )

    compiled = SemanticFrameCompiler(settings=settings, allowed_tools=["sql.query"]).compile(frame)

    assert compiled is not None
    assert compiled.plan.steps[0].args["query"] == "SELECT COUNT(*) AS rtplan_count FROM flathr.\"Series\" se WHERE se.\"Modality\" = 'RTPLAN'"


def test_semantic_sql_compiler_builds_single_modality_scalar_query(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = deterministic_semantic_frame("count of patients that have brain in RTPLAN in dicom", settings).frame
    assert frame is not None

    compiled = SemanticFrameCompiler(settings=settings, allowed_tools=["sql.query"]).compile(frame)

    assert compiled is not None
    query = compiled.plan.steps[0].args["query"]
    assert 'COUNT(DISTINCT p."PatientID") AS patient_count' in query
    assert 'se."Modality" IN (\'RTPLAN\')' in query
    assert "brain" in query
    assert "GROUP BY" not in query.upper()


def test_semantic_sql_compiler_builds_patient_age_count_query(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = SemanticFrame(
        domain="sql",
        intent="count",
        entity="patients",
        metric={"name": "count"},
        filters=[
            {"field": "database", "value": "dicom"},
            {"field": "age", "operator": "gt", "value": 45},
        ],
        output=SemanticOutputContract(kind="scalar", cardinality="single", render_style="scalar", result_entities=["patients"]),
    )

    compiled = SemanticFrameCompiler(settings=settings, allowed_tools=["sql.query"]).compile(frame)

    assert compiled is not None
    query = compiled.plan.steps[0].args["query"]
    assert 'COUNT(DISTINCT p."PatientID") AS patient_count' in query
    assert 'FROM flathr."Patient" p' in query
    assert 'p."PatientBirthDate" < CURRENT_DATE - INTERVAL \'45 years\'' in query
    assert "GROUP BY" not in query.upper()


def test_semantic_sql_compiler_combines_age_with_modality_patient_count(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = SemanticFrame(
        domain="sql",
        intent="count",
        entity="patients",
        metric={"name": "count"},
        targets={"modality": ["RTPLAN"]},
        filters=[
            {"field": "database", "value": "dicom"},
            {"field": "age", "operator": "gte", "value": "45"},
        ],
        output=SemanticOutputContract(kind="scalar", cardinality="single", render_style="scalar", result_entities=["patients"]),
    )

    compiled = SemanticFrameCompiler(settings=settings, allowed_tools=["sql.query"]).compile(frame)

    assert compiled is not None
    query = compiled.plan.steps[0].args["query"]
    assert 'JOIN flathr."Study" st ON st."PatientID" = p."PatientID"' in query
    assert 'JOIN flathr."Series" se ON se."StudyInstanceUID" = st."StudyInstanceUID"' in query
    assert 'p."PatientBirthDate" <= CURRENT_DATE - INTERVAL \'45 years\'' in query
    assert 'se."Modality" IN (\'RTPLAN\')' in query


def test_slurm_domain_compiler_module_preserves_all_jobs_policy(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = deterministic_semantic_frame("average runtime for all jobs in totalseg partition", settings).frame
    assert frame is not None

    compiled = compile_slurm_frame(frame, settings=settings, allowed_tools=["slurm.accounting_aggregate"])

    assert compiled is not None
    assert compiled.plan.steps[0].args["include_all_states"] is True
    assert "state" not in compiled.plan.steps[0].args


def test_deterministic_frame_extracts_multi_partition_slurm_aggregate(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    result = deterministic_semantic_frame(
        "Give me an average of the job times in slicer, totalseg partitions only consider jobs for the last 7 days",
        settings,
    )

    assert result.matched
    assert result.frame is not None
    assert result.frame.domain == "slurm"
    assert result.frame.intent == "aggregate_metric"
    assert result.frame.metric is not None
    assert result.frame.metric.name == "average_elapsed"
    assert result.frame.targets["partition"].model_dump()["values"] == ["slicer", "totalseg"]
    assert result.frame.time_window is not None
    assert result.frame.time_window.label == "Last 7 days"
    assert result.frame.output.kind == "table"


def test_semantic_frame_compiler_uses_grouped_partition_pushdown(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = deterministic_semantic_frame(
        "average job times in slicer, totalseg partitions for the last 7 days",
        settings,
    ).frame
    assert frame is not None

    compiled = SemanticFrameCompiler(
        settings=settings,
        allowed_tools=["slurm.accounting_aggregate"],
    ).compile(frame)

    assert compiled is not None
    assert compiled.strategy == "grouped_pushdown"
    first = compiled.plan.steps[0]
    assert first.action == "slurm.accounting_aggregate"
    assert first.args["metric"] == "average_elapsed"
    assert first.args["group_by"] == "partition"
    assert "__semantic_projection" not in first.args
    assert "limit" not in first.args
    assert first.metadata["semantic_projection"]["values"] == ["slicer", "totalseg"]
    assert first.args["start"].endswith("00:00:00")


def test_slurm_all_jobs_aggregate_uses_all_state_policy(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = deterministic_semantic_frame(
        "what is the average time taken for all jobs in the totalseg partition, only consider jobs for the past 7 days",
        settings,
    ).frame
    assert frame is not None

    compiled = SemanticFrameCompiler(settings=settings, allowed_tools=["slurm.accounting_aggregate"]).compile(frame)

    assert compiled is not None
    first = compiled.plan.steps[0]
    assert first.action == "slurm.accounting_aggregate"
    assert first.args["partition"] == "totalseg"
    assert first.args["include_all_states"] is True
    assert "state" not in first.args
    assert "default_state_applied" not in first.args


def test_slurm_all_jobs_multi_partition_projection_uses_step_metadata(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = deterministic_semantic_frame(
        "what is the average time taken for all jobs in the totalseg and slicer partition, only consider jobs for the past 7 days",
        settings,
    ).frame
    assert frame is not None

    compiled = SemanticFrameCompiler(settings=settings, allowed_tools=["slurm.accounting_aggregate"]).compile(frame)

    assert compiled is not None
    first = compiled.plan.steps[0]
    assert first.action == "slurm.accounting_aggregate"
    assert first.args["group_by"] == "partition"
    assert first.args["include_all_states"] is True
    assert "state" not in first.args
    assert "limit" not in first.args
    assert "__semantic_projection" not in first.args
    assert first.metadata["semantic_projection"] == {
        "field": "partition",
        "values": ["totalseg", "slicer"],
        "source": "semantic_frame",
    }


def test_slurm_unspecified_jobs_aggregate_preserves_completed_default(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = deterministic_semantic_frame(
        "what is the average time taken for jobs in the totalseg partition, only consider jobs for the past 7 days",
        settings,
    ).frame
    assert frame is not None

    compiled = SemanticFrameCompiler(settings=settings, allowed_tools=["slurm.accounting_aggregate"]).compile(frame)

    assert compiled is not None
    first = compiled.plan.steps[0]
    assert first.args["state"] == "COMPLETED"
    assert first.args["default_state_applied"] is True
    assert "include_all_states" not in first.args


def test_slurm_explicit_completed_aggregate_keeps_completed_state(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = deterministic_semantic_frame(
        "average time for completed jobs in the totalseg partition for the past 7 days",
        settings,
    ).frame
    assert frame is not None

    compiled = SemanticFrameCompiler(settings=settings, allowed_tools=["slurm.accounting_aggregate"]).compile(frame)

    assert compiled is not None
    first = compiled.plan.steps[0]
    assert first.args["state"] == "COMPLETED"
    assert "default_state_applied" not in first.args
    assert "include_all_states" not in first.args


def test_slurm_completed_negation_aggregate_uses_all_state_policy(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = deterministic_semantic_frame(
        "average time for jobs in totalseg partition, do not filter by completed and use all jobs",
        settings,
    ).frame
    assert frame is not None

    compiled = SemanticFrameCompiler(settings=settings, allowed_tools=["slurm.accounting_aggregate"]).compile(frame)

    assert compiled is not None
    first = compiled.plan.steps[0]
    assert first.args["include_all_states"] is True
    assert "state" not in first.args
    assert "default_state_applied" not in first.args


def test_task_planner_semantic_frame_uses_llm_first_for_supported_slurm_prompt(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    llm = FakeLLM(
        [
            {
                "domain": "slurm",
                "intent": "aggregate_metric",
                "entity": "jobs",
                "metric": {"name": "average_elapsed", "unit": "seconds"},
                "targets": {"partition": ["slicer", "totalseg"]},
                "filters": [{"field": "include_all_states", "value": True}],
                "time_window": {"phrase": "last 7 days"},
                "output": {
                    "kind": "table",
                    "cardinality": "grouped",
                    "render_style": "metric_table",
                    "group_by": ["partition"],
                    "result_entities": ["jobs"],
                },
            }
        ]
    )
    planner = TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)

    plan = planner.build_plan(
        goal="Give me an average of the job times in slicer , totalseg partitions only consider jobs for the last 7 days",
        planner=PlannerConfig(),
        allowed_tools=["slurm.accounting_aggregate", "slurm.queue"],
        input_payload={},
    )

    assert [step.action for step in plan.steps] == ["slurm.accounting_aggregate", "text.format", "runtime.return"]
    assert plan.steps[0].args["group_by"] == "partition"
    assert "__semantic_projection" not in plan.steps[0].args
    assert plan.steps[0].metadata["semantic_projection"]["values"] == ["slicer", "totalseg"]
    assert planner.last_capability_name == "semantic_frame"
    assert planner.last_llm_calls == 1
    assert planner.last_capability_metadata["semantic_frame_source"] == "llm"
    assert planner.last_capability_metadata["semantic_strategy"] == "grouped_pushdown"


def test_semantic_coverage_rejects_queue_for_runtime_aggregate(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = deterministic_semantic_frame(
        "average job times in slicer, totalseg partitions for the last 7 days",
        settings,
    ).frame
    assert frame is not None
    bad_plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {"id": 1, "action": "slurm.queue", "args": {}, "output": "jobs"},
                {"id": 2, "action": "runtime.return", "input": ["jobs"], "args": {"value": "$jobs"}, "output": "final"},
            ]
        }
    )

    result = SemanticCoverageValidator().validate(frame, bad_plan)

    assert not result.covered
    assert "slurm.accounting_aggregate" in result.errors[0]


def test_semantic_projection_filters_groups_and_recomputes_summary() -> None:
    raw = {
        "result_kind": "accounting_aggregate",
        "metric": "average_elapsed",
        "group_by": "partition",
        "job_count": 10,
        "total_count": 1200,
        "returned_count": 1000,
        "limit": 1000,
        "truncated": True,
        "groups": [
            {
                "key": "hpc",
                "job_count": 2,
                "average_elapsed_seconds": 5,
                "min_elapsed_seconds": 1,
                "max_elapsed_seconds": 9,
                "sum_elapsed_seconds": 10,
            },
            {
                "key": "slicer",
                "job_count": 2,
                "average_elapsed_seconds": 10,
                "min_elapsed_seconds": 8,
                "max_elapsed_seconds": 12,
                "sum_elapsed_seconds": 20,
            },
            {
                "key": "totalseg",
                "job_count": 3,
                "average_elapsed_seconds": 20,
                "min_elapsed_seconds": 10,
                "max_elapsed_seconds": 40,
                "sum_elapsed_seconds": 60,
            },
        ],
    }

    projected = project_semantic_result(
        "slurm.accounting_aggregate",
        {},
        raw,
        metadata={"semantic_projection": {"field": "partition", "values": ["slicer", "totalseg"]}},
    )

    assert [row["key"] for row in projected["groups"]] == ["slicer", "totalseg"]
    assert projected["job_count"] == 5
    assert projected["total_count"] == 5
    assert projected["returned_count"] == 5
    assert projected["limit"] is None
    assert projected["truncated"] is False
    assert projected["source_total_count"] == 1200
    assert projected["source_returned_count"] == 1000
    assert projected["source_limit"] == 1000
    assert projected["source_truncated"] is True
    assert projected["sum_elapsed_seconds"] == 80
    assert projected["average_elapsed_seconds"] == 16
    assert projected["min_elapsed_seconds"] == 8
    assert projected["max_elapsed_seconds"] == 40


def test_legacy_semantic_projection_args_are_sanitized_into_step_metadata() -> None:
    step = ExecutionStep.model_validate(
        {
            "id": 1,
            "action": "slurm.accounting_aggregate",
            "args": {
                "metric": "average_elapsed",
                "group_by": "partition",
                "__semantic_projection": {"field": "partition", "values": ["slicer", "totalseg"]},
                "__semantic_debug": {"raw": True},
            },
            "output": "aggregate",
        }
    )
    assert step.args == {"metric": "average_elapsed", "group_by": "partition"}
    assert step.metadata["semantic_projection"] == {"field": "partition", "values": ["slicer", "totalseg"]}

    resolved = resolve_execution_step(step, {})

    assert resolved.args == {"metric": "average_elapsed", "group_by": "partition"}
    assert resolved.metadata["semantic_projection"] == {"field": "partition", "values": ["slicer", "totalseg"]}


def test_project_semantic_result_keeps_legacy_args_projection_compatibility() -> None:
    raw = {
        "result_kind": "accounting_aggregate",
        "metric": "average_elapsed",
        "group_by": "partition",
        "groups": [
            {"key": "slicer", "job_count": 1, "average_elapsed_seconds": 10},
            {"key": "other", "job_count": 1, "average_elapsed_seconds": 99},
        ],
    }

    projected = project_semantic_result(
        "slurm.accounting_aggregate",
        {"__semantic_projection": {"field": "partition", "values": ["slicer"]}},
        raw,
    )

    assert [row["key"] for row in projected["groups"]] == ["slicer"]


def test_llm_semantic_frame_extractor_rejects_executable_payload(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    llm = FakeLLM(
        [
            {
                "domain": "slurm",
                "intent": "aggregate_metric",
                "tool": "slurm.accounting_aggregate",
                "metric": {"name": "average_elapsed"},
            }
        ]
    )

    result = LLMSemanticFrameExtractor(llm=llm, settings=settings).extract(
        goal="average job runtime by partition",
        allowed_tools=["slurm.accounting_aggregate"],
    )

    assert not result.matched
    assert result.reason == "semantic_frame_contains_executable_payload"


def test_llm_semantic_frame_prompt_contains_only_safe_metadata(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    llm = FakeLLM(
        [
            {
                "domain": "slurm",
                "intent": "aggregate_metric",
                "entity": "jobs",
                "metric": {"name": "average_elapsed"},
                "targets": {"partition": ["slicer", "totalseg"]},
                "time_window": {"phrase": "last 7 days"},
                "output": {"kind": "table", "cardinality": "grouped", "render_style": "metric_table"},
            }
        ]
    )

    result = LLMSemanticFrameExtractor(llm=llm, settings=settings).extract(
        goal="average job runtime for slicer and totalseg partitions for last 7 days",
        allowed_tools=["slurm.accounting_aggregate"],
    )

    assert result.matched
    prompt = llm.user_prompts[0]
    assert "privacy_contract" in prompt
    assert "JobID" not in prompt
    assert "Elapsed" not in prompt
    assert "patientid" not in prompt.lower()
    assert "allowed_metrics" in prompt
    assert "allowed_output_cardinalities" in prompt
    assert "multi_scalar" in prompt
    assert "slurm_state_policy" not in prompt
    assert "state_policy" in prompt


def test_llm_semantic_frame_extractor_rejects_missing_output_intent(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    llm = FakeLLM(
        [
            {
                "domain": "filesystem",
                "intent": "count",
                "entity": "files",
                "targets": {"extension": ["csv", "json"]},
                "output": {"kind": "table"},
            }
        ]
    )

    result = LLMSemanticFrameExtractor(llm=llm, settings=settings).extract(
        goal="count CSV and JSON files separately",
        allowed_tools=["fs.aggregate"],
    )

    assert not result.matched
    assert result.reason == "semantic_frame_missing_output_cardinality"


def test_llm_semantic_frame_extractor_repairs_invalid_state_policy_field(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    llm = FakeLLM(
        [
            {
                "domain": "slurm",
                "intent": "aggregate_metric",
                "entity": "jobs",
                "metric": {"name": "average_elapsed"},
                "targets": {"partition": ["totalseg", "slicer"]},
                "slurm_state_policy": "include_all_states=true",
                "time_window": {"phrase": "last 7 days"},
                "output": {"kind": "table", "cardinality": "grouped", "render_style": "metric_table"},
            },
            {
                "domain": "slurm",
                "intent": "aggregate_metric",
                "entity": "jobs",
                "metric": {"name": "average_elapsed"},
                "targets": {"partition": ["totalseg", "slicer"]},
                "state_policy": {"include_all_states": True},
                "time_window": {"phrase": "last 7 days"},
                "output": {"kind": "table", "cardinality": "grouped", "render_style": "metric_table"},
            },
        ]
    )

    result = LLMSemanticFrameExtractor(llm=llm, settings=settings).extract(
        goal="average runtime for all jobs in totalseg and slicer partitions for last 7 days",
        allowed_tools=["slurm.accounting_aggregate"],
    )

    assert result.matched
    assert result.frame is not None
    assert result.frame.state_policy is not None
    assert result.frame.state_policy.include_all_states is True
    assert len(llm.user_prompts) == 2
    repair_prompt = llm.user_prompts[1]
    assert "repair_semantic_frame_json" in repair_prompt
    assert "pydantic.dev" not in repair_prompt
    assert "JobID" not in repair_prompt


def test_llm_semantic_frame_extractor_recursively_expands_compound_frames(tmp_path: Path) -> None:
    settings = _settings(tmp_path, semantic_frame_max_depth=10)
    llm = FakeLLM(
        [
            {
                "frame_id": "root",
                "composition": "sequence",
                "domain": "diagnostic",
                "intent": "diagnostic",
                "entity": "workspace_diagnostic",
                "output": {"kind": "text", "cardinality": "sectioned", "render_style": "sectioned"},
            },
            {
                "frame_id": "root_expanded",
                "composition": "sequence",
                "domain": "diagnostic",
                "intent": "diagnostic",
                "entity": "workspace_diagnostic",
                "output": {"kind": "text", "cardinality": "sectioned", "render_style": "sectioned"},
                "children": [
                    {
                        "domain": "filesystem",
                        "intent": "count",
                        "entity": "files",
                        "targets": {"extension": ["csv"]},
                        "output": {"kind": "scalar", "cardinality": "single", "render_style": "scalar"},
                        "rationale": "Count CSV files.",
                    },
                    {
                        "frame_id": "nested",
                        "composition": "sequence",
                        "domain": "diagnostic",
                        "intent": "diagnostic",
                        "entity": "system_diagnostic",
                        "output": {"kind": "text", "cardinality": "sectioned", "render_style": "sectioned"},
                        "rationale": "Inspect system health.",
                    },
                ],
            },
            {
                "domain": "shell",
                "intent": "status",
                "entity": "disk",
                "output": {"kind": "status", "cardinality": "single", "render_style": "key_value"},
                "rationale": "Inspect disk status.",
            },
        ]
    )

    result = LLMSemanticFrameExtractor(llm=llm, settings=settings).extract(
        goal="count CSV files, then inspect disk status",
        allowed_tools=["fs.aggregate", "shell.exec"],
    )

    assert result.matched
    assert result.frame is not None
    assert [child.domain for child in result.frame.children] == ["filesystem", "shell"]
    assert result.llm_calls == 3
    assert result.metadata["llm_stage_calls"]["semantic_frame"] == 3
    assert result.metadata["llm_recursion_depth"] == 2
    assert all('"llm_stage"' in prompt for prompt in llm.user_prompts)


def test_llm_stage_recursion_budget_accepts_depth_10_and_rejects_11(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    budget = LLMStageRecursionBudget(settings, stage="semantic_frame")

    assert budget.state(depth=10).remaining_depth == 0
    try:
        budget.state(depth=11)
    except ValueError as exc:
        assert "exceeds maximum 10" in str(exc)
    else:
        raise AssertionError("Expected recursion depth 11 to fail")


def test_semantic_frame_planner_returns_none_when_mode_off(tmp_path: Path) -> None:
    settings = _settings(tmp_path, semantic_frame_mode="off")
    planner = SemanticFramePlanner(
        settings=settings,
        llm=FakeLLM(),
        allowed_tools=["slurm.accounting_aggregate"],
    )

    result = planner.try_build_plan(
        goal="average job times in slicer, totalseg partitions for the last 7 days",
    )

    assert result is None


def test_slurm_grouped_count_frame_compiles_to_queue_group_by_partition(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = deterministic_semantic_frame(
        "count of jobs in each slurm partition separately",
        settings,
    ).frame
    assert frame is not None

    compiled = SemanticFrameCompiler(settings=settings, allowed_tools=["slurm.queue"]).compile(frame)

    assert compiled is not None
    assert compiled.strategy == "grouped_pushdown"
    assert compiled.plan.steps[0].action == "slurm.queue"
    assert compiled.plan.steps[0].args["group_by"] == "partition"
    assert compiled.plan.steps[1].args["source"]["path"] == "grouped"


def test_filesystem_multi_extension_count_compiles_to_fanout_table(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = deterministic_semantic_frame(
        "count CSV and JSON files under the current folder separately",
        settings,
    ).frame
    assert frame is not None

    compiled = SemanticFrameCompiler(settings=settings, allowed_tools=["fs.aggregate"]).compile(frame)

    assert compiled is not None
    assert compiled.strategy == "fan_out"
    actions = [step.action for step in compiled.plan.steps]
    assert actions[:2] == ["fs.aggregate", "fs.aggregate"]
    assert compiled.plan.steps[0].args["pattern"] == "*.csv"
    assert compiled.plan.steps[1].args["pattern"] == "*.json"
    rows = compiled.plan.steps[2].args["source"]
    assert rows[0]["extension"] == "csv"
    assert rows[0]["file_count"]["path"] == "file_count"


def test_shell_process_count_frame_compiles_to_safe_command(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = deterministic_semantic_frame("How many processes are running on this machine?", settings).frame
    assert frame is not None

    compiled = SemanticFrameCompiler(settings=settings, allowed_tools=["shell.exec"]).compile(frame)

    assert compiled is not None
    assert compiled.plan.steps[0].action == "shell.exec"
    assert compiled.plan.steps[0].args["command"] == "ps -eo pid= | wc -l"


def test_sql_validate_frame_compiles_without_query_execution(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    frame = deterministic_semantic_frame(
        "Validate this SQL against dicom: SELECT COUNT(*) FROM flathr.\"Patient\"",
        settings,
    ).frame
    assert frame is not None

    compiled = SemanticFrameCompiler(settings=settings, allowed_tools=["sql.validate"]).compile(frame)

    assert compiled is not None
    assert compiled.plan.steps[0].action == "sql.validate"
    assert compiled.plan.steps[0].args["database"] == "dicom"
    assert "SELECT COUNT" in compiled.plan.steps[0].args["query"]


def test_recursive_child_frames_compile_to_composite_plan(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    parent = SemanticFrame(
        domain="diagnostic",
        intent="diagnostic",
        composition="sequence",
        children=[
            SemanticFrame(domain="filesystem", intent="count", entity="files", targets={"extension": ["csv"]}),
            SemanticFrame(domain="shell", intent="count", entity="processes"),
        ],
        output=SemanticOutputContract(kind="text"),
    )
    frame = SemanticFrameCanonicalizer(settings).canonicalize(parent, goal="count CSV files then count processes")

    compiled = SemanticFrameCompiler(settings=settings, allowed_tools=["fs.aggregate", "shell.exec"]).compile(frame)

    assert compiled is not None
    assert [step.action for step in compiled.plan.steps][-2:] == ["text.format", "runtime.return"]
    assert any(step.action == "fs.aggregate" for step in compiled.plan.steps)
    assert any(step.action == "shell.exec" for step in compiled.plan.steps)


def test_recursive_frame_depth_limit_is_enforced(tmp_path: Path) -> None:
    settings = _settings(tmp_path, semantic_frame_max_depth=3)
    frame = SemanticFrame(
        domain="diagnostic",
        intent="diagnostic",
        children=[
            SemanticFrame(
                domain="diagnostic",
                intent="diagnostic",
                children=[
                    SemanticFrame(
                        domain="diagnostic",
                        intent="diagnostic",
                        children=[
                            SemanticFrame(
                                domain="diagnostic",
                                intent="diagnostic",
                                children=[SemanticFrame(domain="shell", intent="count", entity="processes")],
                            )
                        ],
                    )
                ],
            )
        ],
    )

    try:
        SemanticFrameCanonicalizer(settings).canonicalize(frame, goal="deep diagnostic")
    except ValueError as exc:
        assert "exceeds configured maximum" in str(exc)
    else:
        raise AssertionError("Expected depth limit validation failure")


def test_llm_semantic_frame_extraction_handles_non_slurm_candidate(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    llm = FakeLLM(
        [
            {
                "domain": "filesystem",
                "intent": "count",
                "entity": "files",
                "targets": {"extension": ["csv", "parquet"]},
                "output": {"kind": "table", "cardinality": "multi_scalar", "render_style": "metric_table"},
            }
        ]
    )

    result = LLMSemanticFrameExtractor(llm=llm, settings=settings).extract(
        goal="count CSV and parquet files separately",
        allowed_tools=["fs.aggregate"],
    )

    assert result.matched
    assert result.frame is not None
    assert result.frame.domain == "filesystem"
    assert result.frame.targets["extension"].model_dump()["values"] == ["csv", "parquet"]
