from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan, PlannerConfig
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
    assert first.args["__semantic_projection"]["values"] == ["slicer", "totalseg"]
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


def test_task_planner_semantic_frame_skips_wrong_tool_and_llm_for_supported_slurm_prompt(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    planner = TaskPlanner(llm=FakeLLM(), tools=build_tool_registry(settings), settings=settings)

    plan = planner.build_plan(
        goal="Give me an average of the job times in slicer , totalseg partitions only consider jobs for the last 7 days",
        planner=PlannerConfig(),
        allowed_tools=["slurm.accounting_aggregate", "slurm.queue"],
        input_payload={},
    )

    assert [step.action for step in plan.steps] == ["slurm.accounting_aggregate", "text.format", "runtime.return"]
    assert plan.steps[0].args["group_by"] == "partition"
    assert plan.steps[0].args["__semantic_projection"]["values"] == ["slicer", "totalseg"]
    assert planner.last_capability_name == "semantic_frame"
    assert planner.last_llm_calls == 0
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
        {"__semantic_projection": {"field": "partition", "values": ["slicer", "totalseg"]}},
        raw,
    )

    assert [row["key"] for row in projected["groups"]] == ["slicer", "totalseg"]
    assert projected["job_count"] == 5
    assert projected["sum_elapsed_seconds"] == 80
    assert projected["average_elapsed_seconds"] == 16
    assert projected["min_elapsed_seconds"] == 8
    assert projected["max_elapsed_seconds"] == 40


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
                "output": {"kind": "table"},
            }
        ]
    )

    result = LLMSemanticFrameExtractor(llm=llm, settings=settings).extract(
        goal="average job runtime for slicer and totalseg partitions for last 7 days",
        allowed_tools=["slurm.accounting_aggregate"],
    )

    assert result.matched
    prompt = llm.user_prompts[0]
    assert "raw rows" not in prompt.lower()
    assert "stdout" not in prompt.lower()
    assert "stderr" not in prompt.lower()
    assert "patientid" not in prompt.lower()
    assert "allowed_metrics" in prompt


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
                "output": {"kind": "table"},
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
