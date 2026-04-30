from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionStep, StepLog
from aor_runtime.runtime.capabilities.base import ClassificationContext, CompileContext
from aor_runtime.runtime.capabilities.slurm import SlurmCapabilityPack
from aor_runtime.runtime.presentation import PresentationContext, present_result
from aor_runtime.runtime.result_shape import validate_result_shape


SLURM_ALLOWED_TOOLS = [
    "slurm.queue",
    "slurm.job_detail",
    "slurm.nodes",
    "slurm.node_detail",
    "slurm.partitions",
    "slurm.accounting",
    "slurm.metrics",
    "slurm.slurmdbd_health",
]


def _settings(tmp_path: Path) -> Settings:
    return Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")


def _classify(prompt: str, tmp_path: Path):
    pack = SlurmCapabilityPack()
    result = pack.classify(prompt, ClassificationContext(schema_payload=None, allowed_tools=SLURM_ALLOWED_TOOLS, settings=_settings(tmp_path)))
    assert result.matched is True
    return pack, result


def test_running_jobs_by_partition_compiles_to_filtered_queue(tmp_path: Path) -> None:
    pack, result = _classify("show running jobs by partition", tmp_path)
    plan = pack.compile(result.intent, CompileContext(allowed_tools=SLURM_ALLOWED_TOOLS, settings=_settings(tmp_path)))

    assert plan is not None
    assert [step.action for step in plan.plan.steps] == ["slurm.queue", "runtime.return"]
    assert plan.plan.steps[0].args["state"] == "RUNNING"
    assert plan.plan.steps[0].args["group_by"] == "partition"


def test_pending_jobs_by_partition_compiles_to_filtered_queue(tmp_path: Path) -> None:
    pack, result = _classify("show pending jobs by partition", tmp_path)
    plan = pack.compile(result.intent, CompileContext(allowed_tools=SLURM_ALLOWED_TOOLS, settings=_settings(tmp_path)))

    assert plan is not None
    assert plan.plan.steps[0].args["state"] == "PENDING"
    assert plan.plan.steps[0].args["group_by"] == "partition"


def test_jobs_by_partition_presenter_renders_grouped_table() -> None:
    result = {
        "jobs": [],
        "count": 3,
        "total_count": 3,
        "returned_count": 3,
        "limit": None,
        "truncated": False,
        "filters": {"state": "RUNNING"},
        "group_by": "partition",
        "grouped": {"hpc": 2, "totalseg": 1},
    }

    rendered = present_result(result, PresentationContext(mode="user", source_action="slurm.queue"))

    assert rendered.markdown.startswith("## Running Jobs by Partition")
    assert "| hpc | 2 |" in rendered.markdown
    assert "| totalseg | 1 |" in rendered.markdown


def test_grouped_job_count_result_shape_passes_for_partition_counts() -> None:
    history = [
        StepLog(
            step=ExecutionStep(id=1, action="slurm.queue", args={"group_by": "partition"}, output="jobs_by_partition"),
            result={
                "jobs": [],
                "count": 259,
                "total_count": 259,
                "returned_count": 100,
                "limit": 100,
                "truncated": True,
                "filters": {"state": None, "partition": None},
                "group_by": "partition",
                "grouped": {"slicer": 8, "totalseg": 251},
            },
            success=True,
        ),
        StepLog(
            step=ExecutionStep(id=2, action="runtime.return", args={"value": "$jobs_by_partition.grouped"}, output="final"),
            result={"output": "| Group | Count |", "value": {"slicer": 8, "totalseg": 251}},
            success=True,
        ),
    ]

    result = validate_result_shape("count of jobs in each slurm partition", history, final_content="| Group | Count |\n| slicer | 8 |\n| totalseg | 251 |")

    assert result.success is True


def test_grouped_job_count_result_shape_fails_without_grouping() -> None:
    history = [
        StepLog(
            step=ExecutionStep(id=1, action="slurm.queue", args={}, output="jobs"),
            result={"jobs": [{"job_id": "1"}], "count": 1, "total_count": 1, "group_by": None, "grouped": None},
            success=True,
        ),
        StepLog(
            step=ExecutionStep(id=2, action="runtime.return", args={"value": "$jobs.count"}, output="final"),
            result={"output": "1", "value": 1},
            success=True,
        ),
    ]

    result = validate_result_shape("count of jobs in each slurm partition", history)

    assert result.success is False
    assert result.metadata["final_output_validation"] == "grouped_count_missing"
