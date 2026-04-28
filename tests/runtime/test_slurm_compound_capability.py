from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.runtime.capabilities.base import ClassificationContext, CompileContext
from aor_runtime.runtime.capabilities.slurm import SlurmCapabilityPack


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
    return Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        available_nodes_raw="edge-1,edge-2",
        default_node="edge-1",
    )


def _plan_actions(prompt: str, tmp_path: Path) -> list[str]:
    pack = SlurmCapabilityPack()
    context = ClassificationContext(schema_payload=None, allowed_tools=SLURM_ALLOWED_TOOLS, settings=_settings(tmp_path))
    result = pack.classify(prompt, context)
    assert result.matched is True
    plan = pack.compile(result.intent, CompileContext(allowed_tools=SLURM_ALLOWED_TOOLS, settings=_settings(tmp_path)))
    assert plan is not None
    assert result.metadata["slurm_coverage_passed"] is True
    return [step.action for step in plan.plan.steps]


def test_cluster_status_compiles_to_read_only_compound(tmp_path: Path) -> None:
    actions = _plan_actions("What is the status of my SLURM cluster?", tmp_path)

    assert actions == ["slurm.metrics", "slurm.metrics", "slurm.metrics", "slurm.slurmdbd_health", "runtime.return"]
    assert "shell.exec" not in actions
    assert "python.exec" not in actions


def test_running_pending_and_problematic_nodes_compile_to_three_tools(tmp_path: Path) -> None:
    actions = _plan_actions("How many jobs are running and pending, and are there any problematic nodes?", tmp_path)

    assert actions == ["slurm.nodes", "slurm.queue", "slurm.queue", "runtime.return"]


def test_queue_node_gpu_summary_covers_all_requests(tmp_path: Path) -> None:
    actions = _plan_actions("Summarize queue, node, and GPU status.", tmp_path)

    assert actions == ["slurm.metrics", "slurm.nodes", "slurm.metrics", "runtime.return"]


def test_running_pending_and_problematic_nodes_as_json(tmp_path: Path) -> None:
    pack = SlurmCapabilityPack()
    context = ClassificationContext(schema_payload=None, allowed_tools=SLURM_ALLOWED_TOOLS, settings=_settings(tmp_path))
    result = pack.classify("Show running jobs, pending jobs, and problematic nodes as JSON.", context)
    plan = pack.compile(result.intent, CompileContext(allowed_tools=SLURM_ALLOWED_TOOLS, settings=_settings(tmp_path)))

    assert plan is not None
    assert [step.action for step in plan.plan.steps] == ["slurm.nodes", "slurm.queue", "slurm.queue", "runtime.return"]
    assert plan.plan.steps[-1].args["mode"] == "json"
    assert result.metadata["slurm_requests_missing"] == []


def test_mutation_refusal_uses_runtime_return_only(tmp_path: Path) -> None:
    pack = SlurmCapabilityPack()
    context = ClassificationContext(schema_payload=None, allowed_tools=SLURM_ALLOWED_TOOLS, settings=_settings(tmp_path))
    result = pack.classify("Cancel job 12345 in slurm.", context)
    plan = pack.compile(result.intent, CompileContext(allowed_tools=SLURM_ALLOWED_TOOLS, settings=_settings(tmp_path)))

    assert plan is not None
    assert [step.action for step in plan.plan.steps] == ["runtime.return"]
