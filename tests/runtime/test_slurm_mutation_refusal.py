from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.runtime.capabilities.base import ClassificationContext, CompileContext
from aor_runtime.runtime.capabilities.slurm import SlurmCapabilityPack
from aor_runtime.runtime.presentation import PresentationContext, present_result


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


def test_cancel_prompt_refuses_at_compile_time_without_inspection(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    pack = SlurmCapabilityPack()
    result = pack.classify(
        "cancel all pending jobs in slurm",
        ClassificationContext(schema_payload=None, allowed_tools=SLURM_ALLOWED_TOOLS, settings=settings),
    )
    plan = pack.compile(result.intent, CompileContext(allowed_tools=SLURM_ALLOWED_TOOLS, settings=settings))

    assert plan is not None
    assert [step.action for step in plan.plan.steps] == ["runtime.return"]
    rendered = present_result(
        "This runtime supports read-only SLURM inspection and metrics only. Unsupported operation: cancel.",
        PresentationContext(mode="user"),
    )
    assert "read-only SLURM inspection" in rendered.markdown
