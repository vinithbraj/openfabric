from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.runtime.capabilities.base import ClassificationContext
from aor_runtime.runtime.capabilities.slurm import SlurmCapabilityPack
from aor_runtime.runtime.slurm_semantics import extract_slurm_semantic_frame


SLURM_ALLOWED_TOOLS = ["slurm.accounting_aggregate", "slurm.queue", "runtime.return"]


def _context(tmp_path: Path) -> ClassificationContext:
    return ClassificationContext(
        schema_payload=None,
        allowed_tools=SLURM_ALLOWED_TOOLS,
        settings=Settings(
            workspace_root=tmp_path,
            run_store_path=tmp_path / "runtime.db",
            gateway_url="https://gateway.internal/exec",
            available_nodes_raw="edge-1",
            default_node="edge-1",
        ),
    )


def test_completed_jobs_is_explicit_state_filter(tmp_path: Path) -> None:
    result = SlurmCapabilityPack().classify("average runtime of completed jobs on totalseg partition", _context(tmp_path))

    assert result.intent.state == "COMPLETED"
    assert result.intent.include_all_states is False
    assert result.intent.default_state_applied is False


def test_runtime_prompt_defaults_completed_when_no_state_requested(tmp_path: Path) -> None:
    result = SlurmCapabilityPack().classify("average runtime on totalseg partition", _context(tmp_path))

    assert result.intent.state == "COMPLETED"
    assert result.intent.default_state_applied is True
    assert result.intent.include_all_states is False


def test_all_jobs_runtime_prompt_uses_all_states(tmp_path: Path) -> None:
    result = SlurmCapabilityPack().classify("average runtime of all jobs on totalseg partition", _context(tmp_path))

    assert result.intent.state is None
    assert result.intent.include_all_states is True
    assert result.intent.default_state_applied is False


def test_completed_negation_overrides_default_completed(tmp_path: Path) -> None:
    prompt = "how long on average did jobs take on totalseg partition, don't filter by completed, get all jobs"
    frame = extract_slurm_semantic_frame(prompt)
    result = SlurmCapabilityPack().classify(prompt, _context(tmp_path))

    assert frame.include_all_states is True
    assert [constraint.value for constraint in frame.negated_filters] == ["COMPLETED"]
    assert result.intent.state is None
    assert result.intent.include_all_states is True
    assert result.intent.default_state_applied is False
