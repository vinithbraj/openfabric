from __future__ import annotations

from aor_runtime.core.contracts import ExecutionStep, StepLog
from aor_runtime.runtime.presentation import PresentationContext, present_result
from aor_runtime.runtime.validator import RuntimeValidator


def _duplicated_node_rows() -> list[dict[str, str]]:
    return [
        {"name": "node-a", "state": "drain", "partition": "hpc", "cpus": "16", "memory": "1", "gres": "gpu:1"},
        {"name": "node-a", "state": "drain", "partition": "gpu", "cpus": "16", "memory": "1", "gres": "gpu:1"},
        {"name": "node-b", "state": "drain", "partition": "hpc", "cpus": "32", "memory": "1", "gres": "gpu:2"},
    ]


def test_problematic_nodes_dedupe_partition_rows_for_user_output() -> None:
    result = {
        "nodes": _duplicated_node_rows(),
        "count": 3,
        "partition_row_count": 3,
        "unique_count": 2,
        "summary": {"idle": 0, "allocated": 0, "mixed": 0, "down": 0, "drained": 3, "other": 0},
        "filters": {"state_group": "problematic"},
    }

    rendered = present_result(result, PresentationContext(mode="user", source_action="slurm.nodes"))

    assert rendered.markdown.startswith("## Problematic Nodes")
    assert "\n# " not in rendered.markdown
    assert "2 unique problematic nodes" in rendered.markdown
    assert "3 partition rows" in rendered.markdown
    assert "| node-a | drain | hpc, gpu | gpu:1 |" in rendered.markdown
    assert "Node rows include 3 partition rows for 2 unique nodes." in rendered.markdown


def test_live_node_validator_accepts_duplicated_problematic_partition_rows(tmp_path) -> None:
    result = {
        "nodes": _duplicated_node_rows(),
        "count": 3,
        "partition_row_count": 3,
        "unique_count": 2,
        "summary": {"idle": 0, "allocated": 0, "mixed": 0, "down": 0, "drained": 3, "other": 0},
        "filters": {"state_group": "problematic"},
    }
    log = StepLog(
        step=ExecutionStep(id=1, action="slurm.nodes", args={"state_group": "problematic"}),
        result=result,
        success=True,
    )

    validation, checks = RuntimeValidator().validate([log], goal="show problematic nodes")

    assert validation.success is True
    assert checks[0]["detail"].startswith("slurm nodes semantic validation passed")
