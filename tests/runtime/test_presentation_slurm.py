from __future__ import annotations

import json
from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionStep, StepLog
from aor_runtime.runtime.executor import summarize_final_output
from aor_runtime.runtime.presentation import PresentationContext, present_result


def _settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db", **overrides)


def _compound_slurm_result() -> dict:
    return {
        "summary": {"request_count": 4, "tool_count": 4},
        "results": {
            "cluster_summary": {
                "metric_group": "cluster_summary",
                "payload": {
                    "queue_count": 195,
                    "running_jobs": 8,
                    "pending_jobs": 187,
                    "node_count": 18,
                    "mixed_nodes": 12,
                    "drained_nodes": 6,
                    "problematic_nodes": 6,
                    "gpu_available": True,
                    "total_gpus": 32,
                },
            },
            "node_summary": {
                "metric_group": "node_summary",
                "payload": {"node_count": 18, "by_state": {"mixed": 12, "drained": 6}},
            },
            "partition_summary": {
                "metric_group": "partition_summary",
                "payload": {
                    "partitions": [
                        {"partition": "hpc", "state": "mix", "nodes": "6", "cpus": "0/64/64/128", "gres": "gpu:a100:8"},
                        {"partition": "vllm", "state": "drain", "nodes": "4", "cpus": "0/64/64/128", "gres": "gpu:l40:4"},
                    ]
                },
            },
            "slurmdbd_health": {"available": True, "status": "ok", "message": "SLURM accounting is available."},
        },
        "coverage": {"covered_requests": ["r1"], "missing_requests": []},
        "slurm_semantic_frame": {"requests": ["internal"]},
    }


def test_slurm_compound_renders_markdown_without_internal_blocks() -> None:
    rendered = present_result(_compound_slurm_result(), PresentationContext(mode="user"))

    assert rendered.markdown.startswith("## SLURM Cluster Status")
    assert "\n# " not in rendered.markdown
    assert "Running: 8" in rendered.markdown
    assert "Pending: 187" in rendered.markdown
    assert "Problematic nodes: 6" in rendered.markdown
    assert "## SLURM Partition Summary" in rendered.markdown
    assert "coverage" not in rendered.markdown
    assert "slurm_semantic_frame" not in rendered.markdown
    assert not rendered.markdown.lstrip().startswith("{")


def test_slurm_debug_mode_includes_compact_debug_metadata() -> None:
    rendered = present_result(_compound_slurm_result(), PresentationContext(mode="debug"))

    assert "## Debug Metadata" in rendered.markdown
    assert "```json" in rendered.markdown


def test_runtime_return_uses_presentation_for_openwebui_style_final_output(tmp_path: Path) -> None:
    history = [
        StepLog(
            step=ExecutionStep(id=1, action="slurm.metrics", output="cluster_summary"),
            result={"metric_group": "cluster_summary", "payload": {"queue_count": 10, "running_jobs": 2, "pending_jobs": 8}},
            success=True,
        ),
        StepLog(
            step=ExecutionStep(id=2, action="runtime.return"),
            result={"value": _compound_slurm_result(), "output": json.dumps(_compound_slurm_result())},
            success=True,
        ),
    ]

    output = summarize_final_output("What is the status of my SLURM cluster?", history, settings=_settings(tmp_path))

    assert output["content"].startswith("## Summary")
    assert "## SLURM Cluster Status" in output["content"]
    assert "\n# " not in output["content"]
    assert "coverage" not in output["content"]
    assert not output["content"].lstrip().startswith("{")


def test_explicit_json_strips_internal_telemetry_in_user_mode(tmp_path: Path) -> None:
    history = [
        StepLog(
            step=ExecutionStep(id=1, action="runtime.return"),
            result={"value": _compound_slurm_result(), "output": json.dumps(_compound_slurm_result())},
            success=True,
        )
    ]

    output = summarize_final_output("Show SLURM cluster status as JSON.", history, settings=_settings(tmp_path))
    payload = json.loads(output["content"])

    assert "coverage" not in payload
    assert "slurm_semantic_frame" not in payload
    assert "results" in payload
