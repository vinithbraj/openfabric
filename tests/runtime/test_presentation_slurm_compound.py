from __future__ import annotations

from aor_runtime.runtime.presentation import PresentationContext, present_result


def test_compound_running_pending_problematic_renders_all_sections() -> None:
    result = {
        "summary": {"request_count": 3, "tool_count": 3},
        "results": {
            "running_jobs": {
                "jobs": [
                    {"job_id": "1", "user": "alice", "state": "RUNNING", "partition": "hpc", "name": "train", "time": "1:00", "nodes": "1", "reason": "node-a"}
                ],
                "count": 1,
                "total_count": 1,
                "returned_count": 1,
                "limit": None,
                "truncated": False,
                "filters": {"state": "RUNNING"},
            },
            "pending_jobs": {
                "jobs": [],
                "count": 0,
                "total_count": 0,
                "returned_count": 0,
                "limit": None,
                "truncated": False,
                "filters": {"state": "PENDING"},
            },
            "problematic_nodes": {
                "nodes": [{"name": "node-a", "state": "drain", "partition": "hpc", "cpus": "16", "memory": "1", "gres": "gpu:1"}],
                "count": 1,
                "partition_row_count": 1,
                "unique_count": 1,
                "summary": {"idle": 0, "allocated": 0, "mixed": 0, "down": 0, "drained": 1, "other": 0},
                "filters": {"state_group": "problematic"},
            },
        },
        "coverage": {"covered_requests": ["r1", "r2", "r3"]},
        "slurm_semantic_frame": {"internal": True},
    }

    rendered = present_result(result, PresentationContext(mode="user"))

    assert rendered.markdown.startswith("## SLURM Summary")
    assert "\n# " not in rendered.markdown
    assert "## Running SLURM Jobs" in rendered.markdown
    assert "## Pending SLURM Jobs" in rendered.markdown
    assert "No matching SLURM jobs were found." in rendered.markdown
    assert "## Problematic Nodes" in rendered.markdown
    assert "coverage" not in rendered.markdown
    assert "slurm_semantic_frame" not in rendered.markdown
    assert not rendered.markdown.lstrip().startswith("{")
