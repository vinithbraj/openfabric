from __future__ import annotations

from aor_runtime.runtime.presentation import PresentationContext, present_result


def test_running_job_rows_render_as_markdown_table() -> None:
    result = {
        "jobs": [
            {
                "job_id": "1",
                "user": "alice",
                "state": "RUNNING",
                "partition": "hpc",
                "name": "train",
                "time": "01:00:00",
                "nodes": "1",
                "reason": "node-a",
            }
        ],
        "count": 1,
        "total_count": 1,
        "returned_count": 1,
        "limit": 100,
        "truncated": False,
        "filters": {"state": "RUNNING"},
        "group_by": None,
        "grouped": None,
    }

    rendered = present_result(result, PresentationContext(mode="user", source_action="slurm.queue"))

    assert rendered.markdown.startswith("## Running SLURM Jobs")
    assert "\n# " not in rendered.markdown
    assert "Found **1** running jobs." in rendered.markdown
    assert "| Job ID | User | State | Partition | Name | Time | Nodes | Reason |" in rendered.markdown
    assert "| 1 | alice | RUNNING | hpc | train | 01:00:00 | 1 | node-a |" in rendered.markdown
    assert "Commands Used" not in rendered.markdown


def test_pending_job_truncation_is_explained() -> None:
    jobs = [
        {
            "job_id": str(index),
            "user": "airflow",
            "state": "PENDING",
            "partition": "totalseg",
            "name": f"job_{index}",
            "time": "0:00",
            "nodes": "1",
            "reason": "(Priority)",
        }
        for index in range(100)
    ]
    result = {
        "jobs": jobs,
        "count": 140,
        "total_count": 140,
        "returned_count": 100,
        "limit": 100,
        "truncated": True,
        "filters": {"state": "PENDING"},
        "group_by": None,
        "grouped": None,
    }

    rendered = present_result(result, PresentationContext(mode="user", source_action="slurm.queue", max_rows=5))

    assert rendered.markdown.startswith("## Pending SLURM Jobs")
    assert "Showing **100** of **140** pending jobs." in rendered.markdown
    assert "Showing first 5 rendered rows." in rendered.markdown


def test_zero_queue_rows_are_readable() -> None:
    result = {
        "jobs": [],
        "count": 0,
        "total_count": 0,
        "returned_count": 0,
        "limit": 100,
        "truncated": False,
        "filters": {"state": "RUNNING"},
        "group_by": None,
        "grouped": None,
    }

    rendered = present_result(result, PresentationContext(mode="user", source_action="slurm.queue"))

    assert "No matching SLURM jobs were found." in rendered.markdown


def test_zero_grouped_queue_counts_are_readable() -> None:
    result = {
        "jobs": [],
        "count": 0,
        "total_count": 0,
        "returned_count": 0,
        "limit": 100,
        "truncated": False,
        "filters": {},
        "group_by": "partition",
        "grouped": {},
    }

    rendered = present_result(result, PresentationContext(mode="user", source_action="slurm.queue"))

    assert rendered.markdown.startswith("## Jobs by Partition")
    assert "No matching SLURM jobs were found." in rendered.markdown


def test_zero_duration_accounting_rows_are_readable() -> None:
    result = {
        "jobs": [],
        "count": 0,
        "total_count": 0,
        "returned_count": 0,
        "limit": 100,
        "truncated": False,
        "filters": {"min_elapsed_seconds": 7200},
        "group_by": None,
        "grouped": None,
    }

    rendered = present_result(result, PresentationContext(mode="user", source_action="slurm.accounting"))

    assert rendered.markdown.startswith("## SLURM Jobs Longer Than Requested Duration")
    assert "No jobs longer than the requested duration were found" in rendered.markdown
