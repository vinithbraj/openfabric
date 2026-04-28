from __future__ import annotations

from aor_runtime.runtime.presentation import PresentationContext, present_slurm_result


def test_accounting_aggregate_presents_all_states_note() -> None:
    rendered = present_slurm_result(
        {
            "result_kind": "accounting_aggregate",
            "source": "sacct",
            "metric": "average_elapsed",
            "partition": "totalseg",
            "state": None,
            "include_all_states": True,
            "default_state_applied": False,
            "time_window_label": "Last 7 days",
            "job_count": 2,
            "average_elapsed_human": "15m",
            "min_elapsed_human": "10m",
            "max_elapsed_human": "20m",
            "sum_elapsed_human": "30m",
        },
        PresentationContext(mode="user", source_action="slurm.accounting_aggregate"),
    )

    assert "SLURM Job Runtime" in rendered.markdown
    assert "State filter | All states" in rendered.markdown
    assert "Included all job states" in rendered.markdown
    assert "Queue" not in rendered.markdown


def test_accounting_aggregate_presents_default_completed_note() -> None:
    rendered = present_slurm_result(
        {
            "result_kind": "accounting_aggregate",
            "source": "sacct",
            "metric": "average_elapsed",
            "partition": "totalseg",
            "state": "COMPLETED",
            "include_all_states": False,
            "default_state_applied": True,
            "time_window_label": "Last 7 days",
            "job_count": 2,
            "average_elapsed_human": "15m",
            "min_elapsed_human": "10m",
            "max_elapsed_human": "20m",
            "sum_elapsed_human": "30m",
        },
        PresentationContext(mode="user", source_action="slurm.accounting_aggregate"),
    )

    assert "State filter | Completed jobs (default)" in rendered.markdown
    assert "Defaulted to completed jobs" in rendered.markdown
