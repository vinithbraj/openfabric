from __future__ import annotations

from aor_runtime.runtime.capabilities.slurm import (
    SlurmCompoundIntent,
    SlurmJobCountIntent,
    SlurmMetricsIntent,
    SlurmNodeStatusIntent,
    SlurmUnsupportedMutationIntent,
)
from aor_runtime.runtime.slurm_coverage import validate_slurm_coverage
from aor_runtime.runtime.slurm_semantics import SlurmRequest, SlurmSemanticConstraint, SlurmSemanticFrame


def test_missing_pending_jobs_request_fails() -> None:
    frame = SlurmSemanticFrame(
        query_type="compound",
        requests=[
            SlurmRequest(id="req_1", kind="job_count", raw_text="running jobs", filters={"state": "RUNNING"}, output="count"),
            SlurmRequest(id="req_2", kind="job_count", raw_text="pending jobs", filters={"state": "PENDING"}, output="count"),
        ],
    )
    intent = SlurmJobCountIntent(state="RUNNING")

    result = validate_slurm_coverage(frame, intent)

    assert result.passed is False
    assert [request.id for request in result.missing_requests] == ["req_2"]


def test_problematic_nodes_requires_problematic_state_group() -> None:
    frame = SlurmSemanticFrame(
        query_type="nodes",
        requests=[SlurmRequest(id="req_1", kind="problematic_nodes", raw_text="problematic nodes")],
    )

    assert not validate_slurm_coverage(frame, SlurmNodeStatusIntent(state="down")).passed
    assert validate_slurm_coverage(frame, SlurmNodeStatusIntent(state_group="problematic")).passed


def test_failed_jobs_yesterday_requires_time_window() -> None:
    frame = SlurmSemanticFrame(
        query_type="accounting",
        requests=[
            SlurmRequest(
                id="req_1",
                kind="accounting_jobs",
                raw_text="failed jobs yesterday",
                filters={"state": "FAILED", "start": "2026-04-26 00:00:00", "end": "2026-04-27 00:00:00"},
            )
        ],
        constraints=[
            SlurmSemanticConstraint(
                id="constraint_1",
                kind="time_window",
                raw_text="yesterday",
                value={"start": "2026-04-26 00:00:00", "end": "2026-04-27 00:00:00"},
            )
        ],
    )

    result = validate_slurm_coverage(frame, SlurmJobCountIntent(source="sacct", state="FAILED"))

    assert result.passed is False
    assert result.missing_constraints[0].kind == "time_window"


def test_gpu_availability_requires_gpu_metric() -> None:
    frame = SlurmSemanticFrame(
        query_type="metrics",
        requests=[SlurmRequest(id="req_1", kind="gpu_availability", raw_text="gpu availability")],
    )

    assert not validate_slurm_coverage(frame, SlurmMetricsIntent(metric_group="node_summary")).passed
    assert validate_slurm_coverage(frame, SlurmMetricsIntent(metric_group="gpu_summary")).passed


def test_compound_children_cover_requests() -> None:
    frame = SlurmSemanticFrame(
        query_type="compound",
        requests=[
            SlurmRequest(id="req_1", kind="job_count", raw_text="running jobs", filters={"state": "RUNNING"}),
            SlurmRequest(id="req_2", kind="problematic_nodes", raw_text="problematic nodes"),
        ],
    )
    intent = SlurmCompoundIntent(
        intents=[SlurmJobCountIntent(state="RUNNING"), SlurmNodeStatusIntent(state_group="problematic")]
    )

    assert validate_slurm_coverage(frame, intent).passed


def test_mutation_passes_only_as_refusal() -> None:
    frame = SlurmSemanticFrame(
        query_type="unsupported_mutation",
        requests=[SlurmRequest(id="req_1", kind="unsupported_mutation", raw_text="cancel job")],
    )

    assert not validate_slurm_coverage(frame, SlurmJobCountIntent()).passed
    assert validate_slurm_coverage(frame, SlurmUnsupportedMutationIntent(operation="cancel", reason="read-only")).passed
