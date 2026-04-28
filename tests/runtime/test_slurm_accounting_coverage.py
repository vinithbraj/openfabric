from __future__ import annotations

import pytest

from aor_runtime.core.contracts import ExecutionStep, StepLog
from aor_runtime.config import Settings
from aor_runtime.runtime.capabilities.slurm import (
    SlurmAccountingAggregateIntent,
    SlurmPartitionSummaryIntent,
    SlurmQueueIntent,
)
from aor_runtime.runtime.slurm_coverage import validate_slurm_coverage
from aor_runtime.runtime.slurm_semantics import extract_slurm_semantic_frame
from aor_runtime.runtime.validator import RuntimeValidator


def test_runtime_average_cannot_be_covered_by_queue_or_partitions() -> None:
    frame = extract_slurm_semantic_frame("average runtime on totalseg partition")

    assert not validate_slurm_coverage(frame, SlurmQueueIntent(partition="totalseg")).passed
    assert not validate_slurm_coverage(frame, SlurmPartitionSummaryIntent(partition="totalseg")).passed


def test_default_completed_must_be_recorded_for_default_runtime_state() -> None:
    frame = extract_slurm_semantic_frame("average runtime on totalseg partition")
    start = frame.requests[0].filters["start"]

    wrong = SlurmAccountingAggregateIntent(metric="average_elapsed", partition="totalseg", state="COMPLETED", start=start)
    right = wrong.model_copy(update={"default_state_applied": True})

    assert not validate_slurm_coverage(frame, wrong).passed
    assert validate_slurm_coverage(frame, right).passed


def test_negated_completed_filter_fails_if_completed_state_is_applied() -> None:
    frame = extract_slurm_semantic_frame("average runtime on totalseg partition, don't filter by completed")
    start = frame.requests[0].filters["start"]

    wrong = SlurmAccountingAggregateIntent(metric="average_elapsed", partition="totalseg", state="COMPLETED", start=start)
    right = SlurmAccountingAggregateIntent(
        metric="average_elapsed",
        partition="totalseg",
        include_all_states=True,
        state=None,
        start=start,
    )

    assert not validate_slurm_coverage(frame, wrong).passed
    assert validate_slurm_coverage(frame, right).passed


def test_group_by_and_threshold_must_be_covered() -> None:
    grouped = extract_slurm_semantic_frame("average runtime by partition")
    threshold = extract_slurm_semantic_frame("how many jobs longer than 2 hours on totalseg partition")
    grouped_start = grouped.requests[0].filters["start"]
    threshold_start = threshold.requests[0].filters["start"]

    assert not validate_slurm_coverage(grouped, SlurmAccountingAggregateIntent(metric="average_elapsed")).passed
    assert validate_slurm_coverage(
        grouped,
        SlurmAccountingAggregateIntent(
            metric="average_elapsed",
            group_by="partition",
            state="COMPLETED",
            default_state_applied=True,
            start=grouped_start,
        ),
    ).passed
    assert not validate_slurm_coverage(
        threshold,
        SlurmAccountingAggregateIntent(metric="count_longer_than", partition="totalseg", start=threshold_start),
    ).passed
    assert validate_slurm_coverage(
        threshold,
        SlurmAccountingAggregateIntent(
            metric="count_longer_than",
            partition="totalseg",
            threshold_seconds=7200,
            state="COMPLETED",
            default_state_applied=True,
            start=threshold_start,
        ),
    ).passed


def test_live_accounting_aggregate_validator_is_semantic_not_exact(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("AOR_SLURM_FIXTURE_DIR", raising=False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("live validator should not re-run volatile sacct aggregates")

    monkeypatch.setattr("aor_runtime.runtime.validator.slurm_accounting_aggregate", fail_if_called)
    result = {
        "result_kind": "accounting_aggregate",
        "source": "sacct",
        "metric": "average_elapsed",
        "partition": "totalseg",
        "state": None,
        "include_all_states": True,
        "default_state_applied": False,
        "job_count": 2,
        "total_count": 2,
        "returned_count": 2,
        "truncated": False,
        "average_elapsed_seconds": 30.0,
        "min_elapsed_seconds": 10,
        "max_elapsed_seconds": 50,
        "sum_elapsed_seconds": 60,
    }
    log = StepLog(
        step=ExecutionStep(
            id=1,
            action="slurm.accounting_aggregate",
            args={
                "metric": "average_elapsed",
                "partition": "totalseg",
                "state": None,
                "include_all_states": True,
            },
        ),
        result=result,
        success=True,
    )

    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    validation, checks = RuntimeValidator(settings).validate([log])

    assert validation.success is True
    assert checks[0]["success"] is True
