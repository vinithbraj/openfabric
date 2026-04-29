from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionStep, StepLog
from aor_runtime.runtime.semantic_obligations import maybe_apply_semantic_fallback
from aor_runtime.tools.gateway import GatewayExecResult


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        gateway_url="https://gateway.internal/exec",
        available_nodes_raw="edge-1",
        default_node="edge-1",
    )


def test_slurm_accounting_aggregate_falls_back_to_broad_fetch_and_local_completed_filter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("AOR_SLURM_FIXTURE_DIR", raising=False)
    calls: list[str] = []

    def fake_execute(settings, *, node: str, command: str):
        calls.append(command)
        return GatewayExecResult(
            stdout=(
                "1|alice|FAILED|totalseg|failed_job|00:10:00|1|1G|"
                "2026-04-26T00:00:00|2026-04-26T00:00:00|2026-04-26T00:10:00|1:0\n"
                "2|alice|COMPLETED|totalseg|done_job|00:20:00|1|1G|"
                "2026-04-26T00:00:00|2026-04-26T00:00:00|2026-04-26T00:20:00|0:0\n"
            ),
            stderr="",
            exit_code=0,
        )

    monkeypatch.setattr("aor_runtime.tools.slurm.execute_gateway_command", fake_execute)
    log = StepLog(
        step=ExecutionStep(
            id=1,
            action="slurm.accounting_aggregate",
            args={
                "partition": "totalseg",
                "state": "COMPLETED",
                "metric": "average_elapsed",
                "start": "2026-04-26 00:00:00",
                "limit": 1000,
            },
            output="avg_time",
        ),
        result={
            "result_kind": "accounting_aggregate",
            "metric": "average_elapsed",
            "partition": "totalseg",
            "state": "COMPLETED",
            "include_all_states": False,
            "job_count": 0,
            "average_elapsed_seconds": None,
            "min_elapsed_seconds": None,
            "max_elapsed_seconds": None,
            "sum_elapsed_seconds": 0,
            "total_count": 0,
            "returned_count": 0,
            "truncated": False,
        },
        success=True,
    )

    result = maybe_apply_semantic_fallback(
        _settings(tmp_path),
        goal="average time for completed jobs in the totalseg partition",
        log=log,
    )

    assert result.applied is True
    assert "--partition=totalseg" in calls[0]
    assert "--state=COMPLETED" not in calls[0]
    assert result.log.result["state"] == "COMPLETED"
    assert result.log.result["include_all_states"] is False
    assert result.log.result["job_count"] == 1
    assert result.log.result["average_elapsed_seconds"] == 1200
    assert result.log.result["semantic_fallback"]["strategy"] == "broad_fetch_local_filter"


def test_slurm_accounting_aggregate_does_not_fallback_for_all_states_goal(tmp_path: Path) -> None:
    log = StepLog(
        step=ExecutionStep(
            id=1,
            action="slurm.accounting_aggregate",
            args={"partition": "totalseg", "include_all_states": True, "state": None},
            output="avg_time",
        ),
        result={
            "result_kind": "accounting_aggregate",
            "metric": "average_elapsed",
            "partition": "totalseg",
            "state": None,
            "include_all_states": True,
            "job_count": 2,
        },
        success=True,
    )

    result = maybe_apply_semantic_fallback(
        _settings(tmp_path),
        goal="average runtime of all jobs on the totalseg partition",
        log=log,
    )

    assert result.applied is False
