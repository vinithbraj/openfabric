from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.runtime.capabilities.base import ClassificationContext, CompileContext
from aor_runtime.runtime.capabilities.slurm import (
    SlurmAccountingAggregateIntent,
    SlurmCapabilityPack,
    SlurmPartitionSummaryIntent,
    SlurmQueueIntent,
)
from aor_runtime.runtime.presentation import PresentationContext, present_slurm_result
from aor_runtime.runtime.slurm_aggregations import aggregate_slurm_accounting_jobs
from aor_runtime.runtime.slurm_coverage import validate_slurm_coverage
from aor_runtime.runtime.slurm_semantics import extract_slurm_semantic_frame
from aor_runtime.tools.gateway import GatewayExecResult
from aor_runtime.tools.slurm import parse_elapsed_to_seconds, slurm_accounting_aggregate


SLURM_ALLOWED_TOOLS = [
    "slurm.queue",
    "slurm.job_detail",
    "slurm.nodes",
    "slurm.node_detail",
    "slurm.partitions",
    "slurm.accounting",
    "slurm.accounting_aggregate",
    "slurm.metrics",
    "slurm.slurmdbd_health",
]


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        gateway_url="https://gateway.internal/exec",
        available_nodes_raw="edge-1",
        default_node="edge-1",
    )


def _context(tmp_path: Path) -> ClassificationContext:
    return ClassificationContext(schema_payload=None, allowed_tools=SLURM_ALLOWED_TOOLS, settings=_settings(tmp_path))


def _gateway_result(stdout: str = "") -> GatewayExecResult:
    return GatewayExecResult(stdout=stdout, stderr="", exit_code=0)


def test_parse_elapsed_formats() -> None:
    assert parse_elapsed_to_seconds("02:03") == 123
    assert parse_elapsed_to_seconds("01:02:03") == 3723
    assert parse_elapsed_to_seconds("1-02:03:04") == 93784
    assert parse_elapsed_to_seconds("00:00:00") == 0
    assert parse_elapsed_to_seconds("Unknown") is None


def test_aggregate_average_min_max_sum_and_threshold() -> None:
    jobs = [
        {"partition": "totalseg", "user": "alice", "state": "COMPLETED", "elapsed_seconds": 60},
        {"partition": "totalseg", "user": "alice", "state": "COMPLETED", "elapsed_seconds": 180},
        {"partition": "hpc", "user": "bob", "state": "COMPLETED", "elapsed_seconds": 360},
        {"partition": "hpc", "user": "bob", "state": "COMPLETED", "elapsed": "bad"},
    ]
    intent = SlurmAccountingAggregateIntent(metric="runtime_summary")

    result = aggregate_slurm_accounting_jobs(jobs, intent)

    assert result["job_count"] == 3
    assert result["average_elapsed_seconds"] == 200
    assert result["min_elapsed_seconds"] == 60
    assert result["max_elapsed_seconds"] == 360
    assert result["sum_elapsed_seconds"] == 600
    assert "Ignored 1 accounting rows" in result["warnings"][0]

    threshold = SlurmAccountingAggregateIntent(metric="count_longer_than", threshold_seconds=120)
    counted = aggregate_slurm_accounting_jobs(jobs, threshold)
    assert counted["job_count"] == 2
    assert counted["count_longer_than"] == 2


def test_aggregate_group_by_partition() -> None:
    jobs = [
        {"partition": "totalseg", "elapsed_seconds": 60},
        {"partition": "totalseg", "elapsed_seconds": 180},
        {"partition": "hpc", "elapsed_seconds": 360},
    ]
    intent = SlurmAccountingAggregateIntent(metric="average_elapsed", group_by="partition")

    result = aggregate_slurm_accounting_jobs(jobs, intent)

    groups = {group["key"]: group for group in result["groups"]}
    assert groups["totalseg"]["job_count"] == 2
    assert groups["totalseg"]["average_elapsed_seconds"] == 120
    assert groups["hpc"]["average_elapsed_seconds"] == 360


def test_slurm_accounting_aggregate_tool_uses_sacct(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_execute(settings, *, node: str, command: str):
        calls.append(command)
        return _gateway_result(
            "1|alice|COMPLETED|totalseg|a|00:10:00|1|1G|2026-04-26T00:00:00|2026-04-26T00:00:00|2026-04-26T00:10:00|0:0\n"
            "2|alice|COMPLETED|totalseg|b|00:20:00|1|1G|2026-04-26T00:00:00|2026-04-26T00:00:00|2026-04-26T00:20:00|0:0\n"
        )

    monkeypatch.setattr("aor_runtime.tools.slurm.execute_gateway_command", fake_execute)
    result = slurm_accounting_aggregate(
        _settings(tmp_path),
        partition="totalseg",
        state="COMPLETED",
        metric="average_elapsed",
        start="2026-04-21 00:00:00",
    )

    assert result["job_count"] == 2
    assert result["average_elapsed_seconds"] == 900
    assert "--partition=totalseg" in calls[0]
    assert "--state=COMPLETED" in calls[0]
    assert calls[0].startswith("sacct ")


def test_runtime_prompt_routes_to_accounting_aggregate(tmp_path: Path) -> None:
    result = SlurmCapabilityPack().classify(
        "how long on average did jobs take to run on totalseg partition?",
        _context(tmp_path),
    )

    assert result.matched is True
    assert result.intent.__class__.__name__ == "SlurmAccountingAggregateIntent"
    assert result.intent.metric == "average_elapsed"
    assert result.intent.partition == "totalseg"
    assert result.intent.state == "COMPLETED"
    assert result.intent.default_state_applied is True
    assert result.intent.include_all_states is False
    assert result.intent.start
    assert result.intent.time_window_label == "Last 7 days"

    plan = SlurmCapabilityPack().compile(result.intent, CompileContext(allowed_tools=SLURM_ALLOWED_TOOLS, settings=_settings(tmp_path)))
    assert plan is not None
    assert [step.action for step in plan.plan.steps] == ["slurm.accounting_aggregate", "runtime.return"]
    assert plan.plan.steps[0].args["state"] == "COMPLETED"
    assert plan.plan.steps[0].args["default_state_applied"] is True


def test_completed_runtime_prompt_keeps_completed_filter(tmp_path: Path) -> None:
    result = SlurmCapabilityPack().classify(
        "average runtime of completed jobs on totalseg partition",
        _context(tmp_path),
    )

    assert result.intent.__class__.__name__ == "SlurmAccountingAggregateIntent"
    assert result.intent.state == "COMPLETED"


def test_all_jobs_runtime_prompt_does_not_keep_negated_completed_filter(tmp_path: Path) -> None:
    prompt = "how long on average did the job take to run on totalseg partition, don't filter by completed, get all jobs and then filter ?"

    frame = extract_slurm_semantic_frame(prompt)
    result = SlurmCapabilityPack().classify(prompt, _context(tmp_path))

    assert not [constraint for constraint in frame.constraints if constraint.kind == "job_state"]
    assert frame.source == "sacct"
    assert frame.include_all_states is True
    assert frame.default_state_applied is False
    assert [constraint.value for constraint in frame.negated_filters] == ["COMPLETED"]
    assert result.intent.__class__.__name__ == "SlurmAccountingAggregateIntent"
    assert result.intent.state is None
    assert result.intent.include_all_states is True
    assert result.intent.default_state_applied is False


def test_include_all_states_does_not_pass_state_to_sacct(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_execute(settings, *, node: str, command: str):
        calls.append(command)
        return _gateway_result(
            "1|alice|FAILED|totalseg|a|00:10:00|1|1G|2026-04-26T00:00:00|2026-04-26T00:00:00|2026-04-26T00:10:00|1:0\n"
            "2|alice|COMPLETED|totalseg|b|00:20:00|1|1G|2026-04-26T00:00:00|2026-04-26T00:00:00|2026-04-26T00:20:00|0:0\n"
        )

    monkeypatch.setattr("aor_runtime.tools.slurm.execute_gateway_command", fake_execute)
    result = slurm_accounting_aggregate(
        _settings(tmp_path),
        partition="totalseg",
        state="COMPLETED",
        include_all_states=True,
        metric="average_elapsed",
        start="2026-04-21 00:00:00",
    )

    assert result["job_count"] == 2
    assert result["include_all_states"] is True
    assert result["state"] is None
    assert "--partition=totalseg" in calls[0]
    assert "--state=COMPLETED" not in calls[0]


def test_runtime_prompt_coverage_rejects_queue_and_partitions() -> None:
    frame = extract_slurm_semantic_frame("average runtime of completed jobs in slicer last 7 days")

    assert frame.requests[0].kind == "accounting_aggregate"
    assert validate_slurm_coverage(frame, SlurmQueueIntent(partition="slicer")).passed is False
    assert validate_slurm_coverage(frame, SlurmPartitionSummaryIntent(partition="slicer")).passed is False
    assert validate_slurm_coverage(
        frame,
        SlurmAccountingAggregateIntent(
            metric="average_elapsed",
            partition="slicer",
            state="COMPLETED",
            default_state_applied=False,
            start=frame.requests[0].filters["start"],
            time_window_label="Last 7 days",
        ),
    ).passed


def test_jobs_longer_than_uses_threshold_accounting_aggregate(tmp_path: Path) -> None:
    result = SlurmCapabilityPack().classify("how many completed jobs ran longer than 2 hours on totalseg?", _context(tmp_path))

    assert result.intent.__class__.__name__ == "SlurmAccountingAggregateIntent"
    assert result.intent.metric == "count_longer_than"
    assert result.intent.threshold_seconds == 7200
    assert result.intent.partition == "totalseg"


def test_accounting_aggregate_presentation() -> None:
    rendered = present_slurm_result(
        {
            "result_kind": "accounting_aggregate",
            "metric": "average_elapsed",
            "partition": "totalseg",
            "state": "COMPLETED",
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
    assert "Average runtime on `totalseg`" in rendered.markdown
    assert "15m" in rendered.markdown
    assert "Queue" not in rendered.markdown


def test_median_runtime_fails_safely(tmp_path: Path) -> None:
    result = SlurmCapabilityPack().classify("median runtime of totalseg jobs", _context(tmp_path))

    assert result.matched is True
    assert result.intent.__class__.__name__ == "SlurmFailureIntent"
    assert "Median runtime is not currently supported" in result.intent.message
