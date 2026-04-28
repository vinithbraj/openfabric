from __future__ import annotations

from pathlib import Path

import pytest

from aor_runtime.config import Settings
from aor_runtime.tools.base import ToolExecutionError
from aor_runtime.tools.gateway import GatewayExecResult, GatewayExecStreamChunk
from aor_runtime.tools.slurm import (
    SLURM_FIXTURE_DIR_ENV,
    SlurmAccountingTool,
    SlurmQueueTool,
    SlurmNodesTool,
    parse_elapsed_to_seconds,
    slurm_accounting,
    slurm_job_detail,
    slurm_metrics,
    slurm_node_detail,
    slurm_nodes,
    slurm_partitions,
    slurm_queue,
    slurm_slurmdbd_health,
)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    payload = {
        "workspace_root": tmp_path,
        "run_store_path": tmp_path / "runtime.db",
        "gateway_url": "https://gateway.internal/exec",
        "available_nodes_raw": "edge-1,edge-2",
        "default_node": "edge-1",
    }
    payload.update(overrides)
    return Settings(**payload)


def _gateway_result(stdout: str = "", stderr: str = "", exit_code: int = 0) -> GatewayExecResult:
    return GatewayExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code)


def test_slurm_queue_uses_gateway_with_validated_command(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, str]] = []

    def fake_execute(settings, *, node: str, command: str):
        calls.append({"node": node, "command": command})
        return _gateway_result("12345|alice|RUNNING|gpu|train|01:00:00|2|None\n")

    monkeypatch.setattr("aor_runtime.tools.slurm.execute_gateway_command", fake_execute)
    result = slurm_queue(_settings(tmp_path), gateway_node="edge-2")

    assert result["count"] == 1
    assert calls == [{"node": "edge-2", "command": "squeue -h -o '%i|%u|%T|%P|%j|%M|%D|%R'"}]


def test_slurm_queue_uses_default_gateway_node(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_execute(settings, *, node: str, command: str):
        calls.append(node)
        return _gateway_result("12345|alice|RUNNING|gpu|train|01:00:00|2|None\n")

    monkeypatch.setattr("aor_runtime.tools.slurm.execute_gateway_command", fake_execute)
    slurm_queue(_settings(tmp_path))
    assert calls == ["edge-1"]


def test_slurm_queue_rejects_unavailable_gateway_node_before_http(monkeypatch, tmp_path: Path) -> None:
    def fake_execute(settings, *, node: str, command: str):
        raise AssertionError("gateway should not be called")

    monkeypatch.setattr("aor_runtime.tools.slurm.execute_gateway_command", fake_execute)
    with pytest.raises(ToolExecutionError, match="Node is not available: edge-9"):
        slurm_queue(_settings(tmp_path), gateway_node="edge-9")


def test_slurm_queue_without_default_or_explicit_gateway_node_matches_shell_error(monkeypatch, tmp_path: Path) -> None:
    def fake_execute(settings, *, node: str, command: str):
        raise AssertionError("gateway should not be called")

    monkeypatch.setattr("aor_runtime.tools.slurm.execute_gateway_command", fake_execute)
    settings = _settings(tmp_path, available_nodes_raw="edge-1,edge-2", default_node=None)
    with pytest.raises(ToolExecutionError, match="No node specified and no default node is configured"):
        slurm_queue(settings)


def test_slurm_queue_filters_user_state_and_partition(monkeypatch, tmp_path: Path) -> None:
    def fake_execute(settings, *, node: str, command: str):
        assert "--user" not in command
        return _gateway_result(
            "12345|alice|RUNNING|gpu|train|01:00:00|2|None\n"
            "12346|alice|PENDING|gpu|prep|00:00:00|1|Priority\n"
            "12347|bob|PENDING|cpu|align|00:00:00|1|Resources\n"
        )

    monkeypatch.setattr("aor_runtime.tools.slurm.execute_gateway_command", fake_execute)
    result = slurm_queue(_settings(tmp_path), user="alice", state="PENDING", partition="gpu")
    assert result == {
        "jobs": [
            {
                "job_id": "12346",
                "user": "alice",
                "state": "PENDING",
                "partition": "gpu",
                "name": "prep",
                "time": "00:00:00",
                "nodes": "1",
                "reason": "Priority",
            }
        ],
        "count": 1,
        "total_count": 1,
        "returned_count": 1,
        "limit": 100,
        "truncated": False,
        "filters": {"user": "alice", "state": "PENDING", "partition": "gpu"},
        "group_by": None,
        "grouped": None,
    }


def test_slurm_nodes_with_mocked_sinfo(monkeypatch, tmp_path: Path) -> None:
    def fake_execute(settings, *, node: str, command: str):
        assert command == "sinfo -h -N -o '%N|%t|%P|%c|%m|%G'"
        return _gateway_result(
            "slurm-worker-agatha|idle|gpu|64|256000|gpu:a100:4\n"
            "slurm-worker-bravo|alloc|gpu|64|256000|gpu:a100:4\n"
        )

    monkeypatch.setattr("aor_runtime.tools.slurm.execute_gateway_command", fake_execute)
    result = slurm_nodes(_settings(tmp_path), gateway_node="edge-2")
    assert result["count"] == 2
    assert result["summary"]["idle"] == 1
    assert result["summary"]["allocated"] == 1


def test_slurm_nodes_filters_problematic_and_gpu_only(monkeypatch, tmp_path: Path) -> None:
    def fake_execute(settings, *, node: str, command: str):
        return _gateway_result(
            "slurm-worker-agatha|idle|gpu|64|256000|gpu:a100:4\n"
            "slurm-worker-delta|down|cpu|32|128000|(null)\n"
            "slurm-worker-echo|drain|cpu|32|128000|gpu:v100:1\n"
        )

    monkeypatch.setattr("aor_runtime.tools.slurm.execute_gateway_command", fake_execute)
    problematic = slurm_nodes(_settings(tmp_path), state_group="problematic")
    gpu_only = slurm_nodes(_settings(tmp_path), gpu_only=True)

    assert [node["name"] for node in problematic["nodes"]] == ["slurm-worker-delta", "slurm-worker-echo"]
    assert [node["name"] for node in gpu_only["nodes"]] == ["slurm-worker-agatha", "slurm-worker-echo"]


def test_slurm_problematic_metrics_and_elapsed_parser(monkeypatch, tmp_path: Path) -> None:
    def fake_execute(settings, *, node: str, command: str):
        if command.startswith("sinfo "):
            return _gateway_result(
                "slurm-worker-agatha|idle|gpu|64|256000|gpu:a100:4\n"
                "slurm-worker-delta|down|cpu|32|128000|(null)\n"
                "slurm-worker-echo|drain|cpu|32|128000|gpu:v100:1\n"
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("aor_runtime.tools.slurm.execute_gateway_command", fake_execute)
    result = slurm_metrics(_settings(tmp_path), metric_group="problematic_nodes")

    assert result["payload"]["count"] == 2
    assert parse_elapsed_to_seconds("1-02:03:04") == 93784


def test_slurm_partitions_with_mocked_sinfo(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "aor_runtime.tools.slurm.execute_gateway_command",
        lambda settings, *, node, command: _gateway_result(
            "gpu|up|7-00:00:00|2|up|0/64/64/128|gpu:a100:8\ncpu|up|7-00:00:00|3|up|16/48/0/64|(null)\n"
        ),
    )
    result = slurm_partitions(_settings(tmp_path))
    assert result["partitions"][0]["partition"] == "gpu"
    assert len(result["partitions"]) == 2


def test_slurm_accounting_with_mocked_sacct(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_execute(settings, *, node: str, command: str):
        calls.append(command)
        return _gateway_result(
            "12340|alice|COMPLETED|gpu|train|01:00:00|8|32G|2026-04-25T08:00:00|2026-04-25T08:10:00|2026-04-25T09:10:00|0:0\n"
            "12341|alice|FAILED|cpu|align|00:20:00|4|8G|2026-04-25T10:00:00|2026-04-25T10:05:00|2026-04-25T10:25:00|1:0\n"
        )

    monkeypatch.setattr("aor_runtime.tools.slurm.execute_gateway_command", fake_execute)
    result = slurm_accounting(_settings(tmp_path), user="alice", state="FAILED", gateway_node="edge-2")
    assert result["count"] == 1
    assert result["jobs"][0]["job_id"] == "12341"
    assert "--user=alice" in calls[0]
    assert "--state=FAILED" in calls[0]


def test_slurm_job_detail_with_mocked_scontrol(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "aor_runtime.tools.slurm.execute_gateway_command",
        lambda settings, *, node, command: _gateway_result(
            "JobId=12345 JobName=train UserId=alice(1000) Partition=gpu JobState=RUNNING NumNodes=2 Reason=None\n"
        ),
    )
    result = slurm_job_detail(_settings(tmp_path), job_id="12345")
    assert result["job_id"] == "12345"
    assert result["fields"]["JobState"] == "RUNNING"
    assert result["field_rows"][0]["field"] == "JobId"


def test_slurm_node_detail_with_mocked_scontrol(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "aor_runtime.tools.slurm.execute_gateway_command",
        lambda settings, *, node, command: _gateway_result("NodeName=slurm-worker-agatha State=IDLE Gres=gpu:a100:4\n"),
    )
    result = slurm_node_detail(_settings(tmp_path), node="slurm-worker-agatha")
    assert result["node"] == "slurm-worker-agatha"
    assert result["fields"]["State"] == "IDLE"


def test_slurm_slurmdbd_health_with_mocked_sacctmgr(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def fake_execute(settings, *, node: str, command: str):
        calls.append((node, command))
        return _gateway_result("clusterA|controlhost\n")

    monkeypatch.setattr("aor_runtime.tools.slurm.execute_gateway_command", fake_execute)
    result = slurm_slurmdbd_health(_settings(tmp_path), gateway_node="edge-2")
    assert calls[0] == ("edge-2", "sacctmgr show cluster -P")
    assert result["available"] is True
    assert result["status"] == "ok"


def test_gateway_transport_errors_are_clear(monkeypatch, tmp_path: Path) -> None:
    def missing_binary(settings, *, node: str, command: str):
        raise ToolExecutionError("SLURM binary unavailable: squeue")

    monkeypatch.setattr("aor_runtime.tools.slurm.execute_gateway_command", missing_binary)
    with pytest.raises(ToolExecutionError, match="SLURM binary unavailable: squeue"):
        slurm_queue(_settings(tmp_path))


def test_invalid_job_id_node_and_partition_inputs_rejected(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(ToolExecutionError, match="job_id"):
        slurm_job_detail(settings, job_id="12345; rm -rf /")
    with pytest.raises(ToolExecutionError, match="node names"):
        slurm_node_detail(settings, node="worker-1;rm")
    with pytest.raises(ToolExecutionError, match="unsafe characters"):
        slurm_queue(settings, partition="gpu;rm")


def test_slurm_tools_only_use_read_only_commands(monkeypatch, tmp_path: Path) -> None:
    seen_commands: list[str] = []

    def fake_execute(settings, *, node: str, command: str):
        seen_commands.append(command)
        if command.startswith("squeue "):
            return _gateway_result("12345|alice|RUNNING|gpu|train|01:00:00|2|None\n")
        if command.startswith("sinfo "):
            return _gateway_result("gpu|up|7-00:00:00|2|up|0/64/64/128|gpu:a100:8\n")
        if command.startswith("scontrol show job"):
            return _gateway_result("JobId=12345 JobName=train Partition=gpu JobState=RUNNING\n")
        if command.startswith("sacct "):
            return _gateway_result("12340|alice|COMPLETED|gpu|train|01:00:00|8|32G|2026-04-25T08:00:00|2026-04-25T08:10:00|2026-04-25T09:10:00|0:0\n")
        if command.startswith("sacctmgr show cluster -P"):
            return _gateway_result("clusterA|controlhost\n")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("aor_runtime.tools.slurm.execute_gateway_command", fake_execute)
    settings = _settings(tmp_path)
    slurm_queue(settings)
    slurm_partitions(settings)
    slurm_job_detail(settings, job_id="12345")
    slurm_accounting(settings)
    slurm_slurmdbd_health(settings)
    assert all(command.split()[0] in {"squeue", "sinfo", "scontrol", "sacct", "sacctmgr"} for command in seen_commands)
    assert not any(token in " ".join(seen_commands) for token in ("scancel", "sbatch", "scontrol update"))


def test_fixture_mode_bypasses_gateway(monkeypatch, tmp_path: Path) -> None:
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    (fixture_dir / "squeue.txt").write_text("12345|alice|RUNNING|gpu|train|01:00:00|2|None\n")

    def fake_execute(settings, *, node: str, command: str):
        raise AssertionError("gateway should not be called in fixture mode")

    monkeypatch.setenv(SLURM_FIXTURE_DIR_ENV, str(fixture_dir))
    monkeypatch.setattr("aor_runtime.tools.slurm.execute_gateway_command", fake_execute)
    result = slurm_queue(_settings(tmp_path), gateway_node="edge-2")
    assert result["count"] == 1


def test_streaming_slurm_queue_uses_gateway_chunks(monkeypatch, tmp_path: Path) -> None:
    def fake_stream(settings, *, node: str, command: str):
        assert node == "edge-2"
        assert command == "squeue -h -o '%i|%u|%T|%P|%j|%M|%D|%R'"
        yield GatewayExecStreamChunk(type="stdout", text="12345|alice|RUNNING|gpu|train|01:00:00|2|None\n")
        yield GatewayExecStreamChunk(type="completed", exit_code=0)

    monkeypatch.setattr("aor_runtime.tools.slurm.stream_gateway_command", fake_stream)
    tool = SlurmQueueTool(_settings(tmp_path))
    chunks = list(tool.stream(tool.args_model.model_validate({"gateway_node": "edge-2"})))
    assert chunks[0]["type"] == "stdout"
    assert chunks[-1]["type"] == "completed"


def test_streaming_accounting_builds_structured_result(monkeypatch, tmp_path: Path) -> None:
    tool = SlurmAccountingTool(_settings(tmp_path))
    args = tool.args_model.model_validate({"user": "alice", "state": "FAILED", "gateway_node": "edge-2"})
    result = tool.build_stream_result(
        args,
        stdout="12341|alice|FAILED|cpu|align|00:20:00|4|8G|2026-04-25T10:00:00|2026-04-25T10:05:00|2026-04-25T10:25:00|1:0\n",
        stderr="",
        returncode=0,
        metadata={},
    )
    assert result["count"] == 1
    assert result["jobs"][0]["job_id"] == "12341"
