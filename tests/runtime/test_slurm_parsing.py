from __future__ import annotations

from aor_runtime.tools.slurm import (
    parse_sacct_output,
    parse_scontrol_kv_output,
    parse_sinfo_nodes_output,
    parse_sinfo_partitions_output,
    parse_squeue_output,
    summarize_gpu_gres,
    summarize_jobs_by_state,
    summarize_node_states,
)


def test_parse_squeue_output() -> None:
    jobs = parse_squeue_output("12345|alice|RUNNING|gpu|train|01:00:00|2|None\n12346|bob|PENDING|cpu|align|00:00:00|1|Resources\n")
    assert jobs == [
        {
            "job_id": "12345",
            "user": "alice",
            "state": "RUNNING",
            "partition": "gpu",
            "name": "train",
            "time": "01:00:00",
            "nodes": "2",
            "reason": "None",
        },
        {
            "job_id": "12346",
            "user": "bob",
            "state": "PENDING",
            "partition": "cpu",
            "name": "align",
            "time": "00:00:00",
            "nodes": "1",
            "reason": "Resources",
        },
    ]


def test_parse_sinfo_nodes_output_and_summarize_states() -> None:
    nodes = parse_sinfo_nodes_output(
        "slurm-worker-agatha|idle|gpu|64|256000|gpu:a100:4\n"
        "slurm-worker-bravo|alloc|gpu|64|256000|gpu:a100:4\n"
        "slurm-worker-charlie|mix|cpu|32|128000|(null)\n"
        "slurm-worker-delta|down|cpu|32|128000|(null)\n"
        "slurm-worker-echo|drain|cpu|32|128000|gpu:v100:1\n"
    )
    assert nodes[0]["name"] == "slurm-worker-agatha"
    assert summarize_node_states(nodes) == {
        "idle": 1,
        "allocated": 1,
        "mixed": 1,
        "down": 1,
        "drained": 1,
        "other": 0,
    }


def test_parse_sinfo_partitions_output() -> None:
    partitions = parse_sinfo_partitions_output("gpu|up|7-00:00:00|2|up|0/64/64/128|gpu:a100:8\ncpu*|up|7-00:00:00|3|up|16/48/0/64|(null)\n")
    assert partitions == [
        {
            "partition": "gpu",
            "availability": "up",
            "time_limit": "7-00:00:00",
            "nodes": "2",
            "state": "up",
            "cpus": "0/64/64/128",
            "gres": "gpu:a100:8",
        },
        {
            "partition": "cpu",
            "availability": "up",
            "time_limit": "7-00:00:00",
            "nodes": "3",
            "state": "up",
            "cpus": "16/48/0/64",
            "gres": "(null)",
        },
    ]


def test_parse_sacct_output() -> None:
    jobs = parse_sacct_output("12340|alice|COMPLETED|gpu|train|01:00:00|8|32G|2026-04-25T08:00:00|2026-04-25T08:10:00|2026-04-25T09:10:00|0:0\n")
    assert jobs == [
        {
            "job_id": "12340",
            "user": "alice",
            "state": "COMPLETED",
            "partition": "gpu",
            "name": "train",
            "elapsed": "01:00:00",
            "alloc_cpus": "8",
            "req_mem": "32G",
            "submit": "2026-04-25T08:00:00",
            "start": "2026-04-25T08:10:00",
            "end": "2026-04-25T09:10:00",
            "exit_code": "0:0",
        }
    ]


def test_parse_scontrol_kv_output() -> None:
    fields = parse_scontrol_kv_output("JobId=12345 JobName=train UserId=alice(1000) Partition=gpu JobState=RUNNING NumNodes=2 Reason=None")
    assert fields["JobId"] == "12345"
    assert fields["JobName"] == "train"
    assert fields["Partition"] == "gpu"
    assert fields["JobState"] == "RUNNING"


def test_summarize_jobs_by_state() -> None:
    summary = summarize_jobs_by_state(
        [
            {"state": "RUNNING"},
            {"state": "PENDING"},
            {"state": "PENDING"},
            {"state": "FAILED"},
        ]
    )
    assert summary == {"FAILED": 1, "PENDING": 2, "RUNNING": 1}


def test_summarize_gpu_gres() -> None:
    summary = summarize_gpu_gres(
        [
            {"gres": "gpu:a100:4"},
            {"gres": "gpu:a100:4"},
            {"gres": "(null)"},
            {"gres": "gpu:v100:1"},
        ]
    )
    assert summary == {
        "available": True,
        "nodes_with_gpu": 3,
        "total_gpus": 9,
        "by_gres": {"gpu:a100:4": 2, "gpu:v100:1": 1},
    }
