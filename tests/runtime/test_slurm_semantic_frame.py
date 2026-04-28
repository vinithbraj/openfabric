from __future__ import annotations

from aor_runtime.runtime.slurm_semantics import extract_slurm_semantic_frame


def _kinds(prompt: str) -> list[str]:
    return [request.kind for request in extract_slurm_semantic_frame(prompt).requests]


def test_cluster_health_extracts_cluster_request() -> None:
    frame = extract_slurm_semantic_frame("What is the status of my SLURM cluster?")

    assert frame.query_type == "compound"
    assert "cluster_health" in _kinds("What is the status of my SLURM cluster?")
    assert not any(constraint.kind == "job_user" for constraint in frame.constraints)


def test_running_and_pending_extracts_two_job_count_requests() -> None:
    frame = extract_slurm_semantic_frame("How many jobs are running and pending?")

    requests = frame.requests
    assert [request.kind for request in requests] == ["job_count", "job_count"]
    assert [request.filters["state"] for request in requests] == ["RUNNING", "PENDING"]


def test_problematic_nodes_extracts_request() -> None:
    frame = extract_slurm_semantic_frame("Are there any problematic nodes?")

    assert frame.requests[0].kind == "problematic_nodes"
    assert any(constraint.kind == "node_state" and constraint.value == "problematic" for constraint in frame.constraints)


def test_failed_jobs_yesterday_extracts_accounting_time_window() -> None:
    frame = extract_slurm_semantic_frame("Show failed jobs yesterday.")

    assert frame.requests[0].kind == "accounting_jobs"
    assert frame.requests[0].filters["state"] == "FAILED"
    assert frame.requests[0].filters["start"]
    assert frame.requests[0].filters["end"]


def test_gpu_queue_node_compound_extracts_all_requests() -> None:
    frame = extract_slurm_semantic_frame("Summarize queue, node, and GPU status.")

    assert {request.kind for request in frame.requests} == {"queue_status", "node_status", "gpu_availability"}


def test_mutation_extracts_unsupported_request() -> None:
    frame = extract_slurm_semantic_frame("Drain node slurm-worker-agatha.")

    assert frame.query_type == "unsupported_mutation"
    assert frame.requests[0].kind == "unsupported_mutation"
