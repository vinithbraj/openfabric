from __future__ import annotations

from aor_runtime.runtime.insights import generate_slurm_insights


def test_high_pending_running_ratio_warns_about_queue_pressure() -> None:
    result = generate_slurm_insights({"domain": "slurm", "queue": {"running_jobs": 2, "pending_jobs": 30}})

    assert any(insight.title == "Queue pressure is high" for insight in result.insights)
    assert "30 pending" in result.summary


def test_drained_problematic_nodes_warn_about_node_health() -> None:
    result = generate_slurm_insights(
        {
            "domain": "slurm",
            "nodes": {"problematic_nodes": 2, "drained_nodes": 2, "affected_partition_rows": 6},
        }
    )

    titles = {insight.title for insight in result.insights}
    assert "Node health needs attention" in titles
    assert "Drained nodes may reduce capacity" in titles


def test_gpu_unavailable_warns() -> None:
    result = generate_slurm_insights({"domain": "slurm", "gpu": {"available": False}})

    assert any(insight.title == "GPU capacity may be unavailable" for insight in result.insights)


def test_gpu_available_with_high_pending_notes_scheduler_constraints() -> None:
    result = generate_slurm_insights({"domain": "slurm", "queue": {"running_jobs": 2, "pending_jobs": 30}, "gpu": {"available": True}})

    assert any("scheduling policy" in insight.message for insight in result.insights)


def test_slurmdbd_not_ok_warns() -> None:
    result = generate_slurm_insights({"domain": "slurm", "accounting": {"slurmdbd_status": "down"}})

    assert any(insight.title == "Accounting health warning" for insight in result.insights)


def test_healthy_cluster_produces_healthy_summary() -> None:
    result = generate_slurm_insights(
        {
            "domain": "slurm",
            "queue": {"running_jobs": 3, "pending_jobs": 0},
            "nodes": {"problematic_nodes": 0, "drained_nodes": 0, "down_nodes": 0},
            "gpu": {"available": True},
            "accounting": {"slurmdbd_status": "ok"},
        }
    )

    assert result.insights == []
    assert "looks healthy" in result.summary
