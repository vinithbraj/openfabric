from __future__ import annotations

from aor_runtime.tools.slurm import summarize_gpu_gres


def test_gpu_summary_counts_only_gpu_gres_tokens() -> None:
    result = summarize_gpu_gres(
        [
            {"gres": "gpu:4"},
            {"gres": "gpu:2"},
            {"gres": "gpu:a100:4"},
            {"gres": "gpu_mem:196560"},
            {"gres": "shard:8"},
            {"gres": "slicer:1"},
        ]
    )

    assert result["total_gpus"] == 10
    assert result["nodes_with_gpu"] == 3
    assert result["gpu_memory_gres"] == {"gpu_mem:196560": 1}


def test_gpu_summary_uses_unknown_when_gpu_count_is_not_exact() -> None:
    result = summarize_gpu_gres([{"gres": "gpu"}])

    assert result["available"] is True
    assert result["total_gpus"] == "Unknown"
