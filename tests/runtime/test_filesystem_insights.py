from __future__ import annotations

from aor_runtime.runtime.insights import generate_filesystem_insights


def test_large_filesystem_aggregate_mentions_storage_impact() -> None:
    result = generate_filesystem_insights({"domain": "filesystem", "file_count": 12, "total_size_bytes": 12 * 1024**3})

    assert any(insight.title == "Large storage footprint" for insight in result.insights)
    assert "12 matching files" in result.summary


def test_zero_filesystem_matches_mentions_no_matches() -> None:
    result = generate_filesystem_insights({"domain": "filesystem", "file_count": 0})

    assert any(insight.title == "No matching files" for insight in result.insights)


def test_recursive_filesystem_scope_is_noted() -> None:
    result = generate_filesystem_insights({"domain": "filesystem", "file_count": 2, "recursive": True})

    assert any(insight.title == "Recursive scope" for insight in result.insights)
