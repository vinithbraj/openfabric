from __future__ import annotations

from aor_runtime.runtime.presentation import PresentationContext, present_result


def test_filesystem_aggregate_renders_human_summary() -> None:
    result = {
        "file_count": 2,
        "total_size_bytes": 4512331776,
        "display_size": "4.2 GB (4512331776 bytes)",
        "matches": [{"relative_path": "a.mp4", "size_bytes": 100}, {"relative_path": "b.mp4", "size_bytes": 200}],
    }

    rendered = present_result(result, PresentationContext(mode="user", source_action="fs.aggregate"))

    assert "Found 2 files totaling 4.2 GB" in rendered.markdown
    assert "`a.mp4`" in rendered.markdown
    assert not rendered.markdown.lstrip().startswith("{")
