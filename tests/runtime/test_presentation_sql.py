from __future__ import annotations

from aor_runtime.runtime.presentation import PresentationContext, present_result


def test_sql_scalar_count_renders_readable_count() -> None:
    result = {"database": "dicom", "row_count": 1, "rows": [{"count_value": 28102}]}

    rendered = present_result(result, PresentationContext(mode="user", source_action="sql.query", output_mode="count"))

    assert rendered.markdown == "Count: 28,102"


def test_sql_rows_render_markdown_table_without_raw_json() -> None:
    result = {
        "database": "dicom",
        "row_count": 2,
        "rows": [{"PatientName": "Alice"}, {"PatientName": "Bob"}],
        "sql_final": "SELECT ...",
    }

    rendered = present_result(result, PresentationContext(mode="user", source_action="sql.query"))

    assert "| PatientName |" in rendered.markdown
    assert "Alice" in rendered.markdown
    assert "sql_final" not in rendered.markdown
    assert not rendered.markdown.lstrip().startswith("{")
