from __future__ import annotations

from aor_runtime.runtime.insights import generate_sql_insights


def test_zero_count_result_mentions_no_matches() -> None:
    result = generate_sql_insights({"domain": "sql", "result_count": 0, "result": {"count": 0}})

    assert any(insight.title == "No matching records" for insight in result.insights)
    assert "count of 0" in result.summary


def test_constrained_query_mentions_constraints() -> None:
    result = generate_sql_insights({"domain": "sql", "result_count": 12, "constraints_applied": ["age > 70"]})

    assert any(insight.title == "Constraints were applied" for insight in result.insights)


def test_truncated_sql_result_warns() -> None:
    result = generate_sql_insights({"domain": "sql", "row_count": 500, "truncated": True})

    assert any(insight.title == "Result was truncated" for insight in result.insights)
