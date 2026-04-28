from __future__ import annotations

from aor_runtime.runtime.sql_constraints import extract_sql_constraints


def _first(kind: str, prompt: str):
    frame = extract_sql_constraints(prompt)
    return next(constraint for constraint in frame.constraints if constraint.kind == kind)


def test_above_age_is_strict_gt() -> None:
    constraint = _first("age_comparison", "count patients above age 70")
    assert constraint.operator == "gt"
    assert constraint.value == 70


def test_older_than_is_strict_gt() -> None:
    constraint = _first("age_comparison", "count patients older than 70")
    assert constraint.operator == "gt"
    assert constraint.value == 70


def test_age_and_above_is_gte() -> None:
    constraint = _first("age_comparison", "count patients 70 and above")
    assert constraint.operator == "gte"
    assert constraint.value == 70


def test_at_least_age_is_gte() -> None:
    constraint = _first("age_comparison", "count patients at least 70")
    assert constraint.operator == "gte"
    assert constraint.value == 70


def test_below_age_is_lt() -> None:
    constraint = _first("age_comparison", "count patients below age 18")
    assert constraint.operator == "lt"
    assert constraint.value == 18


def test_between_ages_extracts_between() -> None:
    constraint = _first("age_comparison", "count patients between 40 and 60 years old")
    assert constraint.operator == "between"
    assert constraint.value == {"lower": 40, "upper": 60}


def test_with_two_related_rows_is_exact_count() -> None:
    constraint = _first("related_row_count", "count patients with 2 studies")
    assert constraint.operator == "eq"
    assert constraint.value == 2
    assert constraint.subject == "studies"


def test_with_more_than_related_rows_is_gt() -> None:
    constraint = _first("related_row_count", "count patients with more than 2 studies")
    assert constraint.operator == "gt"
    assert constraint.value == 2


def test_with_at_least_related_rows_is_gte() -> None:
    constraint = _first("related_row_count", "count patients with at least 2 studies")
    assert constraint.operator == "gte"
    assert constraint.value == 2


def test_with_no_related_rows_is_zero() -> None:
    constraint = _first("related_row_count", "count patients with no studies")
    assert constraint.operator == "eq"
    assert constraint.value == 0


def test_unknown_with_constraint_is_preserved() -> None:
    frame = extract_sql_constraints("count patients with recent suspicious activity")
    unknown = _first("unknown", "count patients with recent suspicious activity")
    assert unknown in frame.unresolved_constraints
