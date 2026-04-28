from __future__ import annotations

from aor_runtime.runtime.sql_constraints import extract_sql_constraints


def _projection(prompt: str):
    frame = extract_sql_constraints(prompt)
    assert frame.projections
    return frame.projections[0]


def test_patient_names_projection_is_extracted() -> None:
    projection = _projection("list patient names above age 60 in dicom")
    assert projection.subject == "patient names"
    assert projection.aggregate == "none"


def test_patient_ids_projection_is_extracted() -> None:
    projection = _projection("show patient ids above age 60 in dicom")
    assert projection.subject == "patient ids"


def test_patient_birth_dates_projection_is_extracted() -> None:
    projection = _projection("list patient birth dates above age 60 in dicom")
    assert projection.subject == "patient birth dates"


def test_distinct_modalities_projection_is_extracted() -> None:
    projection = _projection("list distinct modalities in dicom")
    assert projection.subject == "modalities"
    assert projection.distinct is True


def test_top_projection_keeps_limit_constraint() -> None:
    frame = extract_sql_constraints("top 10 patient names in dicom")
    assert frame.query_type == "select"
    assert frame.projections[0].subject == "patient names"
    assert any(constraint.kind == "limit" and constraint.value == 10 for constraint in frame.constraints)


def test_count_patients_does_not_create_projection() -> None:
    frame = extract_sql_constraints("count patients above age 60 in dicom")
    assert frame.query_type == "count"
    assert frame.projections == []


def test_count_distinct_patient_names_creates_count_distinct_projection() -> None:
    frame = extract_sql_constraints("count distinct patient names above age 60 in dicom")
    assert frame.query_type == "count"
    assert frame.projections[0].subject == "patient names"
    assert frame.projections[0].aggregate == "count_distinct"
    assert frame.projections[0].distinct is True


def test_all_rows_request_does_not_create_specific_projection() -> None:
    frame = extract_sql_constraints("list all patients above age 60 in dicom")
    assert frame.projections == []
