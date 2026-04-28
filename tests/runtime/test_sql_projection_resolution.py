from __future__ import annotations

from aor_runtime.runtime.sql_catalog import SqlColumnRef, SqlSchemaCatalog, SqlTableRef
from aor_runtime.runtime.sql_constraints import extract_sql_constraints, resolve_sql_constraints


def _catalog(*, ambiguous_dates: bool = False) -> SqlSchemaCatalog:
    study_columns = [
        SqlColumnRef(schema_name="flathr", table_name="Study", column_name="PatientID"),
        SqlColumnRef(schema_name="flathr", table_name="Study", column_name="Modality"),
    ]
    if ambiguous_dates:
        study_columns.append(SqlColumnRef(schema_name="flathr", table_name="Study", column_name="StudyDate"))
    return SqlSchemaCatalog(
        database="dicom",
        dialect="postgresql",
        tables=[
            SqlTableRef(
                schema_name="flathr",
                table_name="Patient",
                columns=[
                    SqlColumnRef(schema_name="flathr", table_name="Patient", column_name="PatientID"),
                    SqlColumnRef(schema_name="flathr", table_name="Patient", column_name="PatientName"),
                    SqlColumnRef(schema_name="flathr", table_name="Patient", column_name="PatientBirthDate"),
                ],
            ),
            SqlTableRef(schema_name="flathr", table_name="Study", columns=study_columns),
        ],
    )


def _first_projection(prompt: str, *, ambiguous_dates: bool = False):
    frame = resolve_sql_constraints(extract_sql_constraints(prompt), _catalog(ambiguous_dates=ambiguous_dates))
    assert frame.projections
    return frame.projections[0], frame


def test_patient_names_resolve_to_patient_name() -> None:
    projection, frame = _first_projection("list patient names above age 60 in dicom")
    assert projection.resolved_column == "flathr.Patient.PatientName"
    assert frame.unresolved_projections == []


def test_patient_ids_resolve_to_patient_id() -> None:
    projection, frame = _first_projection("list patient ids above age 60 in dicom")
    assert projection.resolved_column == "flathr.Patient.PatientID"
    assert frame.unresolved_projections == []


def test_patient_birth_dates_resolve_to_patient_birth_date() -> None:
    projection, frame = _first_projection("list patient birth dates above age 60 in dicom")
    assert projection.resolved_column == "flathr.Patient.PatientBirthDate"
    assert frame.unresolved_projections == []


def test_distinct_modalities_resolves_unique_catalog_column_and_table() -> None:
    projection, frame = _first_projection("list distinct modalities in dicom")
    assert projection.resolved_table == "flathr.Study"
    assert projection.resolved_column == "flathr.Study.Modality"
    assert frame.target_entity == "flathr.Study"


def test_ambiguous_dates_are_unresolved() -> None:
    projection, frame = _first_projection("list dates in dicom", ambiguous_dates=True)
    assert projection.resolved_column is None
    assert frame.unresolved_projections == [projection]
