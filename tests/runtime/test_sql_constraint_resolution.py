from __future__ import annotations

from aor_runtime.runtime.sql_catalog import SqlColumnRef, SqlSchemaCatalog, SqlTableRef
from aor_runtime.runtime.sql_constraints import extract_sql_constraints, resolve_sql_constraints


def _catalog(*, ambiguous: bool = False, birth_date: bool = True, join_column: bool = True) -> SqlSchemaCatalog:
    patient_columns = [SqlColumnRef(schema_name="flathr", table_name="Patient", column_name="PatientID")]
    if birth_date:
        patient_columns.append(SqlColumnRef(schema_name="flathr", table_name="Patient", column_name="PatientBirthDate"))
    study_columns = [SqlColumnRef(schema_name="flathr", table_name="Study", column_name="StudyInstanceUID")]
    if join_column:
        study_columns.append(SqlColumnRef(schema_name="flathr", table_name="Study", column_name="PatientID"))
    tables = [
        SqlTableRef(schema_name="flathr", table_name="Patient", columns=patient_columns, primary_key_columns=["PatientID"]),
        SqlTableRef(schema_name="flathr", table_name="Study", columns=study_columns),
    ]
    if ambiguous:
        tables.append(SqlTableRef(schema_name="other", table_name="Patient", columns=patient_columns))
    return SqlSchemaCatalog(database="dicom", dialect="postgresql", tables=tables)


def test_age_constraint_resolves_to_birth_date_column() -> None:
    frame = resolve_sql_constraints(extract_sql_constraints("count patients above age 70"), _catalog())
    age = next(constraint for constraint in frame.constraints if constraint.kind == "age_comparison")
    assert age.resolved_column == "flathr.Patient.PatientBirthDate"
    assert not frame.unresolved_constraints


def test_related_studies_resolves_to_study_table_and_shared_join_column() -> None:
    frame = resolve_sql_constraints(extract_sql_constraints("count patients with 2 studies"), _catalog())
    related = next(constraint for constraint in frame.constraints if constraint.kind == "related_row_count")
    assert related.resolved_table == "flathr.Study"
    assert related.metadata["primary_column"] == "PatientID"
    assert related.metadata["related_column"] == "PatientID"


def test_ambiguous_target_table_becomes_unresolved() -> None:
    frame = resolve_sql_constraints(extract_sql_constraints("count patients above age 70"), _catalog(ambiguous=True))
    assert any(constraint.kind == "age_comparison" for constraint in frame.unresolved_constraints)


def test_missing_birth_date_column_becomes_unresolved() -> None:
    frame = resolve_sql_constraints(extract_sql_constraints("count patients above age 70"), _catalog(birth_date=False))
    assert any(constraint.kind == "age_comparison" for constraint in frame.unresolved_constraints)


def test_missing_join_column_becomes_unresolved() -> None:
    frame = resolve_sql_constraints(extract_sql_constraints("count patients with 2 studies"), _catalog(join_column=False))
    assert any(constraint.kind == "related_row_count" for constraint in frame.unresolved_constraints)
