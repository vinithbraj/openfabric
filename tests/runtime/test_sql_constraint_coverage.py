from __future__ import annotations

from aor_runtime.runtime.sql_catalog import SqlColumnRef, SqlSchemaCatalog, SqlTableRef
from aor_runtime.runtime.sql_constraints import extract_sql_constraints, resolve_sql_constraints, validate_sql_constraint_coverage


def _catalog() -> SqlSchemaCatalog:
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
                primary_key_columns=["PatientID"],
            ),
            SqlTableRef(
                schema_name="flathr",
                table_name="Study",
                columns=[
                    SqlColumnRef(schema_name="flathr", table_name="Study", column_name="PatientID"),
                    SqlColumnRef(schema_name="flathr", table_name="Study", column_name="Modality"),
                ],
            ),
        ],
    )


def _frame(prompt: str):
    return resolve_sql_constraints(extract_sql_constraints(prompt), _catalog())


def test_plain_count_fails_when_age_constraint_exists() -> None:
    frame = _frame("count patients above age 70")
    result = validate_sql_constraint_coverage(frame, 'SELECT COUNT(*) AS count_value FROM flathr."Patient"')
    assert result.valid is False


def test_age_predicate_passes_age_coverage() -> None:
    frame = _frame("count patients above age 70")
    result = validate_sql_constraint_coverage(
        frame,
        'SELECT COUNT(*) AS count_value FROM flathr."Patient" WHERE "PatientBirthDate" < CURRENT_DATE - INTERVAL \'70 years\'',
    )
    assert result.valid is True


def test_plain_count_fails_when_related_count_exists() -> None:
    frame = _frame("count patients with 2 studies")
    result = validate_sql_constraint_coverage(frame, 'SELECT COUNT(*) AS count_value FROM flathr."Patient"')
    assert result.valid is False


def test_grouped_having_passes_related_count_coverage() -> None:
    frame = _frame("count patients with 2 studies")
    sql = (
        'SELECT COUNT(*) AS count_value FROM (SELECT p."PatientID" FROM flathr."Patient" p '
        'JOIN flathr."Study" s ON s."PatientID" = p."PatientID" '
        'GROUP BY p."PatientID" HAVING COUNT(s."PatientID") = 2) q'
    )
    result = validate_sql_constraint_coverage(frame, sql)
    assert result.valid is True


def test_sql_missing_one_of_two_constraints_fails() -> None:
    frame = _frame("count patients above age 60 with 2 studies")
    sql = (
        'SELECT COUNT(*) AS count_value FROM (SELECT p."PatientID" FROM flathr."Patient" p '
        'JOIN flathr."Study" s ON s."PatientID" = p."PatientID" '
        'GROUP BY p."PatientID" HAVING COUNT(s."PatientID") = 2) q'
    )
    result = validate_sql_constraint_coverage(frame, sql)
    assert result.valid is False
    assert any(frame.constraint_by_id(item).kind == "age_comparison" for item in result.missing_constraint_ids if frame.constraint_by_id(item))


def test_group_and_limit_coverage_pass_when_clauses_exist() -> None:
    frame = _frame("count studies by modality in dicom")
    sql = 'SELECT s."Modality", COUNT(*) AS count_value FROM flathr."Study" s GROUP BY s."Modality" LIMIT 10'
    result = validate_sql_constraint_coverage(frame, sql)
    assert result.valid is True


def test_select_star_fails_when_specific_projection_requested() -> None:
    frame = _frame("list patient names above age 60 in dicom")
    sql = 'SELECT * FROM flathr."Patient" WHERE "PatientBirthDate" < CURRENT_DATE - INTERVAL \'60 years\''
    result = validate_sql_constraint_coverage(frame, sql)
    assert result.valid is False
    assert result.missing_projection_ids == ["p1"]


def test_sql_missing_projected_column_fails_projection_coverage() -> None:
    frame = _frame("list patient names above age 60 in dicom")
    sql = 'SELECT "PatientID" FROM flathr."Patient" WHERE "PatientBirthDate" < CURRENT_DATE - INTERVAL \'60 years\''
    result = validate_sql_constraint_coverage(frame, sql)
    assert result.valid is False
    assert result.missing_projection_ids == ["p1"]


def test_projected_column_and_age_predicate_pass_coverage() -> None:
    frame = _frame("list patient names above age 60 in dicom")
    sql = 'SELECT "PatientName" FROM flathr."Patient" WHERE "PatientBirthDate" < CURRENT_DATE - INTERVAL \'60 years\''
    result = validate_sql_constraint_coverage(frame, sql)
    assert result.valid is True
    assert result.covered_projection_ids == ["p1"]


def test_distinct_projection_requires_distinct_clause() -> None:
    frame = _frame("list distinct modalities in dicom")
    sql = 'SELECT "Modality" FROM flathr."Study"'
    result = validate_sql_constraint_coverage(frame, sql)
    assert result.valid is False
    assert result.missing_projection_ids == ["p1"]


def test_distinct_projection_passes_with_select_distinct() -> None:
    frame = _frame("list distinct modalities in dicom")
    sql = 'SELECT DISTINCT "Modality" FROM flathr."Study"'
    result = validate_sql_constraint_coverage(frame, sql)
    assert result.valid is True


def test_count_distinct_projection_passes_with_count_distinct() -> None:
    frame = _frame("count distinct patient names above age 60 in dicom")
    sql = (
        'SELECT COUNT(DISTINCT "PatientName") AS count_value FROM flathr."Patient" '
        'WHERE "PatientBirthDate" < CURRENT_DATE - INTERVAL \'60 years\''
    )
    result = validate_sql_constraint_coverage(frame, sql)
    assert result.valid is True
