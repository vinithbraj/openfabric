from __future__ import annotations

from aor_runtime.runtime.sql_catalog import SqlColumnRef, SqlSchemaCatalog, SqlTableRef
from aor_runtime.runtime.sql_safety import normalize_pg_relation_quoting, quote_pg_relation


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
                    SqlColumnRef(schema_name="flathr", table_name="Patient", column_name="PatientBirthDate"),
                ],
            ),
            SqlTableRef(
                schema_name="flathr",
                table_name="Study",
                columns=[SqlColumnRef(schema_name="flathr", table_name="Study", column_name="PatientID")],
            ),
            SqlTableRef(
                schema_name="public",
                table_name="lowercase",
                columns=[SqlColumnRef(schema_name="public", table_name="lowercase", column_name="id")],
            ),
        ],
    )


def test_quote_pg_relation_quotes_mixed_case_table() -> None:
    assert quote_pg_relation("flathr", "Patient") == 'flathr."Patient"'
    assert quote_pg_relation("public", "lowercase") == "public.lowercase"


def test_normalizes_schema_qualified_mixed_case_table() -> None:
    assert normalize_pg_relation_quoting("SELECT COUNT(*) FROM flathr.Patient", _catalog()) == 'SELECT COUNT(*) FROM flathr."Patient"'


def test_normalizes_wrong_combined_quote() -> None:
    assert normalize_pg_relation_quoting('SELECT COUNT(*) FROM "flathr.Patient"', _catalog()) == 'SELECT COUNT(*) FROM flathr."Patient"'


def test_leaves_correct_quote_unchanged() -> None:
    assert normalize_pg_relation_quoting('SELECT COUNT(*) FROM flathr."Patient"', _catalog()) == 'SELECT COUNT(*) FROM flathr."Patient"'


def test_normalizes_join_alias_columns_and_where_columns() -> None:
    sql = (
        "SELECT p.PatientID FROM flathr.Patient p "
        "JOIN flathr.Study s ON s.PatientID = p.PatientID "
        "WHERE p.PatientBirthDate <= CURRENT_DATE - INTERVAL '45 years'"
    )
    assert normalize_pg_relation_quoting(sql, _catalog()) == (
        'SELECT p."PatientID" FROM flathr."Patient" p '
        'JOIN flathr."Study" s ON s."PatientID" = p."PatientID" '
        'WHERE p."PatientBirthDate" <= CURRENT_DATE - INTERVAL \'45 years\''
    )


def test_string_literals_are_not_modified() -> None:
    sql = "SELECT 'flathr.Patient' AS literal FROM flathr.Patient p WHERE p.PatientID = 'PatientID'"
    assert normalize_pg_relation_quoting(sql, _catalog()) == (
        'SELECT \'flathr.Patient\' AS literal FROM flathr."Patient" p WHERE p."PatientID" = \'PatientID\''
    )
