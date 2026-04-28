from __future__ import annotations

from aor_runtime.runtime.sql_catalog import SqlColumnRef, SqlSchemaCatalog, SqlTableRef
from aor_runtime.runtime.sql_resolver import resolve_column_name, resolve_sql_references, resolve_table_name


def _catalog(*, ambiguous: bool = False) -> SqlSchemaCatalog:
    tables = [
        SqlTableRef(
            schema_name="flathr",
            table_name="Patient",
            columns=[
                SqlColumnRef(schema_name="flathr", table_name="Patient", column_name="PatientID"),
                SqlColumnRef(schema_name="flathr", table_name="Patient", column_name="PatientName"),
                SqlColumnRef(schema_name="flathr", table_name="Patient", column_name="PatientBirthDate"),
            ],
        ),
        SqlTableRef(
            schema_name="flathr",
            table_name="Study",
            columns=[SqlColumnRef(schema_name="flathr", table_name="Study", column_name="StudyInstanceUID")],
        ),
    ]
    if ambiguous:
        tables.append(SqlTableRef(schema_name="other", table_name="Patient", columns=[]))
    return SqlSchemaCatalog(database="dicom", dialect="postgresql", tables=tables)


def test_resolves_plural_patient_to_mixed_case_table() -> None:
    table = resolve_table_name("patients", _catalog())
    assert table is not None
    assert table.qualified_name == "flathr.Patient"


def test_resolves_schema_qualified_table() -> None:
    table = resolve_table_name("flathr.Patient", _catalog())
    assert table is not None
    assert table.qualified_name == "flathr.Patient"


def test_ambiguous_table_is_not_guessed() -> None:
    assert resolve_table_name("patients", _catalog(ambiguous=True)) is None
    context = resolve_sql_references("count patients", _catalog(ambiguous=True))
    assert context.ambiguous_tables


def test_age_resolves_to_birth_date_only_when_present() -> None:
    table = resolve_table_name("patients", _catalog())
    assert table is not None
    column = resolve_column_name("age", table)
    assert column is not None
    assert column.column_name == "PatientBirthDate"


def test_patient_name_resolves_to_patient_name() -> None:
    table = resolve_table_name("patients", _catalog())
    assert table is not None
    column = resolve_column_name("patient name", table)
    assert column is not None
    assert column.column_name == "PatientName"
