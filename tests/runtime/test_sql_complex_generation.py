from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.runtime.capabilities.base import ClassificationContext, CompileContext
from aor_runtime.runtime.capabilities.sql import SqlCapabilityPack
from aor_runtime.runtime.sql_catalog import SqlColumnRef, SqlSchemaCatalog, SqlTableRef


def _settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        sql_databases={"dicom": "postgresql+psycopg://example/db"},
        sql_default_database="dicom",
        **overrides,
    )


def _catalog(*, birth_date: bool = True) -> SqlSchemaCatalog:
    patient_columns = [
        SqlColumnRef(schema_name="flathr", table_name="Patient", column_name="PatientID"),
        SqlColumnRef(schema_name="flathr", table_name="Patient", column_name="PatientName"),
    ]
    if birth_date:
        patient_columns.append(SqlColumnRef(schema_name="flathr", table_name="Patient", column_name="PatientBirthDate"))
    return SqlSchemaCatalog(
        database="dicom",
        dialect="postgresql",
        tables=[
            SqlTableRef(schema_name="flathr", table_name="Patient", columns=patient_columns, primary_key_columns=["PatientID"]),
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


def _compile(prompt: str, tmp_path: Path, monkeypatch, *, catalog: SqlSchemaCatalog | None = None, **settings_overrides):
    monkeypatch.setattr("aor_runtime.runtime.capabilities.sql.get_sql_catalog", lambda settings, database: catalog or _catalog())
    settings = _settings(tmp_path, **settings_overrides)
    pack = SqlCapabilityPack()
    result = pack.classify(prompt, ClassificationContext(schema_payload=None, allowed_tools=["sql.query"], settings=settings))
    return pack.compile(result.intent, CompileContext(allowed_tools=["sql.query"], settings=settings))


def test_count_patients_above_age_70_generates_covered_read_only_sql(tmp_path: Path, monkeypatch) -> None:
    compiled = _compile("count patients above age 70 in dicom", tmp_path, monkeypatch)
    assert compiled is not None
    query = compiled.plan.steps[0].args["query"]
    assert query.startswith("SELECT COUNT(*)")
    assert '"PatientBirthDate" < CURRENT_DATE - INTERVAL \'70 years\'' in query
    assert compiled.metadata["sql_constraint_coverage_passed"] is True


def test_count_patients_above_age_80_generates_covered_read_only_sql(tmp_path: Path, monkeypatch) -> None:
    compiled = _compile("count patients above age 80 in dicom", tmp_path, monkeypatch)
    assert compiled is not None
    query = compiled.plan.steps[0].args["query"]
    assert '"PatientBirthDate" < CURRENT_DATE - INTERVAL \'80 years\'' in query
    assert compiled.metadata["sql_constraint_coverage_passed"] is True


def test_count_patients_above_age_60_with_two_studies_generates_join_having_sql(tmp_path: Path, monkeypatch) -> None:
    compiled = _compile("count patients above age 60 with 2 studies in dicom", tmp_path, monkeypatch)
    assert compiled is not None
    query = compiled.plan.steps[0].args["query"]
    assert 'JOIN flathr."Study" s' in query
    assert 'p."PatientBirthDate" < CURRENT_DATE - INTERVAL \'60 years\'' in query
    assert 'HAVING COUNT(s."PatientID") = 2' in query
    assert compiled.metadata["sql_constraint_coverage_passed"] is True


def test_list_patient_names_above_age_generates_projection_and_predicate(tmp_path: Path, monkeypatch) -> None:
    compiled = _compile("list patient names above age 60 in dicom", tmp_path, monkeypatch)
    assert compiled is not None
    query = compiled.plan.steps[0].args["query"]
    assert query.startswith('SELECT "PatientName" FROM flathr."Patient"')
    assert '"PatientBirthDate" < CURRENT_DATE - INTERVAL \'60 years\'' in query
    assert compiled.metadata["sql_projection_coverage_passed"] is True


def test_unresolved_projection_never_emits_select_star(tmp_path: Path, monkeypatch) -> None:
    catalog = SqlSchemaCatalog(
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
                columns=[
                    SqlColumnRef(schema_name="flathr", table_name="Study", column_name="StudyDate"),
                ],
            ),
        ],
    )
    compiled = _compile("list dates in dicom", tmp_path, monkeypatch, catalog=catalog, enable_sql_llm_generation=False)
    assert compiled is not None
    assert [step.action for step in compiled.plan.steps] == ["runtime.return"]
    assert compiled.metadata["sql_error_class"] == "sql_projection_unresolved"


def test_query_is_not_executed_when_constraint_resolution_fails(tmp_path: Path, monkeypatch) -> None:
    compiled = _compile(
        "count patients above age 70 in dicom",
        tmp_path,
        monkeypatch,
        catalog=_catalog(birth_date=False),
        enable_sql_llm_generation=False,
    )
    assert compiled is not None
    assert [step.action for step in compiled.plan.steps] == ["runtime.return"]
    assert compiled.metadata["sql_error_class"] == "sql_constraint_unresolved"
