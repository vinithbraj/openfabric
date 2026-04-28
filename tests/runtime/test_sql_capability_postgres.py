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


def _classify(prompt: str, tmp_path: Path, monkeypatch, **settings_overrides):
    monkeypatch.setattr("aor_runtime.runtime.capabilities.sql.get_sql_catalog", lambda settings, database: _catalog())
    settings = _settings(tmp_path, **settings_overrides)
    pack = SqlCapabilityPack()
    return pack, settings, pack.classify(
        prompt,
        ClassificationContext(schema_payload=None, allowed_tools=["sql.query"], settings=settings),
    )


def test_list_all_tables_returns_non_public_schema(tmp_path: Path, monkeypatch) -> None:
    pack, settings, result = _classify("list all tables in dicom", tmp_path, monkeypatch)
    assert result.matched is True
    compiled = pack.compile(result.intent, CompileContext(allowed_tools=["sql.query"], settings=settings))
    assert compiled is not None
    assert compiled.plan.steps[0].action == "runtime.return"
    assert "flathr.Patient" in compiled.plan.steps[0].args["value"]


def test_count_patients_uses_qualified_mixed_case_table(tmp_path: Path, monkeypatch) -> None:
    pack, settings, result = _classify("count all patients in dicom", tmp_path, monkeypatch)
    compiled = pack.compile(result.intent, CompileContext(allowed_tools=["sql.query"], settings=settings))
    assert compiled is not None
    assert compiled.plan.steps[0].args["query"] == 'SELECT COUNT(*) AS count_value FROM flathr."Patient"'


def test_count_patients_above_age_uses_birth_date(tmp_path: Path, monkeypatch) -> None:
    pack, settings, result = _classify("list the count of patients above 45 years of age in dicom", tmp_path, monkeypatch)
    compiled = pack.compile(result.intent, CompileContext(allowed_tools=["sql.query"], settings=settings))
    assert compiled is not None
    query = compiled.plan.steps[0].args["query"]
    assert 'FROM flathr."Patient"' in query
    assert '"PatientBirthDate" < CURRENT_DATE - INTERVAL \'45 years\'' in query
    assert compiled.metadata["sql_constraint_coverage_passed"] is True


def test_count_patients_above_age_70_is_filtered(tmp_path: Path, monkeypatch) -> None:
    pack, settings, result = _classify("list the count of all patients above age 70 in dicom", tmp_path, monkeypatch)
    compiled = pack.compile(result.intent, CompileContext(allowed_tools=["sql.query"], settings=settings))
    assert compiled is not None
    query = compiled.plan.steps[0].args["query"]
    assert query != 'SELECT COUNT(*) AS count_value FROM flathr."Patient"'
    assert '"PatientBirthDate" < CURRENT_DATE - INTERVAL \'70 years\'' in query
    assert compiled.metadata["sql_constraint_coverage_passed"] is True
    assert not compiled.metadata["sql_constraints_missing"]


def test_count_patients_above_age_80_is_filtered(tmp_path: Path, monkeypatch) -> None:
    pack, settings, result = _classify("list the count of all patients above age 80 in dicom", tmp_path, monkeypatch)
    compiled = pack.compile(result.intent, CompileContext(allowed_tools=["sql.query"], settings=settings))
    assert compiled is not None
    query = compiled.plan.steps[0].args["query"]
    assert query != 'SELECT COUNT(*) AS count_value FROM flathr."Patient"'
    assert '"PatientBirthDate" < CURRENT_DATE - INTERVAL \'80 years\'' in query
    assert compiled.metadata["sql_constraint_coverage_passed"] is True


def test_count_patients_above_age_with_exact_study_count_is_fully_constrained(tmp_path: Path, monkeypatch) -> None:
    pack, settings, result = _classify(
        "list the count of all patients above age 60 and with 2 studies in dicom",
        tmp_path,
        monkeypatch,
    )
    compiled = pack.compile(result.intent, CompileContext(allowed_tools=["sql.query"], settings=settings))
    assert compiled is not None
    query = compiled.plan.steps[0].args["query"]
    assert 'FROM flathr."Patient" p' in query
    assert 'JOIN flathr."Study" s ON s."PatientID" = p."PatientID"' in query
    assert 'p."PatientBirthDate" < CURRENT_DATE - INTERVAL \'60 years\'' in query
    assert 'HAVING COUNT(s."PatientID") = 2' in query
    assert compiled.metadata["sql_constraint_coverage_passed"] is True
    assert not compiled.metadata["sql_constraints_missing"]


def test_count_patients_with_no_studies_uses_left_join_without_quoting_join_keyword(tmp_path: Path, monkeypatch) -> None:
    pack, settings, result = _classify("list the count of all patients with no studies in dicom", tmp_path, monkeypatch)
    compiled = pack.compile(result.intent, CompileContext(allowed_tools=["sql.query"], settings=settings))
    assert compiled is not None
    query = compiled.plan.steps[0].args["query"]
    assert 'LEFT JOIN flathr."Study" s ON s."PatientID" = p."PatientID"' in query
    assert '"LEFT"' not in query
    assert 'HAVING COUNT(s."PatientID") = 0' in query
    assert compiled.metadata["sql_constraint_coverage_passed"] is True


def test_list_patient_names_above_age_selects_projection_and_predicate(tmp_path: Path, monkeypatch) -> None:
    pack, settings, result = _classify("list patient names above age 60 in dicom", tmp_path, monkeypatch)
    compiled = pack.compile(result.intent, CompileContext(allowed_tools=["sql.query"], settings=settings))
    assert compiled is not None
    query = compiled.plan.steps[0].args["query"]
    assert query.startswith('SELECT "PatientName" FROM flathr."Patient"')
    assert '"PatientBirthDate" < CURRENT_DATE - INTERVAL \'60 years\'' in query
    assert compiled.metadata["sql_projection_coverage_passed"] is True
    assert compiled.metadata["sql_projections_missing"] == []


def test_list_patient_ids_above_age_selects_projection_and_predicate(tmp_path: Path, monkeypatch) -> None:
    pack, settings, result = _classify("list patient ids above age 60 in dicom", tmp_path, monkeypatch)
    compiled = pack.compile(result.intent, CompileContext(allowed_tools=["sql.query"], settings=settings))
    assert compiled is not None
    query = compiled.plan.steps[0].args["query"]
    assert query.startswith('SELECT "PatientID" FROM flathr."Patient"')
    assert '"PatientBirthDate" < CURRENT_DATE - INTERVAL \'60 years\'' in query
    assert compiled.metadata["sql_projection_coverage_passed"] is True


def test_list_patient_birth_dates_above_age_selects_projection_and_predicate(tmp_path: Path, monkeypatch) -> None:
    pack, settings, result = _classify("list patient birth dates above age 60 in dicom", tmp_path, monkeypatch)
    compiled = pack.compile(result.intent, CompileContext(allowed_tools=["sql.query"], settings=settings))
    assert compiled is not None
    query = compiled.plan.steps[0].args["query"]
    assert query.startswith('SELECT "PatientBirthDate" FROM flathr."Patient"')
    assert '"PatientBirthDate" < CURRENT_DATE - INTERVAL \'60 years\'' in query
    assert compiled.metadata["sql_projection_coverage_passed"] is True


def test_list_distinct_modalities_uses_schema_resolved_projection(tmp_path: Path, monkeypatch) -> None:
    pack, settings, result = _classify("list distinct modalities in dicom", tmp_path, monkeypatch)
    compiled = pack.compile(result.intent, CompileContext(allowed_tools=["sql.query"], settings=settings))
    assert compiled is not None
    query = compiled.plan.steps[0].args["query"]
    assert query == 'SELECT DISTINCT "Modality" FROM flathr."Study" ORDER BY "Modality"'
    assert compiled.metadata["sql_projection_coverage_passed"] is True


def test_list_patient_names_with_exactly_two_studies_covers_projection_and_related_count(tmp_path: Path, monkeypatch) -> None:
    pack, settings, result = _classify("list patient names with exactly 2 studies in dicom", tmp_path, monkeypatch)
    compiled = pack.compile(result.intent, CompileContext(allowed_tools=["sql.query"], settings=settings))
    assert compiled is not None
    query = compiled.plan.steps[0].args["query"]
    assert 'SELECT p."PatientName"' in query
    assert 'JOIN flathr."Study" s ON s."PatientID" = p."PatientID"' in query
    assert 'GROUP BY p."PatientID", p."PatientName"' in query
    assert 'HAVING COUNT(s."PatientID") = 2' in query
    assert compiled.metadata["sql_projection_coverage_passed"] is True
    assert compiled.metadata["sql_constraint_coverage_passed"] is True


def test_ambiguous_dates_projection_does_not_execute_sql_query(tmp_path: Path, monkeypatch) -> None:
    catalog = _catalog()
    catalog.tables[1].columns.append(SqlColumnRef(schema_name="flathr", table_name="Study", column_name="StudyDate"))
    monkeypatch.setattr("aor_runtime.runtime.capabilities.sql.get_sql_catalog", lambda settings, database: catalog)
    settings = _settings(tmp_path)
    pack = SqlCapabilityPack()
    result = pack.classify(
        "list dates in dicom",
        ClassificationContext(schema_payload=None, allowed_tools=["sql.query"], settings=settings),
    )
    compiled = pack.compile(result.intent, CompileContext(allowed_tools=["sql.query"], settings=settings))
    assert compiled is not None
    assert [step.action for step in compiled.plan.steps] == ["runtime.return"]
    assert compiled.metadata["sql_error_class"] == "sql_projection_unresolved"
    assert compiled.metadata["sql_projections_missing"] == ["p1"]


def test_describe_patient_returns_columns(tmp_path: Path, monkeypatch) -> None:
    pack, settings, result = _classify("describe Patient in dicom", tmp_path, monkeypatch)
    compiled = pack.compile(result.intent, CompileContext(allowed_tools=["sql.query"], settings=settings))
    assert compiled is not None
    assert "PatientBirthDate" in compiled.plan.steps[0].args["value"]


def test_mutation_request_is_rejected_before_sql_query(tmp_path: Path, monkeypatch) -> None:
    pack, settings, result = _classify("delete all patients in dicom", tmp_path, monkeypatch)
    compiled = pack.compile(result.intent, CompileContext(allowed_tools=["sql.query"], settings=settings))
    assert compiled is not None
    assert [step.action for step in compiled.plan.steps] == ["runtime.return"]
    assert "read-only" in compiled.plan.steps[0].args["value"]


def test_broad_sql_uses_sql_llm_path_without_raw_planner(tmp_path: Path, monkeypatch) -> None:
    class FakeLLM:
        def __init__(self, settings):
            self.settings = settings

        def complete_json(self, **kwargs):
            return {
                "matched": True,
                "sql": "SELECT s.Modality, COUNT(*) AS study_count FROM flathr.Study s GROUP BY s.Modality",
                "confidence": 0.95,
                "reason": "grouped count",
                "tables_used": ["flathr.Study"],
                "columns_used": ["flathr.Study.Modality"],
            }

    monkeypatch.setattr("aor_runtime.runtime.capabilities.sql.get_sql_catalog", lambda settings, database: _catalog())
    monkeypatch.setattr("aor_runtime.runtime.capabilities.sql.LLMClient", FakeLLM)
    monkeypatch.setattr("aor_runtime.runtime.capabilities.sql.explain_sql_query", lambda *args, **kwargs: None)
    settings = _settings(tmp_path, enable_sql_llm_generation=True)
    pack = SqlCapabilityPack()
    result = pack.classify(
        "count studies by modality in dicom",
        ClassificationContext(schema_payload=None, allowed_tools=["sql.query"], settings=settings),
    )

    assert result.matched is True
    assert result.metadata["raw_planner_llm_calls"] == 0
    compiled = pack.compile(result.intent, CompileContext(allowed_tools=["sql.query"], settings=settings))
    assert compiled is not None
    assert compiled.planning_mode == "sql_llm_generation"
    query = compiled.plan.steps[0].args["query"]
    assert 'flathr."Study"' in query
    assert 's."Modality"' in query
