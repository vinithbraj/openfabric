from __future__ import annotations

from aor_runtime.runtime.sql_catalog import SqlColumnRef, SqlSchemaCatalog, SqlTableRef
from aor_runtime.runtime.sql_constraints import extract_sql_constraints, resolve_sql_constraints
from aor_runtime.runtime.sql_llm import coerce_sql_generation, generate_sql_for_goal


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def complete_json(self, **kwargs):
        self.calls += 1
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


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
            )
        ],
    )


def _join_catalog() -> SqlSchemaCatalog:
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
                columns=[SqlColumnRef(schema_name="flathr", table_name="Study", column_name="PatientID")],
            ),
        ],
    )


def test_valid_sql_json_is_accepted_and_normalized() -> None:
    result = coerce_sql_generation(
        {
            "matched": True,
            "sql": "SELECT COUNT(*) AS count_value FROM flathr.Patient",
            "confidence": 0.95,
            "reason": "ok",
            "tables_used": ["flathr.Patient"],
            "columns_used": [],
        },
        catalog=_catalog(),
    )
    assert result.matched is True
    assert result.normalized_sql == 'SELECT COUNT(*) AS count_value FROM flathr."Patient"'


def test_malformed_json_is_rejected() -> None:
    result = generate_sql_for_goal(llm=FakeLLM(ValueError("bad json")), goal="count patients", catalog=_catalog())
    assert result.matched is False
    assert "Malformed" in result.reason


def test_non_select_sql_rejected() -> None:
    result = coerce_sql_generation(
        {"matched": True, "sql": "DELETE FROM flathr.Patient", "confidence": 0.99},
        catalog=_catalog(),
    )
    assert result.matched is False
    assert result.validation_failure


def test_low_confidence_rejected() -> None:
    result = coerce_sql_generation(
        {"matched": True, "sql": "SELECT 1", "confidence": 0.2},
        catalog=_catalog(),
    )
    assert result.matched is False


def test_unknown_table_reference_rejected() -> None:
    result = coerce_sql_generation(
        {
            "matched": True,
            "sql": "SELECT * FROM missing.Patient",
            "confidence": 0.95,
            "tables_used": ["missing.Patient"],
        },
        catalog=_catalog(),
    )
    assert result.matched is False
    assert "unknown table" in result.reason


def test_tool_call_like_output_rejected() -> None:
    result = coerce_sql_generation(
        {"matched": True, "sql": "sql.query({\"query\":\"SELECT 1\"})", "confidence": 0.95},
        catalog=_catalog(),
    )
    assert result.matched is False
    assert "tool-call" in result.reason


def test_llm_sql_missing_extracted_age_constraint_is_rejected() -> None:
    catalog = _catalog()
    frame = resolve_sql_constraints(extract_sql_constraints("count patients above age 70"), catalog)
    result = coerce_sql_generation(
        {
            "matched": True,
            "sql": "SELECT COUNT(*) AS count_value FROM flathr.Patient",
            "confidence": 0.95,
            "tables_used": ["flathr.Patient"],
        },
        catalog=catalog,
        constraint_frame=frame,
    )
    assert result.matched is False
    assert "constraint coverage" in str(result.validation_failure)


def test_llm_sql_covering_extracted_age_constraint_is_accepted() -> None:
    catalog = _catalog()
    frame = resolve_sql_constraints(extract_sql_constraints("count patients above age 70"), catalog)
    result = coerce_sql_generation(
        {
            "matched": True,
            "sql": "SELECT COUNT(*) AS count_value FROM flathr.Patient WHERE PatientBirthDate < CURRENT_DATE - INTERVAL '70 years'",
            "confidence": 0.95,
            "tables_used": ["flathr.Patient"],
            "columns_used": ["flathr.Patient.PatientBirthDate"],
            "constraints_addressed": ["target_table", "c1"],
        },
        catalog=catalog,
        constraint_frame=frame,
    )
    assert result.matched is True
    assert 'flathr."Patient"' in str(result.normalized_sql)
    assert '"PatientBirthDate" < CURRENT_DATE' in str(result.normalized_sql)


def test_llm_sql_missing_extracted_study_count_constraint_is_rejected() -> None:
    catalog = _join_catalog()
    frame = resolve_sql_constraints(extract_sql_constraints("count patients with 2 studies"), catalog)
    result = coerce_sql_generation(
        {
            "matched": True,
            "sql": "SELECT COUNT(*) AS count_value FROM flathr.Patient",
            "confidence": 0.95,
            "tables_used": ["flathr.Patient"],
        },
        catalog=catalog,
        constraint_frame=frame,
    )
    assert result.matched is False
    assert "constraint coverage" in str(result.validation_failure)


def test_llm_sql_missing_requested_projection_is_rejected() -> None:
    catalog = _catalog()
    frame = resolve_sql_constraints(extract_sql_constraints("list patient names above age 70"), catalog)
    result = coerce_sql_generation(
        {
            "matched": True,
            "sql": "SELECT PatientID FROM flathr.Patient WHERE PatientBirthDate < CURRENT_DATE - INTERVAL '70 years'",
            "confidence": 0.95,
            "tables_used": ["flathr.Patient"],
            "columns_used": ["flathr.Patient.PatientID", "flathr.Patient.PatientBirthDate"],
        },
        catalog=catalog,
        constraint_frame=frame,
    )
    assert result.matched is False
    assert "projection" in str(result.validation_failure)


def test_llm_sql_with_projection_but_missing_age_constraint_is_rejected() -> None:
    catalog = _catalog()
    frame = resolve_sql_constraints(extract_sql_constraints("list patient names above age 70"), catalog)
    result = coerce_sql_generation(
        {
            "matched": True,
            "sql": "SELECT PatientName FROM flathr.Patient",
            "confidence": 0.95,
            "tables_used": ["flathr.Patient"],
            "columns_used": ["flathr.Patient.PatientName"],
        },
        catalog=catalog,
        constraint_frame=frame,
    )
    assert result.matched is False
    assert "constraint coverage" in str(result.validation_failure)
