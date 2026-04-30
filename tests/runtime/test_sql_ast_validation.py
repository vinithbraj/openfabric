from __future__ import annotations

from pathlib import Path

import pytest

from aor_runtime.config import Settings
from aor_runtime.runtime.sql_ast_validation import normalize_and_validate_sql_ast
from aor_runtime.runtime.sql_catalog import SqlColumnRef, SqlSchemaCatalog, SqlTableRef
from aor_runtime.tools.base import ToolExecutionError
from aor_runtime.tools.sql import SQLValidateTool


def _table(name: str, columns: list[str]) -> SqlTableRef:
    return SqlTableRef(
        schema_name="flathr",
        table_name=name,
        columns=[SqlColumnRef(schema_name="flathr", table_name=name, column_name=column) for column in columns],
    )


def _catalog() -> SqlSchemaCatalog:
    return SqlSchemaCatalog(
        database="dicom",
        dialect="postgresql",
        tables=[
            _table("Patient", ["PatientID", "PatientBirthDate"]),
            _table("Study", ["PatientID", "StudyInstanceUID", "StudyDescription"]),
            _table("Series", ["StudyInstanceUID", "SeriesInstanceUID", "SeriesDescription", "Modality"]),
            _table("Instance", ["SeriesInstanceUID", "SOPInstanceUID"]),
            _table("RTSTRUCT", ["SOPInstanceUID"]),
            _table("StructureSetROISequence", ["SOPInstanceUID", "ROIName", "ROIGenerationAlgorithm"]),
        ],
    )


def test_ast_validation_repairs_unquoted_mixed_case_column() -> None:
    result = normalize_and_validate_sql_ast(
        'SELECT COUNT(DISTINCT PatientID) FROM flathr."Patient"',
        _catalog(),
    )

    assert result.valid is True
    assert 'COUNT(DISTINCT "PatientID")' in result.normalized_sql


def test_ast_validation_rejects_column_on_wrong_alias() -> None:
    result = normalize_and_validate_sql_ast(
        'SELECT COUNT(*) FROM flathr."RTSTRUCT" s WHERE s."PatientID" IS NOT NULL',
        _catalog(),
    )

    assert result.valid is False
    assert any("s.PatientID" in message for message in result.messages)
    assert any("Candidate columns" in message for message in result.messages)


def test_ast_validation_rejects_ambiguous_unqualified_subquery_column() -> None:
    result = normalize_and_validate_sql_ast(
        """
        SELECT COUNT(DISTINCT st."PatientID") AS patient_count
        FROM flathr."Study" st
        JOIN flathr."Series" se ON st."StudyInstanceUID" = se."StudyInstanceUID"
        JOIN flathr."Instance" i ON se."SeriesInstanceUID" = i."SeriesInstanceUID"
        JOIN flathr."RTSTRUCT" rs ON i."SOPInstanceUID" = rs."SOPInstanceUID"
        WHERE rs."SOPInstanceUID" IN (
          SELECT "SOPInstanceUID"
          FROM flathr."RTSTRUCT" s
          JOIN flathr."StructureSetROISequence" roi ON s."SOPInstanceUID" = roi."SOPInstanceUID"
          WHERE LOWER(roi."ROIName") LIKE '%brain%'
        )
        """,
        _catalog(),
    )

    assert result.valid is False
    assert any("ambiguous column: SOPInstanceUID" in message for message in result.messages)


def test_ast_validation_accepts_qualified_subquery_column() -> None:
    result = normalize_and_validate_sql_ast(
        """
        SELECT COUNT(DISTINCT st."PatientID") AS patient_count
        FROM flathr."Study" st
        JOIN flathr."Series" se ON st."StudyInstanceUID" = se."StudyInstanceUID"
        JOIN flathr."Instance" i ON se."SeriesInstanceUID" = i."SeriesInstanceUID"
        JOIN flathr."RTSTRUCT" rs ON i."SOPInstanceUID" = rs."SOPInstanceUID"
        WHERE rs."SOPInstanceUID" IN (
          SELECT s."SOPInstanceUID"
          FROM flathr."RTSTRUCT" s
          JOIN flathr."StructureSetROISequence" roi ON s."SOPInstanceUID" = roi."SOPInstanceUID"
          WHERE LOWER(roi."ROIName") LIKE '%brain%'
        )
        """,
        _catalog(),
    )

    assert result.valid is True


def test_sql_validate_tool_uses_ast_validation_before_execution(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        sql_databases={"dicom": "postgresql+psycopg://example/db"},
        sql_default_database="dicom",
    )
    monkeypatch.setattr("aor_runtime.tools.sql.get_sql_catalog", lambda _settings, _database: _catalog())

    with pytest.raises(ToolExecutionError, match="s.PatientID"):
        SQLValidateTool(settings).invoke(
            {
                "database": "dicom",
                "query": 'SELECT COUNT(*) FROM flathr."RTSTRUCT" s WHERE s."PatientID" IS NOT NULL',
            }
        )
