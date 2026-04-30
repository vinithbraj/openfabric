from __future__ import annotations

import sqlite3
from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.runtime.lifecycle import ActiveRunRegistry, CancellationToken, ToolInvocationContext
from aor_runtime.tools.sql import SQLValidateTool, sql_query


def _settings_for_sqlite(tmp_path: Path, *, sql_row_limit: int = 1) -> Settings:
    database_path = tmp_path / "dicom.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT)")
        connection.executemany(
            "INSERT INTO patients (id, name) VALUES (?, ?)",
            [(index, f"patient-{index}") for index in range(75)],
        )
        connection.commit()
    return Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        sql_databases={"dicom": f"sqlite:///{database_path}"},
        sql_default_database="dicom",
        sql_row_limit=sql_row_limit,
    )


def test_sql_query_returns_all_rows_without_execution_cap(tmp_path: Path) -> None:
    settings = _settings_for_sqlite(tmp_path, sql_row_limit=1)

    result = sql_query(settings, "SELECT id, name FROM patients ORDER BY id", database="dicom")

    assert result["row_count"] == 75
    assert result["returned_count"] == 75
    assert len(result["rows"]) == 75
    assert result["rows"][0] == {"id": 0, "name": "patient-0"}
    assert result["rows"][-1] == {"id": 74, "name": "patient-74"}
    assert result["limit"] is None
    assert result["truncated"] is False


def test_sql_query_tool_unregisters_managed_worker_after_large_result(tmp_path: Path) -> None:
    settings = _settings_for_sqlite(tmp_path, sql_row_limit=1)
    registry = ActiveRunRegistry(shutdown_grace_seconds=0.5, process_kill_grace_seconds=0.1)
    context = ToolInvocationContext(
        cancellation=CancellationToken(),
        process_registry=registry,
        worker_join_timeout_seconds=0.5,
        tool_process_kill_grace_seconds=0.1,
    )

    from aor_runtime.tools.sql import SQLQueryTool

    result = SQLQueryTool(settings).invoke(
        {"database": "dicom", "query": "SELECT id, name FROM patients ORDER BY id"},
        context=context,
    )

    assert result["row_count"] == 75
    assert registry.active_process_count() == 0


def test_sql_validate_tool_validates_without_executing_query(tmp_path: Path) -> None:
    settings = _settings_for_sqlite(tmp_path)
    tool = SQLValidateTool(settings)

    result = tool.invoke({"database": "dicom", "query": "SELECT COUNT(*) AS count_value FROM patients"})

    assert result["database"] == "dicom"
    assert result["valid"] is True
    assert result["query"] == "SELECT COUNT(*) AS count_value FROM patients"
    assert "was not executed" in result["explanation"]
