from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from aor_runtime.api.app import create_app
from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan


def _write_sql_spec(tmp_path: Path) -> Path:
    spec_path = tmp_path / "openwebui_sql.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "version: 1",
                "name: openwebui_sql",
                "runtime:",
                "  max_retries: 0",
                "tools:",
                "  - sql.query",
            ]
        )
    )
    return spec_path


def _settings(tmp_path: Path, spec_path: Path, db_path: Path) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        sql_databases={"dicom": f"sqlite:///{db_path}"},
        sql_default_database="dicom",
        openai_compat_enabled=True,
        openai_compat_model_name="openfabric-agent",
        openai_compat_spec_path=str(spec_path),
    )


def _create_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE patients(id INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO patients(id) VALUES (?)", [(1,), (2,)])


def _patch_sql_plan(engine, monkeypatch) -> None:
    def fake_build_plan(**kwargs):
        engine.planner.last_policies_used = ["sql_preference"]
        engine.planner.last_high_level_plan = None
        engine.planner.last_planning_mode = "deterministic_intent"
        engine.planner.last_llm_calls = 0
        engine.planner.last_llm_intent_calls = 0
        engine.planner.last_raw_planner_llm_calls = 0
        engine.planner.last_error_stage = None
        engine.planner.last_plan_repairs = []
        engine.planner.last_plan_canonicalized = False
        engine.planner.last_original_execution_plan = None
        engine.planner.last_canonicalized_execution_plan = None
        engine.planner.last_capability_metadata = {"sql_final": "SELECT COUNT(*) AS count_value FROM patients;"}
        return ExecutionPlan.model_validate(
            {
                "steps": [
                    {
                        "id": 1,
                        "action": "sql.query",
                        "args": {"database": "dicom", "query": "SELECT COUNT(*) AS count_value FROM patients;"},
                        "output": "sql_result",
                    },
                    {
                        "id": 2,
                        "action": "runtime.return",
                        "input": ["sql_result"],
                        "args": {"value": {"$ref": "sql_result", "path": "rows.0.count_value"}, "mode": "count"},
                    },
                ]
            }
        )

    monkeypatch.setattr(engine.planner, "build_plan", fake_build_plan)


def test_openwebui_non_streaming_response_is_polished_markdown(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "dicom.db"
    _create_db(db_path)
    spec_path = _write_sql_spec(tmp_path)
    app = create_app(_settings(tmp_path, spec_path, db_path))
    _patch_sql_plan(app.state.engine, monkeypatch)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "openfabric-agent", "messages": [{"role": "user", "content": "list the count of all patients in dicom"}]},
    )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert "Thinking..." not in content
    assert "Plan ready" not in content
    assert "Validating" not in content
    assert "Validation passed" not in content
    assert "## Result" in content
    assert "Count: 2" in content
    assert "## Query Used" in content
    assert "SELECT COUNT(*) AS count_value FROM patients" in content
    assert not content.lstrip().startswith("{")


def test_openwebui_streaming_hides_lifecycle_noise_and_streams_final_markdown(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "dicom.db"
    _create_db(db_path)
    spec_path = _write_sql_spec(tmp_path)
    app = create_app(_settings(tmp_path, spec_path, db_path))
    _patch_sql_plan(app.state.engine, monkeypatch)
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "openfabric-agent",
            "messages": [{"role": "user", "content": "list the count of all patients in dicom"}],
            "stream": True,
        },
    ) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert "Thinking..." not in body
    assert "Plan ready" not in body
    assert "Validating" not in body
    assert "Validation passed" not in body
    assert "Query Used" in body
    assert "SELECT COUNT(*) AS count_value FROM patients" in body
    assert "data: [DONE]" in body
