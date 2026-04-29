from __future__ import annotations

import json
from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan
from aor_runtime.runtime.engine import ExecutionEngine
from aor_runtime.runtime.policies import PlanContractViolation


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "examples" / "general_purpose_assistant.yaml"


def _engine(tmp_path: Path) -> ExecutionEngine:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    return ExecutionEngine(settings)


def _session(engine: ExecutionEngine, task: str):
    session_meta = engine.create_session(str(SPEC_PATH), {"task": task}, trigger="manual")
    session = engine.session_manager.get_session(session_meta["id"])
    assert session is not None
    return session


def _write_plan() -> ExecutionPlan:
    return ExecutionPlan.model_validate(
        {
            "steps": [
                {"id": 1, "action": "fs.write", "args": {"path": "notes.txt", "content": "hello"}},
                {"id": 2, "action": "fs.read", "args": {"path": "notes.txt"}},
            ]
        }
    )


def test_decorate_final_output_reformats_raw_json_in_user_mode(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    decorated = engine._decorate_final_output(
        {
            "goal": "status of the slurm cluster",
            "history": [
                {
                    "step": {"id": 1, "action": "slurm.metrics", "args": {"metric_group": "cluster_summary"}, "output": "cluster"},
                    "result": {
                        "metric_group": "cluster_summary",
                        "payload": {"queue_count": 10, "running_jobs": 2, "pending_jobs": 8},
                    },
                    "success": True,
                }
            ],
            "planning_metadata": {},
        },
        {
            "content": json.dumps({"metric_group": "cluster_summary", "payload": {"queue_count": 10, "running_jobs": 2, "pending_jobs": 8}}),
            "artifacts": [],
            "metadata": {},
        },
        status="completed",
        metrics={"llm_calls": 0},
    )

    assert not decorated["content"].lstrip().startswith("{")
    assert "SLURM" in decorated["content"]
    assert "queue" in decorated["content"].lower()


def test_successful_planner_run_persists_policies_and_logs_event(tmp_path: Path, monkeypatch) -> None:
    engine = _engine(tmp_path)
    session = _session(engine, "Create notes.txt with hello")

    def fake_build_plan(**kwargs):
        engine.planner.last_policies_used = ["filesystem_preference", "efficiency"]
        engine.planner.last_high_level_plan = None
        engine.planner.last_planning_mode = "direct"
        engine.planner.last_llm_calls = 1
        engine.planner.last_error_stage = None
        return _write_plan()

    monkeypatch.setattr(engine.planner, "build_plan", fake_build_plan)

    engine._run_planner(session)

    assert session.state["policies_used"] == ["filesystem_preference", "efficiency"]
    assert session.state["high_level_plan"] is None
    assert session.state["step_outputs"] == {}
    assert session.state["metrics"]["llm_calls"] == 1
    assert session.state["plan_canonicalized"] is False
    assert session.state["plan_repairs"] == []
    events = engine.store.get_events(session.id)
    planner_started = next(event for event in events if event["event_type"] == "planner.started")
    planner_completed = next(event for event in events if event["event_type"] == "planner.completed")
    assert planner_started["payload"]["planning_mode"] == "direct"
    assert planner_completed["payload"]["policies"] == ["filesystem_preference", "efficiency"]
    assert planner_completed["payload"]["planning_mode"] == "direct"
    assert planner_completed["payload"]["high_level_plan"] is None
    assert planner_completed["payload"]["goal"] == "Create notes.txt with hello"
    assert planner_completed["payload"]["execution_plan"]["steps"] == session.state["plan"]["steps"]
    assert planner_completed["payload"]["canonicalization_changed"] is False
    assert planner_completed["payload"]["repair_trace"] == []
    assert planner_completed["payload"]["original_execution_plan"] is None
    assert "steps" in planner_completed["payload"]


def test_hierarchical_planner_run_persists_high_level_plan_and_metrics(tmp_path: Path, monkeypatch) -> None:
    engine = _engine(tmp_path)
    session = _session(engine, "Query patients and save the top 10 to a file")

    def fake_build_plan(**kwargs):
        engine.planner.last_policies_used = ["sql_preference", "filesystem_preference", "efficiency"]
        engine.planner.last_high_level_plan = ["query patients", "save the top 10 to a file"]
        engine.planner.last_planning_mode = "hierarchical"
        engine.planner.last_llm_calls = 2
        engine.planner.last_error_stage = None
        return ExecutionPlan.model_validate(
            {
                "steps": [
                    {"id": 1, "action": "sql.query", "args": {"database": "clinical_db", "query": "SELECT 1"}},
                    {"id": 2, "action": "fs.write", "args": {"path": "patients.txt", "content": "ok"}},
                ]
            }
        )

    monkeypatch.setattr(engine.planner, "build_plan", fake_build_plan)

    engine._run_planner(session)

    assert session.state["high_level_plan"] == ["query patients", "save the top 10 to a file"]
    assert session.state["metrics"]["llm_calls"] == 2
    assert session.state["plan_canonicalized"] is False
    events = engine.store.get_events(session.id)
    planner_started = next(event for event in events if event["event_type"] == "planner.started")
    planner_completed = next(event for event in events if event["event_type"] == "planner.completed")
    assert planner_started["payload"]["planning_mode"] == "hierarchical"
    assert planner_completed["payload"]["planning_mode"] == "hierarchical"
    assert planner_completed["payload"]["high_level_plan"] == ["query patients", "save the top 10 to a file"]


def test_successful_planner_run_logs_canonicalization_trace(tmp_path: Path, monkeypatch) -> None:
    engine = _engine(tmp_path)
    session = _session(engine, "Query patients and save them to patients.csv")
    canonical_plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "sql.query",
                    "args": {"database": "clinical_db", "query": "SELECT name FROM patients"},
                    "output": "step_1_rows",
                },
                {
                    "id": 2,
                    "action": "fs.write",
                    "input": ["step_1_rows"],
                    "args": {"path": "patients.csv", "content": "{\"ok\": true}"},
                },
            ]
        }
    )

    def fake_build_plan(**kwargs):
        engine.planner.last_policies_used = ["sql_preference", "filesystem_preference", "efficiency"]
        engine.planner.last_high_level_plan = ["query patients", "save them"]
        engine.planner.last_planning_mode = "hierarchical"
        engine.planner.last_llm_calls = 2
        engine.planner.last_error_stage = None
        engine.planner.last_original_execution_plan = {
            "steps": [
                {
                    "id": 1,
                    "action": "sql.query",
                    "args": {"database": "clinical_db", "query": "SELECT name FROM patients"},
                    "output": "rows",
                },
                {
                    "id": 2,
                    "action": "fs.write",
                    "input": ["csv_result"],
                    "args": {"path": "patients.csv", "content": {"$ref": "csv", "path": "csv"}},
                },
            ]
        }
        engine.planner.last_canonicalized_execution_plan = canonical_plan.model_dump()
        engine.planner.last_plan_repairs = ["output:rows->step_1_rows", "input:synthesized_from_refs"]
        engine.planner.last_plan_canonicalized = True
        return canonical_plan

    monkeypatch.setattr(engine.planner, "build_plan", fake_build_plan)

    engine._run_planner(session)

    assert session.state["plan_canonicalized"] is True
    assert session.state["plan_repairs"] == ["output:rows->step_1_rows", "input:synthesized_from_refs"]
    events = engine.store.get_events(session.id)
    planner_completed = next(event for event in events if event["event_type"] == "planner.completed")
    assert planner_completed["payload"]["canonicalization_changed"] is True
    assert planner_completed["payload"]["repair_trace"] == ["output:rows->step_1_rows", "input:synthesized_from_refs"]
    assert planner_completed["payload"]["original_execution_plan"]["steps"][0]["output"] == "rows"


def test_successful_planner_run_persists_llm_intent_metadata(tmp_path: Path, monkeypatch) -> None:
    engine = _engine(tmp_path)
    session = _session(engine, "Is the cluster busy right now?")

    def fake_build_plan(**kwargs):
        engine.planner.last_policies_used = ["efficiency", "llm_intent_extractor"]
        engine.planner.last_high_level_plan = None
        engine.planner.last_planning_mode = "llm_intent_extractor"
        engine.planner.last_llm_calls = 1
        engine.planner.last_llm_intent_calls = 1
        engine.planner.last_raw_planner_llm_calls = 0
        engine.planner.last_llm_intent_type = "SlurmMetricsIntent"
        engine.planner.last_llm_intent_confidence = 0.86
        engine.planner.last_llm_intent_reason = "Broad cluster summary."
        engine.planner.last_capability_name = "slurm"
        engine.planner.last_error_stage = None
        return ExecutionPlan.model_validate(
            {
                "steps": [
                    {"id": 1, "action": "slurm.metrics", "args": {"metric_group": "cluster_summary"}},
                    {"id": 2, "action": "runtime.return", "args": {"value": {"queue_count": 4}, "mode": "json"}},
                ]
            }
        )

    monkeypatch.setattr(engine.planner, "build_plan", fake_build_plan)

    engine._run_planner(session)

    assert session.state["metrics"]["llm_calls"] == 1
    assert session.state["metrics"]["llm_intent_calls"] == 1
    assert session.state["metrics"]["raw_planner_llm_calls"] == 0
    assert session.state["planning_metadata"]["planning_mode"] == "llm_intent_extractor"
    assert session.state["planning_metadata"]["llm_intent_type"] == "SlurmMetricsIntent"
    events = engine.store.get_events(session.id)
    planner_completed = next(event for event in events if event["event_type"] == "planner.completed")
    assert planner_completed["payload"]["planning_mode"] == "llm_intent_extractor"
    assert planner_completed["payload"]["capability"] == "slurm"
    assert planner_completed["payload"]["llm_intent_type"] == "SlurmMetricsIntent"
    assert planner_completed["payload"]["llm_intent_calls"] == 1
    assert planner_completed["payload"]["raw_planner_llm_calls"] == 0


def test_planner_failure_clears_stale_policies_used_and_logs_stage(tmp_path: Path, monkeypatch) -> None:
    engine = _engine(tmp_path)
    session = _session(engine, "Find all txt files in this folder and provide list as csv")
    engine.planner.last_policies_used = ["stale_policy"]

    def failing_build_plan(**kwargs):
        engine.planner.last_policies_used = ["efficiency"]
        engine.planner.last_high_level_plan = ["find files", "format as csv"]
        engine.planner.last_planning_mode = "hierarchical"
        engine.planner.last_llm_calls = 2
        engine.planner.last_error_stage = "refine"
        engine.planner.last_raw_output = '{"steps":[{"id":1,"action":"fs.write","args":{"content":"hello" + "world"}}]}'
        engine.planner.last_error_type = "JSONDecodeError"
        raise RuntimeError("planner boom")

    monkeypatch.setattr(engine.planner, "build_plan", failing_build_plan)

    engine._run_planner(session)

    assert session.state["status"] == "failed"
    assert session.state["policies_used"] == []
    assert session.state["high_level_plan"] is None
    assert session.state["metrics"]["llm_calls"] == 2
    metadata = session.state["final_output"]["metadata"]
    assert metadata["planner_error_type"] == "JSONDecodeError"
    assert '"content":"hello" + "world"' in metadata["planner_raw_output_preview"]
    events = engine.store.get_events(session.id)
    planner_failed = next(event for event in events if event["event_type"] == "planner.failed")
    assert planner_failed["payload"]["error_type"] == "JSONDecodeError"
    assert planner_failed["payload"]["stage"] == "refine"
    assert planner_failed["payload"]["planning_mode"] == "hierarchical"
    assert planner_failed["payload"]["policies"] == ["efficiency"]
    assert '"content":"hello" + "world"' in planner_failed["payload"]["raw_output_preview"]


def test_decomposition_failure_logs_decompose_stage(tmp_path: Path, monkeypatch) -> None:
    engine = _engine(tmp_path)
    session = _session(engine, "Query patients and then save results after filtering the top 10 rows")

    def failing_build_plan(**kwargs):
        engine.planner.last_policies_used = ["sql_preference", "efficiency"]
        engine.planner.last_high_level_plan = None
        engine.planner.last_planning_mode = "hierarchical"
        engine.planner.last_llm_calls = 1
        engine.planner.last_error_stage = "decompose"
        engine.planner.last_raw_output = '{"tasks":[]}'
        engine.planner.last_error_type = "ValidationError"
        raise RuntimeError("decomposer boom")

    monkeypatch.setattr(engine.planner, "build_plan", failing_build_plan)

    engine._run_planner(session)

    events = engine.store.get_events(session.id)
    planner_failed = next(event for event in events if event["event_type"] == "planner.failed")
    assert planner_failed["payload"]["stage"] == "decompose"
    assert planner_failed["payload"]["planning_mode"] == "hierarchical"
    assert session.state["high_level_plan"] is None
    assert session.state["metrics"]["llm_calls"] == 1


def test_planner_connection_failure_is_normalized_to_llm_error(tmp_path: Path, monkeypatch) -> None:
    engine = _engine(tmp_path)
    session = _session(engine, "Count patients in dicom")

    def failing_build_plan(**kwargs):
        engine.planner.last_policies_used = ["sql_preference"]
        engine.planner.last_high_level_plan = None
        engine.planner.last_planning_mode = "direct"
        engine.planner.last_llm_calls = 1
        engine.planner.last_error_stage = "direct"
        engine.planner.last_raw_output = None
        engine.planner.last_error_type = "APIConnectionError"
        raise RuntimeError("Connection error.")

    monkeypatch.setattr(engine.planner, "build_plan", failing_build_plan)

    engine._run_planner(session)

    assert session.state["final_output"]["content"].startswith("LLM connection error.")
    assert "Suggested prompts:" not in session.state["final_output"]["content"]
    metadata = session.state["final_output"]["metadata"]
    assert metadata["planner_error_type"] == "APIConnectionError"
    assert metadata["error_source"] == "llm"
    assert metadata["error_kind"] == "connection"
    assert metadata["error_target"] == engine.settings.llm_base_url
    assert metadata["error_detail"] == "Connection error."
    assert metadata["suggestion_count"] >= 1
    events = engine.store.get_events(session.id)
    planner_failed = next(event for event in events if event["event_type"] == "planner.failed")
    assert planner_failed["payload"]["error"] == "LLM connection error."
    assert planner_failed["payload"]["error_source"] == "llm"
    assert planner_failed["payload"]["error_kind"] == "connection"


def test_planner_timeout_failure_is_normalized_to_llm_timeout(tmp_path: Path, monkeypatch) -> None:
    engine = _engine(tmp_path)
    session = _session(engine, "Count patients in dicom after checking connectivity")

    def failing_build_plan(**kwargs):
        engine.planner.last_policies_used = ["sql_preference"]
        engine.planner.last_high_level_plan = None
        engine.planner.last_planning_mode = "hierarchical"
        engine.planner.last_llm_calls = 2
        engine.planner.last_error_stage = "refine"
        engine.planner.last_raw_output = None
        engine.planner.last_error_type = "APITimeoutError"
        raise RuntimeError("Request timed out.")

    monkeypatch.setattr(engine.planner, "build_plan", failing_build_plan)

    engine._run_planner(session)

    assert session.state["final_output"]["content"].startswith("LLM timeout error.")
    assert "Suggested prompts:" not in session.state["final_output"]["content"]
    metadata = session.state["final_output"]["metadata"]
    assert metadata["error_source"] == "llm"
    assert metadata["error_kind"] == "timeout"
    assert metadata["suggestion_count"] >= 1


def test_terminal_sql_auth_failure_is_normalized_and_redacted(tmp_path: Path) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        sql_databases={"dicom": "postgresql+psycopg://admin:admin@localhost:5432/dicom"},
        sql_default_database="dicom",
    )
    engine = ExecutionEngine(settings)
    session = _session(engine, "Count patients in dicom")
    session.state["retries"] = 2

    engine._handle_retry_or_failure(
        session,
        node_name="executor",
        reason="tool_execution_failed",
        detail="(psycopg.OperationalError) postgresql+psycopg://admin:admin@localhost:5432/dicom fe_sendauth: no password supplied",
        extra_context={
            "step": {"id": 1, "action": "sql.query", "args": {"database": "dicom", "query": "SELECT COUNT(*) FROM patient"}},
            "history": [],
        },
    )

    assert session.state["final_output"]["content"].startswith("Database authentication error for 'dicom'.")
    assert "Suggested prompts:" not in session.state["final_output"]["content"]
    metadata = session.state["final_output"]["metadata"]
    assert metadata["error_source"] == "sql"
    assert metadata["error_kind"] == "authentication"
    assert metadata["error_target"] == "dicom"
    assert "***:***@" in metadata["error_detail"]
    assert "admin:admin@" not in metadata["error_detail"]
    assert metadata["failure_type"] == "execution_failure"
    events = engine.store.get_events(session.id)
    executor_failed = next(event for event in events if event["event_type"] == "executor.failed")
    assert executor_failed["payload"]["error"] == "Database authentication error for 'dicom'."
    assert executor_failed["payload"]["error_source"] == "sql"


def test_terminal_sql_connection_failure_is_normalized(tmp_path: Path) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        sql_databases={"dicom": "postgresql+psycopg://admin:admin@localhost:5432/dicom"},
        sql_default_database="dicom",
    )
    engine = ExecutionEngine(settings)
    session = _session(engine, "Count patients in dicom")
    session.state["retries"] = 2

    engine._handle_retry_or_failure(
        session,
        node_name="executor",
        reason="tool_execution_failed",
        detail="(psycopg.OperationalError) connection refused",
        extra_context={
            "step": {"id": 1, "action": "sql.query", "args": {"database": "dicom", "query": "SELECT COUNT(*) FROM patient"}},
            "history": [],
        },
    )

    assert session.state["final_output"]["content"].startswith("Database connection error for 'dicom'.")
    assert "Suggested prompts:" not in session.state["final_output"]["content"]
    metadata = session.state["final_output"]["metadata"]
    assert metadata["error_source"] == "sql"
    assert metadata["error_kind"] == "connection"


def test_terminal_gateway_request_failure_is_normalized(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    session = _session(engine, "Using shell, list files")
    session.state["retries"] = 2

    engine._handle_retry_or_failure(
        session,
        node_name="executor",
        reason="tool_execution_failed",
        detail="Gateway request failed: HTTPConnectionPool(host='127.0.0.1', port=8787): Max retries exceeded with url: /exec (Caused by NewConnectionError('connection refused'))",
        extra_context={
            "step": {"id": 1, "action": "shell.exec", "args": {"node": "local", "command": "ls"}},
            "history": [],
        },
    )

    assert session.state["final_output"]["content"].startswith("Gateway connection error for node 'local'.")
    assert "Suggested prompts:" not in session.state["final_output"]["content"]
    metadata = session.state["final_output"]["metadata"]
    assert metadata["error_source"] == "gateway"
    assert metadata["error_kind"] == "connection"
    assert metadata["error_target"] == "local"


def test_terminal_gateway_response_failure_is_normalized(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    session = _session(engine, "Using shell, list files")
    session.state["retries"] = 2

    engine._handle_retry_or_failure(
        session,
        node_name="executor",
        reason="tool_execution_failed",
        detail="Gateway response validation failed: 1 validation error for GatewayExecResult",
        extra_context={
            "step": {"id": 1, "action": "shell.exec", "args": {"node": "local", "command": "ls"}},
            "history": [],
        },
    )

    assert session.state["final_output"]["content"].startswith("Gateway response error for node 'local'.")
    assert "Suggested prompts:" not in session.state["final_output"]["content"]
    metadata = session.state["final_output"]["metadata"]
    assert metadata["error_source"] == "gateway"
    assert metadata["error_kind"] == "response"


def test_unclassified_sql_error_message_is_preserved(tmp_path: Path) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        sql_databases={"dicom": "postgresql+psycopg://admin:admin@localhost:5432/dicom"},
        sql_default_database="dicom",
    )
    engine = ExecutionEngine(settings)
    session = _session(engine, "Count patients in dicom")
    session.state["retries"] = 2

    engine._handle_retry_or_failure(
        session,
        node_name="executor",
        reason="tool_execution_failed",
        detail="Unsafe query",
        extra_context={
            "step": {"id": 1, "action": "sql.query", "args": {"database": "dicom", "query": "DELETE FROM patient"}},
            "history": [],
        },
    )

    assert session.state["final_output"]["content"].startswith("Unsafe query")
    assert "Suggested prompts:" not in session.state["final_output"]["content"]
    metadata = session.state["final_output"]["metadata"]
    assert "error_source" not in metadata
    assert metadata["failure_type"] == "execution_failure"


def test_unsafe_query_failure_is_non_retryable(tmp_path: Path) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        sql_databases={"dicom": "postgresql+psycopg://admin:admin@localhost:5432/dicom"},
        sql_default_database="dicom",
    )
    engine = ExecutionEngine(settings)
    session = _session(engine, "Delete all patients in dicom")

    engine._handle_retry_or_failure(
        session,
        node_name="executor",
        reason="tool_execution_failed",
        detail="Unsafe query",
        extra_context={
            "step": {"id": 1, "action": "sql.query", "args": {"database": "dicom", "query": "DELETE FROM patient"}},
            "history": [],
        },
    )

    assert session.state["status"] == "failed"
    assert session.state["final_output"]["content"].startswith("Unsafe query")
    assert "Suggested prompts:" not in session.state["final_output"]["content"]
    assert session.state["failure_context"]["retryable"] is False
    assert session.state["final_output"]["metadata"]["retryable"] is False
    assert session.state["final_output"]["metadata"]["failure_type"] == "unsupported_mutating_operation"


def test_planner_contract_failure_metadata_is_preserved(tmp_path: Path, monkeypatch) -> None:
    engine = _engine(tmp_path)
    session = _session(engine, "Delete all patients in dicom")

    def failing_build_plan(**kwargs):
        engine.planner.last_policies_used = ["sql_preference"]
        engine.planner.last_high_level_plan = None
        engine.planner.last_planning_mode = "direct"
        engine.planner.last_llm_calls = 1
        engine.planner.last_error_stage = "direct"
        raise PlanContractViolation(
            "Unsafe query",
            tier="hard",
            code="unsafe_sql",
            violations=[],
        )

    monkeypatch.setattr(engine.planner, "build_plan", failing_build_plan)

    engine._run_planner(session)

    metadata = session.state["final_output"]["metadata"]
    assert metadata["contract_violation"] is True
    assert metadata["violation_tier"] == "hard"
    assert metadata["violation_code"] == "unsafe_sql"
    assert metadata["suggestion_count"] >= 1
    events = engine.store.get_events(session.id)
    planner_failed = next(event for event in events if event["event_type"] == "planner.failed")
    assert planner_failed["payload"]["contract_violation"] is True
    assert planner_failed["payload"]["violation_tier"] == "hard"


def test_retry_and_terminal_failure_clear_policies_used(tmp_path: Path) -> None:
    engine = _engine(tmp_path)

    retry_session = _session(engine, "Retry path")
    retry_session.state["policies_used"] = ["sql_preference"]
    retry_session.state["high_level_plan"] = ["query data"]
    retry_session.state["step_outputs"] = {"query_data": {"rows": []}}
    engine._handle_retry_or_failure(
        retry_session,
        node_name="executor",
        reason="tool_execution_failed",
        detail="boom",
        extra_context={},
    )
    assert retry_session.state["status"] == "retrying"
    assert retry_session.state["policies_used"] == []
    assert retry_session.state["high_level_plan"] is None
    assert retry_session.state["step_outputs"] == {}

    terminal_session = _session(engine, "Terminal failure path")
    terminal_session.state["policies_used"] = ["filesystem_preference"]
    terminal_session.state["high_level_plan"] = ["write file"]
    terminal_session.state["step_outputs"] = {"write_data": {"content": "ok"}}
    terminal_session.state["retries"] = 2
    engine._handle_retry_or_failure(
        terminal_session,
        node_name="executor",
        reason="tool_execution_failed",
        detail="boom",
        extra_context={},
    )
    assert terminal_session.state["status"] == "failed"
    assert terminal_session.state["policies_used"] == []
    assert terminal_session.state["high_level_plan"] is None
    assert terminal_session.state["step_outputs"] == {}


def test_retry_failure_context_is_summarized_and_size_capped(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    session = _session(engine, "Retry with compact history")

    huge_history = []
    for index in range(5):
        huge_history.append(
            {
                "step": {"id": index + 1, "action": "sql.query", "args": {"database": "dicom", "query": "SELECT * FROM patient"}},
                "success": index < 4,
                "result": {"rows": [{"patient_id": row, "name": "x" * 200} for row in range(200)], "row_count": 200},
                "error": "boom" if index == 4 else None,
            }
        )

    engine._handle_retry_or_failure(
        session,
        node_name="executor",
        reason="tool_execution_failed",
        detail="write failed",
        extra_context={
            "step": {"id": 6, "action": "fs.write", "args": {"path": "reports/output.json", "content": {"rows": "too big"}}},
            "history": huge_history,
        },
    )

    failure_context = session.state["failure_context"]
    assert session.state["status"] == "retrying"
    assert "history" not in failure_context
    assert failure_context["failed_step"] == "fs.write"
    assert len(failure_context["summary"]) <= 3
    assert failure_context["summary"][-1]["action"] == "sql.query"
    assert json.dumps(failure_context, ensure_ascii=False, sort_keys=True)
    assert len(json.dumps(failure_context, ensure_ascii=False, sort_keys=True)) <= 4096
    events = engine.store.get_events(session.id)
    executor_failed = next(event for event in events if event["event_type"] == "executor.failed")
    assert "history" not in executor_failed["payload"]
