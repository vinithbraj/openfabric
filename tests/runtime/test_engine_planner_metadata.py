from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan
from aor_runtime.runtime.engine import ExecutionEngine


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


def test_successful_planner_run_persists_policies_and_logs_event(tmp_path: Path, monkeypatch) -> None:
    engine = _engine(tmp_path)
    session = _session(engine, "Create notes.txt with hello")

    def fake_build_plan(**kwargs):
        engine.planner.last_policies_used = ["filesystem_preference", "efficiency"]
        return _write_plan()

    monkeypatch.setattr(engine.planner, "build_plan", fake_build_plan)

    engine._run_planner(session)

    assert session.state["policies_used"] == ["filesystem_preference", "efficiency"]
    events = engine.store.get_events(session.id)
    planner_completed = next(event for event in events if event["event_type"] == "planner.completed")
    assert planner_completed["payload"]["policies"] == ["filesystem_preference", "efficiency"]
    assert "steps" in planner_completed["payload"]


def test_planner_failure_clears_stale_policies_used(tmp_path: Path, monkeypatch) -> None:
    engine = _engine(tmp_path)
    session = _session(engine, "This planner run should fail")
    engine.planner.last_policies_used = ["stale_policy"]

    def failing_build_plan(**kwargs):
        engine.planner.last_policies_used = ["efficiency"]
        engine.planner.last_raw_output = '{"steps":[{"id":1,"action":"fs.write","args":{"content":"hello" + "world"}}]}'
        engine.planner.last_error_type = "JSONDecodeError"
        raise RuntimeError("planner boom")

    monkeypatch.setattr(engine.planner, "build_plan", failing_build_plan)

    engine._run_planner(session)

    assert session.state["status"] == "failed"
    assert session.state["policies_used"] == []
    metadata = session.state["final_output"]["metadata"]
    assert metadata["planner_error_type"] == "JSONDecodeError"
    assert '"content":"hello" + "world"' in metadata["planner_raw_output_preview"]
    events = engine.store.get_events(session.id)
    planner_failed = next(event for event in events if event["event_type"] == "planner.failed")
    assert planner_failed["payload"]["error_type"] == "JSONDecodeError"
    assert planner_failed["payload"]["policies"] == ["efficiency"]
    assert '"content":"hello" + "world"' in planner_failed["payload"]["raw_output_preview"]


def test_retry_and_terminal_failure_clear_policies_used(tmp_path: Path) -> None:
    engine = _engine(tmp_path)

    retry_session = _session(engine, "Retry path")
    retry_session.state["policies_used"] = ["sql_preference"]
    engine._handle_retry_or_failure(
        retry_session,
        node_name="executor",
        reason="tool_execution_failed",
        detail="boom",
        extra_context={},
    )
    assert retry_session.state["status"] == "retrying"
    assert retry_session.state["policies_used"] == []

    terminal_session = _session(engine, "Terminal failure path")
    terminal_session.state["policies_used"] = ["filesystem_preference"]
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
