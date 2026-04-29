from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan, ExecutionStep, StepLog
from aor_runtime.runtime.engine import ExecutionEngine
from aor_runtime.runtime.executor import PlanExecutor
from aor_runtime.runtime.validator import RuntimeValidator
from aor_runtime.tools.factory import build_tool_registry
from aor_runtime.tools.gateway import GatewayExecResult


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "examples" / "general_purpose_assistant.yaml"


def _engine(tmp_path: Path) -> ExecutionEngine:
    settings = Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        available_nodes_raw="local",
        default_node="local",
        shell_mode="permissive",
    )
    return ExecutionEngine(settings)


def _session(engine: ExecutionEngine, task: str):
    session_meta = engine.create_session(str(SPEC_PATH), {"task": task}, trigger="manual")
    session = engine.session_manager.get_session(session_meta["id"])
    assert session is not None
    return session


def test_executor_fs_not_exists_succeeds_when_path_is_absent(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    executor = PlanExecutor(build_tool_registry(settings))
    step = ExecutionStep(id=1, action="fs.not_exists", args={"path": "missing.txt"})

    log = executor.execute_step(step)

    assert log.success is True
    assert log.result["exists"] is False


def test_executor_fs_not_exists_fails_when_path_exists(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    (tmp_path / "present.txt").write_text("hello")
    executor = PlanExecutor(build_tool_registry(settings))
    step = ExecutionStep(id=1, action="fs.not_exists", args={"path": "present.txt"})

    log = executor.execute_step(step)

    assert log.success is False
    assert log.error == "Path still exists: present.txt"


def test_validator_validates_positive_and_negative_existence(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    tools = build_tool_registry(settings)
    validator = RuntimeValidator(settings)
    (tmp_path / "present.txt").write_text("hello")

    exists_step = ExecutionStep(id=1, action="fs.exists", args={"path": "present.txt"})
    not_exists_step = ExecutionStep(id=2, action="fs.not_exists", args={"path": "missing.txt"})
    logs = [
        StepLog(step=exists_step, result=tools.invoke("fs.exists", {"path": "present.txt"}), success=True),
        StepLog(step=not_exists_step, result=tools.invoke("fs.not_exists", {"path": "missing.txt"}), success=True),
    ]

    validation, checks = validator.validate(logs, goal="check file presence and absence")

    assert validation.success is True
    assert all(bool(check["success"]) for check in checks)


def test_validator_fs_exists_uses_recorded_result_for_preconditions(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    tools = build_tool_registry(settings)
    validator = RuntimeValidator(settings)
    target = tmp_path / "present.txt"
    target.write_text("hello")
    step = ExecutionStep(id=1, action="fs.exists", args={"path": "present.txt"})
    log = StepLog(step=step, result=tools.invoke("fs.exists", {"path": "present.txt"}), success=True)
    target.unlink()

    validation, checks = validator.validate([log], goal="delete present.txt")

    assert validation.success is True
    assert checks[0]["name"] == "step_1_fs.exists"


def test_validator_fs_not_exists_fails_when_path_exists(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    tools = build_tool_registry(settings)
    validator = RuntimeValidator(settings)
    (tmp_path / "present.txt").write_text("hello")
    step = ExecutionStep(id=1, action="fs.not_exists", args={"path": "present.txt"})
    log = StepLog(step=step, result=tools.invoke("fs.not_exists", {"path": "present.txt"}), success=True)

    validation, checks = validator.validate([log], goal="ensure file was deleted")

    assert validation.success is False
    assert checks[0]["name"] == "step_1_fs.not_exists"


def test_delete_flow_with_fs_not_exists_completes_without_retry(tmp_path: Path, monkeypatch) -> None:
    engine = _engine(tmp_path)
    target = tmp_path / "clear.txt"
    target.write_text("delete me")
    session = _session(engine, "remove clear.txt")

    def fake_execute_gateway_command(settings, *, node: str, command: str):
        if command == "rm clear.txt":
            target.unlink()
        return GatewayExecResult(stdout="", stderr="", exit_code=0)

    def fake_build_plan(**kwargs):
        engine.planner.last_policies_used = ["filesystem_preference", "efficiency"]
        return ExecutionPlan.model_validate(
            {
                "steps": [
                    {"id": 1, "action": "fs.exists", "args": {"path": "clear.txt"}},
                    {"id": 2, "action": "shell.exec", "args": {"command": "rm clear.txt"}},
                    {"id": 3, "action": "fs.not_exists", "args": {"path": "clear.txt"}},
                ]
            }
        )

    monkeypatch.setattr(engine.planner, "build_plan", fake_build_plan)
    monkeypatch.setattr("aor_runtime.tools.shell.execute_gateway_command", fake_execute_gateway_command)

    paused = engine.resume_session(session.id, trigger="manual")
    assert paused["awaiting_confirmation"] is True
    assert paused["confirmation_kind"] == "dangerous_step"

    final_state = engine.resume_session(session.id, trigger="manual", approve_dangerous=True)

    assert final_state["status"] == "completed"
    assert final_state["retries"] == 0
    assert final_state["validation"]["success"] is True
    assert final_state["final_output"]["content"].startswith("true")
    assert target.exists() is False

    payload = engine.get_session(session.id)
    assert payload is not None
    events = payload["events"]
    assert not any(event["event_type"] == "validator.failed" for event in events)
    assert not any(event["event_type"] == "planner.failed" for event in events)
