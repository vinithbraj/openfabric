from __future__ import annotations

from pathlib import Path

import requests

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan, ExecutionStep, StepLog
from aor_runtime.runtime.engine import ExecutionEngine
from aor_runtime.runtime.executor import summarize_final_output
from aor_runtime.runtime.validator import RuntimeValidator
from aor_runtime.tools.base import ToolExecutionError
from aor_runtime.tools.shell import run_shell


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "examples" / "general_purpose_assistant.yaml"


class _DummyResponse:
    def __init__(self, payload, *, http_error: Exception | None = None) -> None:
        self.payload = payload
        self.http_error = http_error

    def raise_for_status(self) -> None:
        if self.http_error is not None:
            raise self.http_error

    def json(self):
        return self.payload


def _engine(tmp_path: Path) -> ExecutionEngine:
    settings = Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        gateway_url="https://gateway.internal/exec",
        available_nodes_raw="edge-1,local",
        default_node="local",
    )
    return ExecutionEngine(settings)


def test_shell_exec_posts_to_gateway_with_explicit_node(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        gateway_url="https://gateway.internal/exec",
        available_nodes_raw="edge-1,local",
        default_node="local",
    )
    observed: dict[str, object] = {}

    def fake_post(url, json, timeout):
        observed["url"] = url
        observed["json"] = json
        observed["timeout"] = timeout
        return _DummyResponse({"stdout": "ok\n", "stderr": "", "exit_code": 0})

    monkeypatch.setattr("aor_runtime.tools.gateway.requests.post", fake_post)

    result = run_shell(settings, "uname -a", node="edge-1")

    assert result["stdout"] == "ok\n"
    assert result["stderr"] == ""
    assert result["returncode"] == 0
    assert result["node"] == "edge-1"
    assert observed == {
        "url": "https://gateway.internal/exec",
        "json": {"node": "edge-1", "command": "uname -a"},
        "timeout": 30.0,
    }


def test_shell_exec_without_node_uses_default_node(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        gateway_url="https://gateway.internal/exec",
        available_nodes_raw="edge-1,local",
        default_node="local",
    )
    observed: dict[str, object] = {}

    def fake_post(url, json, timeout):
        observed["json"] = json
        return _DummyResponse({"stdout": "local\n", "stderr": "", "exit_code": 0})

    monkeypatch.setattr("aor_runtime.tools.gateway.requests.post", fake_post)

    result = run_shell(settings, "hostname")

    assert result["node"] == "local"
    assert observed["json"] == {"node": "local", "command": "hostname"}


def test_shell_exec_without_node_uses_implicit_localhost_default(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        gateway_url="https://gateway.internal/exec",
    )
    observed: dict[str, object] = {}

    def fake_post(url, json, timeout):
        observed["json"] = json
        return _DummyResponse({"stdout": "localhost\n", "stderr": "", "exit_code": 0})

    monkeypatch.setattr("aor_runtime.tools.gateway.requests.post", fake_post)

    result = run_shell(settings, "hostname")

    assert result["node"] == "localhost"
    assert observed["json"] == {"node": "localhost", "command": "hostname"}


def test_shell_exec_rejects_disallowed_node_before_http(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        gateway_url="https://gateway.internal/exec",
        available_nodes_raw="edge-1",
    )
    called = False

    def fake_post(url, json, timeout):
        nonlocal called
        called = True
        return _DummyResponse({"stdout": "", "stderr": "", "exit_code": 0})

    monkeypatch.setattr("aor_runtime.tools.gateway.requests.post", fake_post)

    try:
        run_shell(settings, "hostname", node="edge-9")
        raise AssertionError("Expected disallowed node to raise.")
    except ToolExecutionError as exc:
        assert "Node is not available" in str(exc)
    assert called is False


def test_shell_exec_wraps_non_2xx_as_tool_error(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        gateway_url="https://gateway.internal/exec",
        available_nodes_raw="edge-1",
        default_node="edge-1",
    )

    def fake_post(url, json, timeout):
        return _DummyResponse({}, http_error=requests.HTTPError("500 Server Error"))

    monkeypatch.setattr("aor_runtime.tools.gateway.requests.post", fake_post)

    try:
        run_shell(settings, "hostname")
        raise AssertionError("Expected HTTP error to raise.")
    except ToolExecutionError as exc:
        assert "Gateway request failed" in str(exc)


def test_shell_exec_wraps_invalid_response(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        gateway_url="https://gateway.internal/exec",
        available_nodes_raw="edge-1",
        default_node="edge-1",
    )

    def fake_post(url, json, timeout):
        return _DummyResponse({"stdout": "ok"})

    monkeypatch.setattr("aor_runtime.tools.gateway.requests.post", fake_post)

    try:
        run_shell(settings, "hostname")
        raise AssertionError("Expected invalid payload to raise.")
    except ToolExecutionError as exc:
        assert "Gateway response validation failed" in str(exc)


def test_shell_exec_final_output_uses_stdout() -> None:
    log = StepLog(
        step=ExecutionStep(id=1, action="shell.exec", args={"node": "edge-1", "command": "hostname"}),
        result={"command": "hostname", "node": "edge-1", "stdout": "edge-1\n", "stderr": "", "returncode": 0},
        success=True,
    )

    output = summarize_final_output("Run hostname remotely", [log])

    assert output["content"] == "edge-1"


def test_validator_checks_shell_exec_returncode(tmp_path: Path) -> None:
    validator = RuntimeValidator(Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db"))
    success_log = StepLog(
        step=ExecutionStep(id=1, action="shell.exec", args={"node": "edge-1", "command": "hostname"}),
        result={"command": "hostname", "node": "edge-1", "stdout": "edge-1\n", "stderr": "", "returncode": 0},
        success=True,
    )
    failed_log = StepLog(
        step=ExecutionStep(id=2, action="shell.exec", args={"node": "edge-1", "command": "rm /tmp/file"}),
        result={"command": "rm /tmp/file", "node": "edge-1", "stdout": "", "stderr": "permission denied", "returncode": 1},
        success=True,
    )

    success_validation, success_checks = validator.validate([success_log], goal="Run hostname remotely")
    failed_validation, failed_checks = validator.validate([failed_log], goal="Delete file remotely")

    assert success_validation.success is True
    assert success_checks[0]["detail"] == "returncode=0"
    assert failed_validation.success is False
    assert failed_checks[0]["detail"] == "returncode=1"


def test_dangerous_remote_shell_command_pauses_and_resumes(monkeypatch, tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    spec_path = tmp_path / "edge_remote.yaml"
    spec_path.write_text(
        """
version: 1
name: edge_remote
planner:
  temperature: 0.0
runtime:
  max_retries: 0
nodes:
  default: edge-1
  endpoints:
    - name: edge-1
      url: https://edge-1.gateway.internal/exec
tools:
  - shell.exec
""".strip()
    )
    session = engine.create_session(str(spec_path), {"task": "Delete temp.log on node edge-1"}, trigger="manual")

    def fake_post(url, json, timeout):
        return _DummyResponse({"stdout": "removed\n", "stderr": "", "exit_code": 0})

    monkeypatch.setattr("aor_runtime.tools.gateway.requests.post", fake_post)

    def fake_build_plan(**kwargs):
        engine.planner.last_policies_used = ["efficiency"]
        return ExecutionPlan.model_validate(
            {"steps": [{"id": 1, "action": "shell.exec", "args": {"node": "edge-1", "command": "rm temp.log"}}]}
        )

    monkeypatch.setattr(engine.planner, "build_plan", fake_build_plan)

    paused = engine.resume_session(session["id"], trigger="manual")
    assert paused["awaiting_confirmation"] is True
    assert paused["confirmation_kind"] == "dangerous_step"
    assert "edge-1" in str(paused["confirmation_message"])

    final_state = engine.resume_session(session["id"], trigger="manual", approve_dangerous=True)

    assert final_state["status"] == "completed"
    assert final_state["validation"]["success"] is True
    assert final_state["final_output"]["content"] == "removed"


def test_engine_uses_gateway_url_from_runtime_spec_nodes(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    engine = ExecutionEngine(settings)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        """
version: 1
name: node_configured_runtime
planner:
  temperature: 0.0
runtime:
  max_retries: 0
nodes:
  default: edge-1
  endpoints:
    - name: edge-1
      url: https://edge-1.gateway.internal/exec
tools:
  - shell.exec
""".strip()
    )
    observed: dict[str, object] = {}

    def fake_post(url, json, timeout):
        observed["url"] = url
        observed["json"] = json
        observed["timeout"] = timeout
        return _DummyResponse({"stdout": "edge-1\n", "stderr": "", "exit_code": 0})

    monkeypatch.setattr("aor_runtime.tools.gateway.requests.post", fake_post)

    def fake_build_plan(**kwargs):
        engine.planner.last_policies_used = ["efficiency"]
        engine.planner.last_raw_output = None
        engine.planner.last_error_type = None
        return ExecutionPlan.model_validate(
            {"steps": [{"id": 1, "action": "shell.exec", "args": {"command": "hostname"}}]}
        )

    monkeypatch.setattr(engine.planner, "build_plan", fake_build_plan)

    state = engine.run_spec(str(spec_path), {"task": "Run hostname"})

    assert state["status"] == "completed"
    assert state["final_output"]["content"] == "edge-1"
    assert observed == {
        "url": "https://edge-1.gateway.internal/exec",
        "json": {"node": "edge-1", "command": "hostname"},
        "timeout": 30.0,
    }
