from __future__ import annotations

import logging
from pathlib import Path

from fastapi.testclient import TestClient

from gateway_agent.app import create_app
from gateway_agent.config import Settings


def _client(tmp_path: Path, **overrides) -> TestClient:
    payload = {
        "node_name": "localhost",
        "workdir": tmp_path,
        "exec_timeout_seconds": 1,
    }
    payload.update(overrides)
    settings = Settings(**payload)
    return TestClient(create_app(settings))


def test_healthz_returns_configured_node(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "node": "localhost"}


def test_capabilities_returns_agent_version_and_capability_list(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/capabilities")

    assert response.status_code == 200
    assert response.json() == {
        "node": "localhost",
        "version": "0.4.0",
        "capabilities": [
            {"name": "healthz", "description": "Report agent health and the configured logical node."},
            {
                "name": "exec",
                "description": "Execute a local shell command when the request node matches the configured node.",
            },
            {
                "name": "exec_stream",
                "description": "Execute a local shell command and stream stdout/stderr when the request node matches the configured node.",
            },
        ],
    }


def test_exec_runs_command_successfully(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post("/exec", json={"node": "localhost", "command": "printf 'hello'"})

    assert response.status_code == 200
    assert response.json() == {"stdout": "hello", "stderr": "", "exit_code": 0}


def test_exec_returns_non_zero_exit_without_transport_failure(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post("/exec", json={"node": "localhost", "command": "echo 'nope' >&2; exit 7"})

    assert response.status_code == 200
    assert response.json() == {"stdout": "", "stderr": "nope\n", "exit_code": 7}


def test_exec_rejects_node_mismatch(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post("/exec", json={"node": "edge-1", "command": "hostname"})

    assert response.status_code == 400
    assert response.json()["detail"] == "Node mismatch. This agent serves node 'localhost'."


def test_exec_rejects_blank_command(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post("/exec", json={"node": "localhost", "command": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "Command is required."


def test_exec_times_out_with_exit_code_124(tmp_path: Path) -> None:
    client = _client(tmp_path, exec_timeout_seconds=0.01)

    response = client.post("/exec", json={"node": "localhost", "command": "sleep 0.1"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["stdout"] == ""
    assert payload["exit_code"] == 124
    assert "timed out" in payload["stderr"]


def test_exec_stream_emits_stdout_stderr_and_completion(tmp_path: Path) -> None:
    client = _client(tmp_path)

    with client.stream(
        "POST",
        "/exec/stream",
        json={"node": "localhost", "command": "printf 'hello\\n'; printf 'oops\\n' >&2"},
    ) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert "event: stdout" in body
    assert "event: stderr" in body
    assert "event: completed" in body
    assert "hello\\n" in body
    assert "oops\\n" in body
    assert '"exit_code": 0' in body


def test_exec_stream_reports_timeout_completion(tmp_path: Path) -> None:
    client = _client(tmp_path, exec_timeout_seconds=0.01)

    with client.stream(
        "POST",
        "/exec/stream",
        json={"node": "localhost", "command": "sleep 0.1"},
    ) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert "event: stderr" in body
    assert "timed out" in body
    assert "event: completed" in body
    assert '"exit_code": 124' in body


def test_exec_traces_command_only_when_enabled(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    client = _client(tmp_path, trace_commands=True)

    response = client.post("/exec", json={"node": "localhost", "command": "printf 'hello'"})

    assert response.status_code == 200
    assert "Gateway exec on localhost: printf 'hello'" in caplog.text


def test_exec_does_not_trace_command_by_default(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    client = _client(tmp_path)

    response = client.post("/exec", json={"node": "localhost", "command": "printf 'hello'"})

    assert response.status_code == 200
    assert "Gateway exec on localhost: printf 'hello'" not in caplog.text


def test_exec_stream_traces_command_only_when_enabled(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    client = _client(tmp_path, trace_commands=True)

    with client.stream(
        "POST",
        "/exec/stream",
        json={"node": "localhost", "command": "printf 'hello\\n'"},
    ) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert "event: completed" in body
    assert "Gateway exec stream on localhost: printf 'hello\\n'" in caplog.text
