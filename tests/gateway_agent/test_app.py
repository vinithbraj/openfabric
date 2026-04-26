from __future__ import annotations

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
        "version": "0.2.0",
        "capabilities": [
            {"name": "healthz", "description": "Report agent health and the configured logical node."},
            {
                "name": "exec",
                "description": "Execute a local shell command when the request node matches the configured node.",
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
