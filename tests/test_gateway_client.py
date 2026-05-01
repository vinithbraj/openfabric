from __future__ import annotations

import json
from urllib import error as urllib_error

import pytest

from agent_runtime.capabilities.filesystem import ListDirectoryCapability
from agent_runtime.core.config import RuntimeConfig
from agent_runtime.core.types import ActionNode
from agent_runtime.execution.errors import GatewayExecutionError
from agent_runtime.execution.gateway_client import GatewayClient


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _ = exc_type, exc, tb
        return None


def _node() -> ActionNode:
    return ActionNode(
        id="node-fs",
        task_id="task-fs",
        description="list files",
        semantic_verb="read",
        capability_id="filesystem.list_directory",
        operation_id="list_directory",
        arguments={"path": "."},
        safety_labels=[],
    )


def test_gateway_client_resolves_default_node_and_builds_command() -> None:
    client = GatewayClient(
        RuntimeConfig(
            gateway_default_node="worker-a",
            gateway_endpoints={"worker-a": "http://gateway-a:8787"},
        )
    )

    node = client.resolve_node({})
    url = client.resolve_url(node)
    command = client.build_command(ListDirectoryCapability(), {"path": ".", "limit": 10})

    assert node == "worker-a"
    assert url == "http://gateway-a:8787"
    assert "gateway_agent.remote_runner" in command
    assert "--operation" in command
    assert "filesystem.list_directory" in command


def test_gateway_client_supports_per_request_node_override() -> None:
    client = GatewayClient(
        RuntimeConfig(
            gateway_default_node="worker-a",
            gateway_endpoints={
                "worker-a": "http://gateway-a:8787",
                "worker-b": "http://gateway-b:8787",
            },
        )
    )

    assert client.resolve_node({"gateway_node": "worker-b"}) == "worker-b"
    assert client.resolve_url("worker-b") == "http://gateway-b:8787"


def test_gateway_client_invokes_exec_and_parses_remote_json(monkeypatch: pytest.MonkeyPatch) -> None:
    client = GatewayClient(RuntimeConfig(gateway_url="http://gateway:8787"))

    def fake_urlopen(request, timeout):
        _ = timeout
        assert request.full_url == "http://gateway:8787/exec"
        body = json.loads(request.data.decode("utf-8"))
        assert body["node"] == "localhost"
        assert "filesystem.list_directory" in body["command"]
        return _FakeHTTPResponse(
            {
                "stdout": json.dumps(
                    {
                        "status": "success",
                        "data_preview": {"entries": [{"name": "README.md", "path": "README.md"}]},
                        "metadata": {"entry_count": 1},
                    }
                ),
                "stderr": "",
                "exit_code": 0,
            }
        )

    monkeypatch.setattr("agent_runtime.execution.gateway_client.urllib_request.urlopen", fake_urlopen)

    result = client.invoke(
        node=_node(),
        capability=ListDirectoryCapability(),
        arguments={"path": "."},
        execution_context={},
    )

    assert result.status == "success"
    assert result.data_preview["entries"][0]["name"] == "README.md"
    assert result.metadata["gateway_node"] == "localhost"


def test_gateway_client_handles_gateway_unavailability(monkeypatch: pytest.MonkeyPatch) -> None:
    client = GatewayClient(RuntimeConfig(gateway_url="http://gateway:8787"))

    def fake_urlopen(request, timeout):
        _ = request, timeout
        raise urllib_error.URLError("connection refused")

    monkeypatch.setattr("agent_runtime.execution.gateway_client.urllib_request.urlopen", fake_urlopen)

    with pytest.raises(GatewayExecutionError):
        client.invoke(
            node=_node(),
            capability=ListDirectoryCapability(),
            arguments={"path": "."},
            execution_context={},
        )
