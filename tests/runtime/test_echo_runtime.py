from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from agent_runtime.core.orchestrator import AgentRuntime
from aor_runtime.api.app import create_app
from aor_runtime.config import Settings
from aor_runtime.runtime.engine import ExecutionEngine, extract_prompt


class FakeAgentRuntime:
    """Small fake runtime injected into the OpenAI-compatible API tests."""

    def handle_request(self, raw_prompt: str, context: dict | None = None) -> str:
        payload = dict(context or {})
        callback = payload.get("event_callback")
        if callable(callback):
            callback("[Prompt Classification] Started\n\nPlanning is underway.\n\n")
        return f"handled: {raw_prompt}"


class ConfirmingRuntime:
    """Fake runtime that pauses for confirmation and can resume from a stored trace."""

    def __init__(self) -> None:
        self.last_failure_summary: dict | None = None
        self.last_planning_trace = None

    def handle_request(self, raw_prompt: str, context: dict | None = None) -> str:
        payload = dict(context or {})
        if payload.get("confirmation"):
            self.last_failure_summary = None
            return f"approved: {raw_prompt}"
        self.last_failure_summary = {"category": "confirmation_required"}
        self.last_planning_trace = SimpleNamespace(raw_prompt=raw_prompt)
        return "## Confirmation Required\n\nI am waiting for approval."

    def replay_from_trace(self, trace, context: dict | None = None) -> str:
        _ = trace
        payload = dict(context or {})
        assert payload.get("confirmation") is True
        return "approved from trace"


def _client() -> TestClient:
    settings = Settings(openai_compat_model_name="OpenFABRIC Echo")
    return TestClient(create_app(settings, agent_runtime=FakeAgentRuntime()))


def test_create_app_defaults_agent_runtime_gateway_to_localhost() -> None:
    settings = Settings(gateway_url=None, gateway_endpoints={}, default_node=None)

    app = create_app(settings)

    runtime = app.state.agent_runtime
    assert runtime.execution_engine.safety_policy.config.gateway_default_node == "localhost"
    assert runtime.execution_engine.safety_policy.config.gateway_url == "http://127.0.0.1:8787"


def test_create_app_preserves_explicit_gateway_configuration() -> None:
    settings = Settings(
        gateway_url="http://gateway.example:9000",
        gateway_endpoints={"worker-a": "http://gateway.worker-a:8787"},
        default_node="worker-a",
    )

    app = create_app(settings)

    runtime = app.state.agent_runtime
    assert runtime.execution_engine.safety_policy.config.gateway_default_node == "worker-a"
    assert runtime.execution_engine.safety_policy.config.gateway_url == "http://gateway.worker-a:8787"


def test_create_app_enables_shell_when_shell_mode_is_not_disabled() -> None:
    settings = Settings(shell_mode="read_only")

    app = create_app(settings)

    runtime = app.state.agent_runtime
    assert runtime.execution_engine.safety_policy.config.allow_shell_execution is True


def test_create_app_disables_shell_when_shell_mode_is_disabled() -> None:
    settings = Settings(shell_mode="disabled")

    app = create_app(settings)

    runtime = app.state.agent_runtime
    assert runtime.execution_engine.safety_policy.config.allow_shell_execution is False


def test_extract_prompt_prefers_task_field() -> None:
    assert extract_prompt({"task": "hello", "prompt": "ignored"}) == "hello"


def test_engine_echoes_prompt() -> None:
    engine = ExecutionEngine(Settings())

    result = engine.run_spec("examples/general_purpose_assistant.yaml", {"task": "count patients"})

    assert result["status"] == "completed"
    assert result["final_output"]["content"] == "count patients"


def test_healthz_reports_echo_mode() -> None:
    response = _client().get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "mode": "echo"}


def test_openai_chat_completion_echoes_latest_user_prompt() -> None:
    client = _client()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [
                {"role": "system", "content": "ignored"},
                {"role": "user", "content": "list all nodes"},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "handled: list all nodes"


def test_openai_chat_completion_stream_echoes_prompt_and_done() -> None:
    client = _client()

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "auto",
            "stream": True,
            "messages": [{"role": "user", "content": "hello stream"}],
        },
    ) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert "[Prompt Classification] Started" in body
    assert "handled: hello stream" in body
    assert "data: [DONE]" in body


def test_openai_chat_completion_confirmation_flow_round_trips() -> None:
    client = TestClient(create_app(Settings(openai_compat_model_name="OpenFABRIC Echo"), agent_runtime=ConfirmingRuntime()))

    initial = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "delete the file"}],
        },
    )

    assert initial.status_code == 200
    confirmation_content = initial.json()["choices"][0]["message"]["content"]
    assert "Confirmation Required" in confirmation_content
    assert "Confirmation ID:" in confirmation_content

    approved = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [
                {"role": "user", "content": "delete the file"},
                {"role": "assistant", "content": confirmation_content},
                {"role": "user", "content": "approve"},
            ],
        },
    )

    assert approved.status_code == 200
    assert approved.json()["choices"][0]["message"]["content"] == "approved from trace"


def test_openai_chat_completion_confirmation_can_be_cancelled() -> None:
    client = TestClient(create_app(Settings(openai_compat_model_name="OpenFABRIC Echo"), agent_runtime=ConfirmingRuntime()))

    initial = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "delete the file"}],
        },
    )
    confirmation_content = initial.json()["choices"][0]["message"]["content"]

    cancelled = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [
                {"role": "user", "content": "delete the file"},
                {"role": "assistant", "content": confirmation_content},
                {"role": "user", "content": "cancel"},
            ],
        },
    )

    assert cancelled.status_code == 200
    assert "Confirmation Cancelled" in cancelled.json()["choices"][0]["message"]["content"]


def test_run_endpoint_echoes_prompt() -> None:
    client = _client()

    response = client.post(
        "/runs",
        json={"spec_path": "examples/general_purpose_assistant.yaml", "input": {"task": "echo me"}},
    )

    assert response.status_code == 200
    assert response.json()["final_output"]["content"] == "echo me"
