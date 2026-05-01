from __future__ import annotations

from fastapi.testclient import TestClient

from aor_runtime.api.app import create_app
from aor_runtime.config import Settings
from aor_runtime.runtime.engine import ExecutionEngine, extract_prompt


def _client() -> TestClient:
    settings = Settings(openai_compat_model_name="OpenFABRIC Echo")
    return TestClient(create_app(settings))


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
    assert response.json()["choices"][0]["message"]["content"] == "list all nodes"


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
    assert "hello stream" in body
    assert "data: [DONE]" in body


def test_run_endpoint_echoes_prompt() -> None:
    client = _client()

    response = client.post(
        "/runs",
        json={"spec_path": "examples/general_purpose_assistant.yaml", "input": {"task": "echo me"}},
    )

    assert response.status_code == 200
    assert response.json()["final_output"]["content"] == "echo me"
