from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from aor_runtime.api.app import create_app
from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan


class _JsonResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class _StreamResponse:
    def __init__(self, events: list[tuple[str, dict]]) -> None:
        self.events = events

    def raise_for_status(self) -> None:
        return None

    def __enter__(self) -> "_StreamResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def iter_lines(self, decode_unicode: bool = True):
        for event_name, payload in self.events:
            yield f"event: {event_name}"
            yield f"data: {__import__('json').dumps(payload)}"
            yield ""


def _write_shell_spec(tmp_path: Path) -> Path:
    spec_path = tmp_path / "shell_api.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "version: 1",
                "name: shell_api",
                "runtime:",
                "  max_retries: 0",
                "nodes:",
                "  default: localhost",
                "  endpoints:",
                "    - name: localhost",
                "      url: https://gateway.internal/exec",
                "tools:",
                "  - shell.exec",
            ]
        )
    )
    return spec_path


def _settings(tmp_path: Path, spec_path: Path) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        gateway_url="https://gateway.internal/exec",
        available_nodes_raw="localhost",
        default_node="localhost",
        shell_mode="permissive",
        openai_compat_enabled=True,
        openai_compat_model_name="openfabric-agent",
        openai_compat_spec_path=str(spec_path),
    )


def _patch_shell_plan(engine, monkeypatch) -> None:
    def fake_build_plan(**kwargs):
        engine.planner.last_policies_used = ["efficiency"]
        engine.planner.last_high_level_plan = None
        engine.planner.last_planning_mode = "validator_enforced_action_planner"
        engine.planner.last_llm_calls = 0
        engine.planner.last_error_stage = None
        engine.planner.last_plan_repairs = []
        engine.planner.last_plan_canonicalized = False
        engine.planner.last_original_execution_plan = None
        engine.planner.last_canonicalized_execution_plan = None
        return ExecutionPlan.model_validate(
            {
                "steps": [
                    {
                        "id": 1,
                        "action": "shell.exec",
                        "args": {"node": "localhost", "command": "printf 'hello\\n'; printf 'warn\\n' >&2"},
                    }
                ]
            }
        )

    monkeypatch.setattr(engine.planner, "build_plan", fake_build_plan)


def _patch_planner_failure(engine, monkeypatch) -> None:
    def failing_build_plan(**kwargs):
        engine.planner.last_policies_used = ["filesystem_preference"]
        engine.planner.last_high_level_plan = None
        engine.planner.last_planning_mode = "validator_enforced_action_planner"
        engine.planner.last_llm_calls = 0
        engine.planner.last_error_stage = "direct"
        engine.planner.last_plan_repairs = []
        engine.planner.last_plan_canonicalized = False
        engine.planner.last_original_execution_plan = None
        engine.planner.last_canonicalized_execution_plan = None
        engine.planner.last_error_type = "RuntimeError"
        engine.planner.last_raw_output = None
        raise RuntimeError("planner boom")

    monkeypatch.setattr(engine.planner, "build_plan", failing_build_plan)


def _patch_gateway_requests(monkeypatch) -> None:
    def fake_post(url, json, timeout, stream=False):
        if stream:
            return _StreamResponse(
                [
                    ("stdout", {"type": "stdout", "text": "hello\n"}),
                    ("stderr", {"type": "stderr", "text": "warn\n"}),
                    ("completed", {"type": "completed", "exit_code": 0}),
                ]
            )
        return _JsonResponse({"stdout": "hello\n", "stderr": "warn\n", "exit_code": 0})

    monkeypatch.setattr("aor_runtime.tools.gateway.requests.post", fake_post)


def test_session_events_endpoint_and_stream(tmp_path: Path, monkeypatch) -> None:
    spec_path = _write_shell_spec(tmp_path)
    app = create_app(_settings(tmp_path, spec_path))
    engine = app.state.engine
    _patch_shell_plan(engine, monkeypatch)
    _patch_gateway_requests(monkeypatch)
    client = TestClient(app)

    session = engine.create_session(str(spec_path), {"task": "run"}, trigger="manual", stream_shell_output=True)
    engine.resume_session(session["id"], trigger="manual", stream_shell_output=True)

    events_response = client.get(f"/sessions/{session['id']}/events")
    assert events_response.status_code == 200
    events = events_response.json()
    assert any(event["event_type"] == "executor.step.output" for event in events)
    started_event = next(event for event in events if event["event_type"] == "executor.step.started")
    assert started_event["payload"]["command"] == "printf 'hello\\n'; printf 'warn\\n' >&2"

    second_event_id = events[1]["id"]
    filtered = client.get(f"/sessions/{session['id']}/events", params={"after_id": second_event_id})
    assert filtered.status_code == 200
    assert all(int(event["id"]) > int(second_event_id) for event in filtered.json())

    with client.stream("GET", f"/sessions/{session['id']}/events/stream") as response:
        body = response.read().decode()
    assert response.status_code == 200
    assert "event: planner.started" in body
    assert "event: executor.step.output" in body
    assert "event: finalize.completed" in body


def test_runs_stream_and_openai_compat_stream(tmp_path: Path, monkeypatch) -> None:
    spec_path = _write_shell_spec(tmp_path)
    app = create_app(_settings(tmp_path, spec_path))
    engine = app.state.engine
    _patch_shell_plan(engine, monkeypatch)
    _patch_gateway_requests(monkeypatch)
    client = TestClient(app)

    with client.stream("POST", "/runs/stream", json={"spec_path": str(spec_path), "input": {"task": "run"}}) as response:
        body = response.read().decode()
    assert response.status_code == 200
    assert "event: planner.started" in body
    assert "event: executor.step.output" in body
    assert "event: finalize.completed" in body

    models = client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "openfabric-agent"

    completion = client.post(
        "/v1/chat/completions",
        json={
            "model": "openfabric-agent",
            "messages": [{"role": "user", "content": "run the command"}],
            "stream": False,
        },
    )
    assert completion.status_code == 200
    content = completion.json()["choices"][0]["message"]["content"]
    assert "## Result" in content
    assert "hello" in content
    assert "## Command Used" in content

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "openfabric-agent",
            "messages": [{"role": "user", "content": "run the command"}],
            "stream": True,
        },
    ) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert "Thinking..." not in body
    assert "Executing shell.exec on localhost:" not in body
    assert "printf" in body
    assert "## Command Used" in body
    assert "hello" in body
    assert "[stderr] warn\\n" not in body
    assert "data: [DONE]" in body
    assert app.state.active_runs.active_run_count() == 0
    assert app.state.active_runs.active_process_count() == 0


def test_failed_run_and_openai_surfaces_hide_prompt_suggestions_by_default(tmp_path: Path, monkeypatch) -> None:
    spec_path = _write_shell_spec(tmp_path)
    app = create_app(_settings(tmp_path, spec_path))
    engine = app.state.engine
    _patch_planner_failure(engine, monkeypatch)
    client = TestClient(app)

    run_response = client.post(
        "/runs",
        json={"spec_path": str(spec_path), "input": {"task": "Read the meeting notes and return line 2."}},
    )
    assert run_response.status_code == 200
    run_payload = run_response.json()
    assert "Suggested prompts:" not in run_payload["final_output"]["content"]
    assert run_payload["final_output"]["metadata"]["failure_type"] == "ambiguous_file_reference"
    assert run_payload["final_output"]["metadata"]["suggestion_count"] >= 1

    session_response = client.get(f"/sessions/{run_payload['session_id']}")
    assert session_response.status_code == 200
    session_payload = session_response.json()
    assert session_payload["latest_snapshot"]["final_output"]["metadata"]["failure_type"] == "ambiguous_file_reference"

    completion = client.post(
        "/v1/chat/completions",
        json={
            "model": "openfabric-agent",
            "messages": [{"role": "user", "content": "Read the meeting notes and return line 2."}],
            "stream": False,
        },
    )
    assert completion.status_code == 200
    assert "Suggested prompts:" not in completion.json()["choices"][0]["message"]["content"]

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "openfabric-agent",
            "messages": [{"role": "user", "content": "Read the meeting notes and return line 2."}],
            "stream": True,
        },
    ) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert "Suggested prompts:" not in body
