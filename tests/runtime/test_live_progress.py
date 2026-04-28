from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan
from aor_runtime.runtime.engine import ExecutionEngine


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


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        gateway_url="https://gateway.internal/exec",
        available_nodes_raw="localhost",
        default_node="localhost",
        shell_mode="permissive",
    )


def _write_shell_spec(tmp_path: Path) -> Path:
    spec_path = tmp_path / "shell.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "version: 1",
                "name: shell_progress",
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


def _write_slurm_spec(tmp_path: Path) -> Path:
    spec_path = tmp_path / "slurm.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "version: 1",
                "name: slurm_progress",
                "runtime:",
                "  max_retries: 0",
                "nodes:",
                "  default: localhost",
                "  endpoints:",
                "    - name: localhost",
                "      url: https://gateway.internal/exec",
                "tools:",
                "  - slurm.queue",
            ]
        )
    )
    return spec_path


def _patch_shell_plan(engine: ExecutionEngine, monkeypatch) -> None:
    def fake_build_plan(**kwargs):
        engine.planner.last_policies_used = ["efficiency"]
        engine.planner.last_high_level_plan = None
        engine.planner.last_planning_mode = "direct"
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


def _patch_slurm_plan(engine: ExecutionEngine, monkeypatch) -> None:
    def fake_build_plan(**kwargs):
        engine.planner.last_policies_used = ["efficiency"]
        engine.planner.last_high_level_plan = None
        engine.planner.last_planning_mode = "deterministic_intent"
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
                        "action": "slurm.queue",
                        "args": {"gateway_node": "localhost"},
                    }
                ]
            }
        )

    monkeypatch.setattr(engine.planner, "build_plan", fake_build_plan)


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


def test_stream_shell_output_events_are_persisted_and_incremental(tmp_path: Path, monkeypatch) -> None:
    engine = ExecutionEngine(_settings(tmp_path))
    spec_path = _write_shell_spec(tmp_path)
    _patch_shell_plan(engine, monkeypatch)
    _patch_gateway_requests(monkeypatch)

    session = engine.create_session(str(spec_path), {"task": "run command"}, trigger="manual", stream_shell_output=True)
    state = engine.resume_session(session["id"], trigger="manual", stream_shell_output=True)

    assert state["status"] == "completed"
    events = engine.store.get_events(session["id"])
    event_types = [event["event_type"] for event in events]
    assert "executor.step.output" in event_types
    assert "validator.started" in event_types
    assert "finalize.started" in event_types

    output_events = [event for event in events if event["event_type"] == "executor.step.output"]
    assert output_events[0]["payload"]["channel"] == "stdout"
    assert output_events[0]["payload"]["text"] == "hello\n"
    assert output_events[1]["payload"]["channel"] == "stderr"
    assert output_events[1]["payload"]["text"] == "warn\n"

    later_events = engine.store.get_events_after(session["id"], after_id=output_events[0]["id"])
    assert later_events
    assert all(int(event["id"]) > int(output_events[0]["id"]) for event in later_events)


def test_stream_slurm_output_events_are_persisted_and_incremental(tmp_path: Path, monkeypatch) -> None:
    engine = ExecutionEngine(_settings(tmp_path))
    spec_path = _write_slurm_spec(tmp_path)
    _patch_slurm_plan(engine, monkeypatch)
    _patch_gateway_requests(monkeypatch)

    session = engine.create_session(str(spec_path), {"task": "show slurm queue"}, trigger="manual", stream_shell_output=True)
    state = engine.resume_session(session["id"], trigger="manual", stream_shell_output=True)

    assert state["status"] == "completed"
    events = engine.store.get_events(session["id"])
    output_events = [event for event in events if event["event_type"] == "executor.step.output"]
    started_event = next(event for event in events if event["event_type"] == "executor.step.started")
    assert started_event["payload"]["command"] == "squeue -h -o '%i|%u|%T|%P|%j|%M|%D|%R'"
    assert output_events
    assert output_events[0]["payload"]["action"] == "slurm.queue"
    assert output_events[0]["payload"]["channel"] == "stdout"
    assert "hello" in output_events[0]["payload"]["text"]
