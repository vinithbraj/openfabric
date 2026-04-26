from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from aor_runtime import cli
from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan
from aor_runtime.runtime.engine import ExecutionEngine


runner = CliRunner()


class FakeEngine:
    def __init__(self) -> None:
        self.run_calls = 0
        self.payloads: list[dict] = []

    def run_spec(self, spec_path: str, payload: dict, dry_run: bool = False) -> dict:
        self.run_calls += 1
        self.payloads.append(dict(payload))
        task = str(payload.get("task", ""))
        return {
            "run_id": "test-run",
            "final_output": {"content": f"echo:{task}", "artifacts": [], "metadata": {}},
            "awaiting_confirmation": False,
        }


def test_chat_new_resets_session_history(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "_build_engine", lambda config_path=None: FakeEngine())
    spec_path = tmp_path / "spec.yaml"

    result = runner.invoke(
        cli.app,
        ["chat", str(spec_path)],
        input="hello\n/last\n/new\n/last\n/exit\n",
    )

    assert result.exit_code == 0
    assert "Commands: /exit, /quit, /history, /last, /new, /clear, /capabilities" in result.stdout
    assert '"role": "user"' in result.stdout
    assert '"content": "hello"' in result.stdout
    assert "Started a new conversation." in result.stdout
    assert "No turns yet." in result.stdout


def test_chat_clear_resets_history_and_reprints_banner(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "_build_engine", lambda config_path=None: FakeEngine())
    spec_path = tmp_path / "spec.yaml"

    result = runner.invoke(
        cli.app,
        ["chat", str(spec_path)],
        input="hello\n/clear\n/last\n/exit\n",
    )

    assert result.exit_code == 0
    assert result.stdout.count(f"Interactive session started for {spec_path}. Type /exit to quit.") == 2
    assert result.stdout.count("Commands: /exit, /quit, /history, /last, /new, /clear, /capabilities") == 2
    assert "No turns yet." in result.stdout


def test_chat_capabilities_lists_spec_tools_and_registry_packs(monkeypatch, tmp_path: Path) -> None:
    engine = FakeEngine()
    monkeypatch.setattr(cli, "_build_engine", lambda config_path=None: engine)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "version: 1",
                "name: demo_agent",
                "description: Demo runtime for chat capabilities.",
                "tools:",
                "  - fs.read",
                "  - fs.search_content",
                "  - shell.exec",
                "nodes:",
                "  default: localhost",
                "  endpoints:",
                "    - name: localhost",
                "      url: http://127.0.0.1:8787/exec",
            ]
        )
    )

    result = runner.invoke(
        cli.app,
        ["chat", str(spec_path)],
        input="/capabilities\n/exit\n",
    )

    assert result.exit_code == 0
    assert '"name": "demo_agent"' in result.stdout
    assert '"fs.search_content"' in result.stdout
    assert '"capability_packs"' in result.stdout
    assert '"name": "filesystem"' in result.stdout
    assert '"name": "compound"' in result.stdout
    assert '"/capabilities"' in result.stdout
    assert engine.run_calls == 0


def test_chat_default_history_lookback_is_zero(monkeypatch, tmp_path: Path) -> None:
    engine = FakeEngine()
    monkeypatch.setattr(cli, "_build_engine", lambda config_path=None: engine)
    spec_path = tmp_path / "spec.yaml"

    result = runner.invoke(
        cli.app,
        ["chat", str(spec_path)],
        input="hello\nagain\n/exit\n",
    )

    assert result.exit_code == 0
    assert engine.run_calls == 2
    assert "session_history" not in engine.payloads[0]
    assert "session_history" not in engine.payloads[1]


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


def test_chat_progress_streams_shell_output(monkeypatch, tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "version: 1",
                "name: chat_progress",
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
    engine = ExecutionEngine(
        Settings(
            workspace_root=tmp_path,
            run_store_path=tmp_path / "runtime.db",
            gateway_url="https://gateway.internal/exec",
            available_nodes_raw="localhost",
            default_node="localhost",
        )
    )

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

    monkeypatch.setattr(engine.planner, "build_plan", fake_build_plan)
    monkeypatch.setattr("aor_runtime.tools.gateway.requests.post", fake_post)
    monkeypatch.setattr(cli, "_build_engine", lambda config_path=None: engine)

    result = runner.invoke(
        cli.app,
        ["chat", str(spec_path), "--progress"],
        input="run it\n/exit\n",
    )

    assert result.exit_code == 0
    assert "Thinking..." in result.stdout
    assert "Executing: shell.exec" in result.stdout
    assert "hello" in result.stdout
    assert "Finished: completed" in result.stdout


def test_chat_failure_prints_prompt_suggestions(monkeypatch, tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "version: 1",
                "name: chat_failure",
                "runtime:",
                "  max_retries: 0",
                "tools:",
                "  - fs.read",
            ]
        )
    )
    engine = ExecutionEngine(Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db"))

    def failing_build_plan(**kwargs):
        engine.planner.last_policies_used = ["filesystem_preference"]
        engine.planner.last_high_level_plan = None
        engine.planner.last_planning_mode = "direct"
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
    monkeypatch.setattr(cli, "_build_engine", lambda config_path=None: engine)

    result = runner.invoke(
        cli.app,
        ["chat", str(spec_path)],
        input="Read the meeting notes and return line 2.\n/exit\n",
    )

    assert result.exit_code == 0
    assert "Suggested prompts:" in result.stdout
    assert "meeting_notes.txt" in result.stdout
