from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from aor_runtime import cli


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
