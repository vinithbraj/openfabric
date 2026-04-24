from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from aor_runtime import cli


runner = CliRunner()


class FakeEngine:
    def run_spec(self, spec_path: str, payload: dict, dry_run: bool = False) -> dict:
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
    assert "Commands: /exit, /quit, /history, /last, /new, /clear" in result.stdout
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
    assert result.stdout.count("Commands: /exit, /quit, /history, /last, /new, /clear") == 2
    assert "No turns yet." in result.stdout
