from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from aor_runtime import cli


runner = CliRunner()


CONFIG_TEXT = """
server:
  host: 0.0.0.0
  port: 9010
llm:
  base_url: http://127.0.0.1:8000/v1
  api_key: local
  default_model: model-y
  default_temperature: 0.3
  timeout_seconds: 60
runtime:
  allow_destructive_shell: false
  max_plan_retries: 4
sql:
  database_url: null
  databases: {}
  default_database: null
  row_limit: 444
  timeout_seconds: 12
""".strip()


def test_build_engine_loads_yaml_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(CONFIG_TEXT)

    engine = cli._build_engine(config_path)

    assert engine.settings.server_host == "0.0.0.0"
    assert engine.settings.server_port == 9010
    assert engine.settings.default_model == "model-y"
    assert engine.settings.sql_row_limit == 444


def test_serve_uses_yaml_host_port(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(CONFIG_TEXT)
    observed: dict[str, object] = {}

    def fake_run(app, host, port, reload, factory):
        observed["app"] = app
        observed["host"] = host
        observed["port"] = port
        observed["reload"] = reload
        observed["factory"] = factory

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    result = runner.invoke(cli.app, ["serve", "--config", str(config_path)])

    assert result.exit_code == 0
    assert observed["app"] == "aor_runtime.api.app:create_app"
    assert observed["host"] == "0.0.0.0"
    assert observed["port"] == 9010
    assert observed["factory"] is True


def test_serve_cli_overrides_yaml_host_port(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(CONFIG_TEXT)
    observed: dict[str, object] = {}

    def fake_run(app, host, port, reload, factory):
        observed["host"] = host
        observed["port"] = port

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    result = runner.invoke(cli.app, ["serve", "--config", str(config_path), "--host", "127.0.0.1", "--port", "8123"])

    assert result.exit_code == 0
    assert observed["host"] == "127.0.0.1"
    assert observed["port"] == 8123
