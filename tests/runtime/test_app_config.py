from __future__ import annotations

from pathlib import Path

import pytest

from aor_runtime.app_config import APP_CONFIG_FILENAME, load_app_config
from aor_runtime.config import get_settings
from aor_runtime.tools.sql import resolve_sql_databases


CONFIG_TEXT = """
server:
  host: 0.0.0.0
  port: 9000
llm:
  base_url: http://127.0.0.1:9001/v1
  api_key: test-key
  default_model: model-x
  default_temperature: 0.2
  timeout_seconds: 33
runtime:
  allow_destructive_shell: true
  max_plan_retries: 5
sql:
  database_url: null
  databases:
    analytics_db: sqlite:////tmp/analytics.db
  default_database: analytics_db
  row_limit: 123
  timeout_seconds: 17
""".strip()


def test_load_app_config_from_explicit_path(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(CONFIG_TEXT)

    app_config, resolved = load_app_config(config_path)

    assert resolved == config_path.resolve()
    assert app_config.server.host == "0.0.0.0"
    assert app_config.server.port == 9000
    assert app_config.llm.default_model == "model-x"
    assert app_config.sql.databases == {"analytics_db": "sqlite:////tmp/analytics.db"}


def test_get_settings_auto_discovers_config_yaml(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / APP_CONFIG_FILENAME
    config_path.write_text(CONFIG_TEXT)
    monkeypatch.chdir(tmp_path)

    settings = get_settings()

    assert settings.app_config_path == config_path.resolve()
    assert settings.server_host == "0.0.0.0"
    assert settings.server_port == 9000
    assert settings.llm_base_url == "http://127.0.0.1:9001/v1"
    assert settings.llm_api_key == "test-key"
    assert settings.default_model == "model-x"
    assert settings.allow_destructive_shell is True
    assert settings.max_plan_retries == 5
    assert settings.sql_default_database == "analytics_db"
    assert settings.sql_row_limit == 123
    assert settings.sql_timeout_seconds == 17


def test_get_settings_missing_config_raises_helpful_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="Create config.yaml or pass --config"):
        get_settings()


def test_load_app_config_rejects_invalid_schema(tmp_path: Path) -> None:
    config_path = tmp_path / APP_CONFIG_FILENAME
    config_path.write_text(
        """
server:
  host: 127.0.0.1
  port: 0
""".strip()
    )

    with pytest.raises(ValueError, match="server.port"):
        load_app_config(config_path)


def test_resolve_sql_databases_uses_yaml_mapping(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / APP_CONFIG_FILENAME
    config_path.write_text(CONFIG_TEXT)
    monkeypatch.chdir(tmp_path)

    settings = get_settings()

    assert resolve_sql_databases(settings) == {"analytics_db": "sqlite:////tmp/analytics.db"}
