from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.llm.client import LLMClient


def _settings(tmp_path: Path, *, default_model: str) -> Settings:
    return Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db", default_model=default_model)


def test_resolve_model_uses_settings_default_when_requested_model_is_none(tmp_path: Path) -> None:
    client = LLMClient(_settings(tmp_path, default_model="model-y"))
    client._discovered_models = ["model-y", "model-z"]

    assert client.resolve_model(None) == "model-y"


def test_resolve_model_keeps_explicit_override_when_available(tmp_path: Path) -> None:
    client = LLMClient(_settings(tmp_path, default_model="model-y"))
    client._discovered_models = ["model-x", "model-y"]

    assert client.resolve_model("model-x") == "model-x"
