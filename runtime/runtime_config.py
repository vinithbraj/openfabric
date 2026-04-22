from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "artifacts" / "runtime_config" / "settings.json"
_LOCK = threading.RLock()
_CACHE_DATA: dict[str, Any] | None = None
_CACHE_MTIME: float | None = None


BOOL_SETTINGS: dict[str, dict[str, Any]] = {
    "chat_progress_enabled": {
        "env": "OPENFABRIC_GATEWAY_STREAM_PROGRESS",
        "default": True,
        "label": "Chat progress traces",
        "description": "Stream planning, validation, and step trace blocks back through chat responses.",
        "scope": "gateway",
    },
    "console_event_logs_enabled": {
        "env": "OPENFABRIC_CONSOLE_EVENT_LOGS",
        "default": True,
        "label": "Console event logs",
        "description": "Print [EVENT] lines for emitted workflow events in the local server terminal.",
        "scope": "gateway_and_local_agents",
    },
    "console_full_event_logs_enabled": {
        "env": "OPENFABRIC_FULL_EVENT_LOGS",
        "default": False,
        "label": "Full event payloads",
        "description": "Show complete event payloads instead of compact excerpts in terminal event logs.",
        "scope": "gateway_and_local_agents",
    },
    "console_debug_logs_enabled": {
        "env": "OPENFABRIC_DEBUG_LOGS",
        "default": True,
        "label": "Debug logs",
        "description": "Emit yellow debug lines from agents and runtime helpers.",
        "scope": "gateway_and_local_agents",
    },
    "console_raw_logs_enabled": {
        "env": "OPENFABRIC_RAW_LOGS",
        "default": True,
        "label": "Raw logs",
        "description": "Emit magenta raw diagnostic lines from agent execution helpers.",
        "scope": "gateway_and_local_agents",
    },
}


def _config_path() -> Path:
    override = os.getenv("OPENFABRIC_RUNTIME_CONFIG_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return _DEFAULT_CONFIG_PATH


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _env_default(key: str) -> bool:
    spec = BOOL_SETTINGS[key]
    return _coerce_bool(os.getenv(spec["env"]), spec["default"])


def _read_file_locked() -> dict[str, Any]:
    global _CACHE_DATA, _CACHE_MTIME
    path = _config_path()
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        _CACHE_DATA = {}
        _CACHE_MTIME = None
        return {}

    if _CACHE_DATA is not None and _CACHE_MTIME == mtime:
        return deepcopy(_CACHE_DATA)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}

    _CACHE_DATA = data
    _CACHE_MTIME = mtime
    return deepcopy(data)


def _write_file_locked(data: dict[str, Any]) -> None:
    global _CACHE_DATA, _CACHE_MTIME
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **data,
        "_meta": {
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)
    try:
        _CACHE_MTIME = path.stat().st_mtime
    except FileNotFoundError:
        _CACHE_MTIME = None
    _CACHE_DATA = payload


def current_runtime_settings() -> dict[str, bool]:
    with _LOCK:
        file_data = _read_file_locked()
        return {
            key: _coerce_bool(file_data.get(key), _env_default(key))
            for key in BOOL_SETTINGS
        }


def get_runtime_setting(key: str) -> bool:
    if key not in BOOL_SETTINGS:
        raise KeyError(key)
    return current_runtime_settings()[key]


def update_runtime_settings(values: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(key for key in values if key not in BOOL_SETTINGS)
    if unknown:
        raise KeyError(", ".join(unknown))

    with _LOCK:
        file_data = _read_file_locked()
        file_data.pop("_meta", None)
        for key, value in values.items():
            file_data[key] = _coerce_bool(value, _env_default(key))
        _write_file_locked(file_data)
    return describe_runtime_settings()


def reset_runtime_settings(keys: list[str] | None = None) -> dict[str, Any]:
    with _LOCK:
        file_data = _read_file_locked()
        file_data.pop("_meta", None)
        if keys:
            for key in keys:
                file_data.pop(key, None)
        else:
            file_data = {}
        _write_file_locked(file_data)
    return describe_runtime_settings()


def describe_runtime_settings() -> dict[str, Any]:
    with _LOCK:
        file_data = _read_file_locked()
        values = {
            key: _coerce_bool(file_data.get(key), _env_default(key))
            for key in BOOL_SETTINGS
        }
        meta = file_data.get("_meta", {})
        updated_at = meta.get("updated_at") if isinstance(meta, dict) else None

    return {
        "config_path": str(_config_path()),
        "updated_at": updated_at,
        "settings": [
            {
                "id": key,
                "label": spec["label"],
                "description": spec["description"],
                "scope": spec["scope"],
                "env_var": spec["env"],
                "default": spec["default"],
                "effective_default": _env_default(key),
                "value": values[key],
            }
            for key, spec in BOOL_SETTINGS.items()
        ],
    }
