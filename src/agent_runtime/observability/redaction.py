"""Redaction helpers for user-visible observability events."""

from __future__ import annotations

import re
from typing import Any

from agent_runtime.observability.events import PipelineEvent

DEFAULT_MAX_STRING_LENGTH = 1000
DEFAULT_MAX_LIST_ITEMS = 20
DEFAULT_MAX_DICT_DEPTH = 4

_SENSITIVE_KEY_TOKENS = (
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "bearer",
    "cookie",
    "credential",
    "credentials",
    "env",
    "key",
    "password",
    "private",
    "secret",
    "session",
    "token",
)
_SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE),
)


def _key_is_sensitive(key: str) -> bool:
    lowered = str(key or "").strip().lower()
    return any(token in lowered for token in _SENSITIVE_KEY_TOKENS)


def _value_is_sensitive(value: str) -> bool:
    text = str(value or "")
    if any(pattern.search(text) for pattern in _SENSITIVE_VALUE_PATTERNS):
        return True
    if "\n" in text and any(
        line.strip().lower().startswith(prefix)
        for prefix in (
            "api_key=",
            "apikey=",
            "password=",
            "secret=",
            "token=",
            "authorization:",
        )
        for line in text.splitlines()
    ):
        return True
    return False


def _truncate_string(value: str, max_length: int) -> str:
    text = str(value or "")
    if len(text) <= max_length:
        return text
    return text[:max_length] + "...[truncated]"


def _sanitize_value(
    value: Any,
    *,
    depth: int,
    max_depth: int,
    max_string_length: int,
    max_list_items: int,
) -> Any:
    if depth > max_depth:
        return "[truncated depth]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if _value_is_sensitive(value):
            return "[redacted]"
        return _truncate_string(value, max_string_length)
    if isinstance(value, list):
        items = [
            _sanitize_value(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_string_length=max_string_length,
                max_list_items=max_list_items,
            )
            for item in value[:max_list_items]
        ]
        if len(value) > max_list_items:
            items.append(f"...[{len(value) - max_list_items} more items]")
        return items
    if isinstance(value, tuple):
        return _sanitize_value(
            list(value),
            depth=depth,
            max_depth=max_depth,
            max_string_length=max_string_length,
            max_list_items=max_list_items,
        )
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _key_is_sensitive(key_text):
                sanitized[key_text] = "[redacted]"
                continue
            sanitized[key_text] = _sanitize_value(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_string_length=max_string_length,
                max_list_items=max_list_items,
            )
        return sanitized
    return _truncate_string(repr(value), max_string_length)


def redact_event(event: PipelineEvent, policy: str) -> PipelineEvent:
    """Return a sanitized event copy suitable for user-visible sinks."""

    max_string_length = DEFAULT_MAX_STRING_LENGTH
    max_list_items = DEFAULT_MAX_LIST_ITEMS
    max_depth = DEFAULT_MAX_DICT_DEPTH
    if policy == "strict":
        max_string_length = min(max_string_length, 400)
        max_list_items = min(max_list_items, 10)
        max_depth = min(max_depth, 3)

    sanitized = event.model_copy(
        update={
            "summary": _truncate_string(event.summary, max_string_length),
            "details": _sanitize_value(
                dict(event.details),
                depth=0,
                max_depth=max_depth,
                max_string_length=max_string_length,
                max_list_items=max_list_items,
            ),
        }
    )
    return sanitized

