"""Shared constants for compatibility-facing runtime surfaces."""

from __future__ import annotations


DEFAULT_COMPAT_SPEC_PATH = "compat://inline"


def is_compat_spec_placeholder(value: str | None) -> bool:
    """Return whether one spec-path value is the neutral compatibility placeholder."""

    normalized = str(value or "").strip()
    return not normalized or normalized == DEFAULT_COMPAT_SPEC_PATH

