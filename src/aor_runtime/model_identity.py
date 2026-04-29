from __future__ import annotations

from aor_runtime import __version__


DEFAULT_OPENAI_COMPAT_MODEL_NAME = f"OpenFABRIC v{__version__}"
LEGACY_OPENAI_COMPAT_MODEL_NAMES = {"general-purpose-assistant"}


def normalize_openai_compat_model_name(value: str | None) -> str:
    normalized = str(value or "").strip()
    if normalized in LEGACY_OPENAI_COMPAT_MODEL_NAMES:
        return DEFAULT_OPENAI_COMPAT_MODEL_NAME
    return normalized or DEFAULT_OPENAI_COMPAT_MODEL_NAME


def is_accepted_openai_compat_model_name(requested: str, configured: str) -> bool:
    normalized = str(requested or "").strip()
    return normalized == str(configured or "").strip() or normalized in LEGACY_OPENAI_COMPAT_MODEL_NAMES
