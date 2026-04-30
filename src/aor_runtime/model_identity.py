"""OpenFABRIC Runtime Module: aor_runtime.model_identity

Purpose:
    Define model identifiers exposed through OpenAI-compatible APIs.

Responsibilities:
    Map the runtime version to the OpenFABRIC model id while accepting legacy aliases.

Data flow / Interfaces:
    Used by API request validation and model-list endpoints.

Boundaries:
    Keeps public model identity stable without changing planner model selection.
"""

from __future__ import annotations

from aor_runtime import __version__


DEFAULT_OPENAI_COMPAT_MODEL_NAME = f"OpenFABRIC v{__version__}"
LEGACY_OPENAI_COMPAT_MODEL_NAMES = {"general-purpose-assistant"}


def normalize_openai_compat_model_name(value: str | None) -> str:
    """Normalize openai compat model name for the surrounding runtime workflow.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by OpenFABRIC runtime support code paths that import or call aor_runtime.model_identity.normalize_openai_compat_model_name.
    """
    normalized = str(value or "").strip()
    if normalized in LEGACY_OPENAI_COMPAT_MODEL_NAMES:
        return DEFAULT_OPENAI_COMPAT_MODEL_NAME
    return normalized or DEFAULT_OPENAI_COMPAT_MODEL_NAME


def is_accepted_openai_compat_model_name(requested: str, configured: str) -> bool:
    """Is accepted openai compat model name for the surrounding runtime workflow.

    Inputs:
        Receives requested, configured for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by OpenFABRIC runtime support code paths that import or call aor_runtime.model_identity.is_accepted_openai_compat_model_name.
    """
    normalized = str(requested or "").strip()
    return normalized == str(configured or "").strip() or normalized in LEGACY_OPENAI_COMPAT_MODEL_NAMES
