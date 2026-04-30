"""OpenFABRIC Runtime Module: aor_runtime.runtime.slurm_safety

Purpose:
    Validate SLURM arguments and block unsupported or unsafe SLURM operations.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


SAFE_SLURM_TOKEN_RE = re.compile(r"^[A-Za-z0-9._-]+$")
SAFE_SLURM_TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?$")
SHELL_META_RE = re.compile(r"[;|&$`><\n]")
DISALLOWED_RAW_KEYS = {"command", "argv", "shell", "gateway_command", "tool", "tool_name", "steps", "execution_plan"}
MUTATION_KEYWORD_RE = re.compile(
    r"\b(?:sbatch|scancel|scontrol\s+update|drain|resume|requeue|suspend|hold|release|kill|delete|"
    r"restart|systemctl|shutdown|poweroff)\b",
    re.IGNORECASE,
)
ALLOWED_JOB_STATES = {"RUNNING", "PENDING", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}
ALLOWED_NODE_STATES = {"idle", "allocated", "mixed", "down", "drained"}
ALLOWED_NODE_GROUPS = {"idle", "allocated", "mixed", "down", "drained", "problematic", "all"}
ALLOWED_GROUP_BY = {"state", "user", "partition", "node", "job_name"}
ALLOWED_METRIC_GROUPS = {
    "cluster_summary",
    "queue_summary",
    "node_summary",
    "problematic_nodes",
    "partition_summary",
    "gpu_summary",
    "accounting_summary",
    "slurmdbd_health",
    "scheduler_health",
    "accounting_health",
}


@dataclass(frozen=True)
class SlurmSafetyResult:
    """Represent slurm safety result within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SlurmSafetyResult.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.slurm_safety.SlurmSafetyResult and related tests.
    """
    valid: bool
    reason: str = ""


def validate_slurm_intent_safety(intent: Any) -> SlurmSafetyResult:
    """Validate slurm intent safety for the surrounding runtime workflow.

    Inputs:
        Receives intent for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_safety.validate_slurm_intent_safety.
    """
    try:
        _validate_payload(_dump_intent(intent))
        _validate_known_fields(intent)
    except ValueError as exc:
        return SlurmSafetyResult(valid=False, reason=str(exc))
    return SlurmSafetyResult(valid=True)


def validate_slurm_frame_safety(frame: Any) -> SlurmSafetyResult:
    """Validate slurm frame safety for the surrounding runtime workflow.

    Inputs:
        Receives frame for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_safety.validate_slurm_frame_safety.
    """
    try:
        payload = frame.to_dict() if hasattr(frame, "to_dict") else frame
        _validate_payload(payload)
    except ValueError as exc:
        return SlurmSafetyResult(valid=False, reason=str(exc))
    return SlurmSafetyResult(valid=True)


def safe_slurm_token(value: str | None, *, field_name: str) -> str | None:
    """Safe slurm token for the surrounding runtime workflow.

    Inputs:
        Receives value, field_name for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_safety.safe_slurm_token.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if SHELL_META_RE.search(text) or "/" in text:
        raise ValueError(f"SLURM {field_name} may not contain shell metacharacters or path separators.")
    if not SAFE_SLURM_TOKEN_RE.fullmatch(text):
        raise ValueError(f"SLURM {field_name} contains unsafe characters.")
    return text


def safe_time_value(value: str | None, *, field_name: str) -> str | None:
    """Safe time value for the surrounding runtime workflow.

    Inputs:
        Receives value, field_name for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_safety.safe_time_value.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if SHELL_META_RE.search(text):
        raise ValueError(f"SLURM {field_name} may not contain shell metacharacters.")
    if not SAFE_SLURM_TIME_RE.fullmatch(text):
        raise ValueError(f"SLURM {field_name} must be YYYY-MM-DD or YYYY-MM-DD HH:MM[:SS].")
    return text


def normalize_job_state(state: str | None) -> str | None:
    """Normalize job state for the surrounding runtime workflow.

    Inputs:
        Receives state for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_safety.normalize_job_state.
    """
    if state is None:
        return None
    normalized = safe_slurm_token(state, field_name="state")
    if normalized is None:
        return None
    normalized = normalized.upper()
    if normalized not in ALLOWED_JOB_STATES:
        raise ValueError(f"Unsupported SLURM job state: {state}")
    return normalized


def normalize_node_state(state: str | None) -> str | None:
    """Normalize node state for the surrounding runtime workflow.

    Inputs:
        Receives state for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_safety.normalize_node_state.
    """
    if state is None:
        return None
    normalized = safe_slurm_token(state, field_name="state")
    if normalized is None:
        return None
    normalized = normalized.lower()
    if normalized not in ALLOWED_NODE_STATES:
        raise ValueError(f"Unsupported SLURM node state: {state}")
    return normalized


def normalize_node_state_group(state_group: str | None) -> str | None:
    """Normalize node state group for the surrounding runtime workflow.

    Inputs:
        Receives state_group for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_safety.normalize_node_state_group.
    """
    if state_group is None:
        return None
    normalized = safe_slurm_token(state_group, field_name="state_group")
    if normalized is None:
        return None
    normalized = normalized.lower()
    if normalized not in ALLOWED_NODE_GROUPS:
        raise ValueError(f"Unsupported SLURM node state group: {state_group}")
    return normalized


def normalize_group_by(group_by: str | None) -> str | None:
    """Normalize group by for the surrounding runtime workflow.

    Inputs:
        Receives group_by for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_safety.normalize_group_by.
    """
    if group_by is None:
        return None
    normalized = safe_slurm_token(group_by, field_name="group_by")
    if normalized is None:
        return None
    normalized = normalized.lower()
    if normalized not in ALLOWED_GROUP_BY:
        raise ValueError(f"Unsupported SLURM group_by: {group_by}")
    return normalized


def normalize_metric_group(metric_group: str | None) -> str | None:
    """Normalize metric group for the surrounding runtime workflow.

    Inputs:
        Receives metric_group for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_safety.normalize_metric_group.
    """
    if metric_group is None:
        return None
    normalized = safe_slurm_token(metric_group, field_name="metric_group")
    if normalized is None:
        return None
    normalized = normalized.lower()
    if normalized not in ALLOWED_METRIC_GROUPS:
        raise ValueError(f"Unsupported SLURM metric group: {metric_group}")
    return normalized


def _validate_payload(value: Any) -> None:
    """Handle the internal validate payload helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_safety._validate_payload.
    """
    if isinstance(value, dict):
        raw_keys = {str(key).lower() for key in value}
        if raw_keys.intersection(DISALLOWED_RAW_KEYS):
            raise ValueError("SLURM intents may not contain raw command/tool/plan fields.")
        for item in value.values():
            _validate_payload(item)
        return
    if isinstance(value, list):
        for item in value:
            _validate_payload(item)
        return
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return
        if SHELL_META_RE.search(text):
            raise ValueError("SLURM intent values may not contain shell metacharacters.")
        if MUTATION_KEYWORD_RE.search(text):
            raise ValueError("SLURM mutation/admin operations are not allowed in read-only intents.")


def _validate_known_fields(intent: Any) -> None:
    """Handle the internal validate known fields helper path for this module.

    Inputs:
        Receives intent for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_safety._validate_known_fields.
    """
    class_name = type(intent).__name__
    if class_name == "SlurmCompoundIntent":
        for child in getattr(intent, "intents", []) or []:
            result = validate_slurm_intent_safety(child)
            if not result.valid:
                raise ValueError(result.reason)
        return
    if hasattr(intent, "state"):
        state = getattr(intent, "state", None)
        if class_name in {"SlurmQueueIntent", "SlurmJobDetailIntent", "SlurmAccountingIntent", "SlurmAccountingAggregateIntent", "SlurmJobCountIntent"}:
            normalize_job_state(state)
        elif state is not None:
            normalize_node_state(state)
    if hasattr(intent, "state_group"):
        normalize_node_state_group(getattr(intent, "state_group", None))
    if hasattr(intent, "group_by"):
        normalize_group_by(getattr(intent, "group_by", None))
    if hasattr(intent, "metric_group"):
        normalize_metric_group(getattr(intent, "metric_group", None))
    for field in ("user", "partition", "node", "job_id", "gateway_node"):
        if hasattr(intent, field):
            safe_slurm_token(getattr(intent, field, None), field_name=field)
    for field in ("start", "end"):
        if hasattr(intent, field):
            safe_time_value(getattr(intent, field, None), field_name=field)


def _dump_intent(intent: Any) -> Any:
    """Handle the internal dump intent helper path for this module.

    Inputs:
        Receives intent for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.slurm_safety._dump_intent.
    """
    if hasattr(intent, "model_dump"):
        return intent.model_dump()
    if hasattr(intent, "to_dict"):
        return intent.to_dict()
    return intent
