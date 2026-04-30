"""OpenFABRIC Runtime Module: aor_runtime.runtime.error_normalization

Purpose:
    Normalize low-level tool and validation failures into user-safe error categories.

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

from aor_runtime.config import Settings


_URL_CREDENTIAL_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*://)([^:/@\s]+):([^/@\s]+)@")
_AT_CREDENTIAL_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*://)([^:/@\s]+)@")

_LLM_CONNECTION_TYPES = {"APIConnectionError", "ConnectError", "ConnectionError"}
_LLM_TIMEOUT_TYPES = {"APITimeoutError", "ReadTimeout", "ConnectTimeout", "TimeoutException"}

_SQL_AUTH_MARKERS = (
    "fe_sendauth",
    "no password supplied",
    "password authentication failed",
    "peer authentication failed",
    "authentication failed",
    "scram authentication",
)
_CONNECTION_MARKERS = (
    "connection failed",
    "connection refused",
    "could not connect",
    "failed to connect",
    "network is unreachable",
    "name or service not known",
    "temporary failure in name resolution",
    "could not translate host name",
    "server closed the connection unexpectedly",
)
_TIMEOUT_MARKERS = ("timed out", "timeout", "time-out")
_GATEWAY_CONFIG_MARKERS = ("gateway url is not configured",)


@dataclass(frozen=True)
class NormalizedRuntimeError:
    """Represent normalized runtime error within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by NormalizedRuntimeError.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.error_normalization.NormalizedRuntimeError and related tests.
    """
    message: str
    source: str
    kind: str
    target: str | None = None
    detail: str | None = None

    def as_metadata(self) -> dict[str, Any]:
        """As metadata for NormalizedRuntimeError instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through NormalizedRuntimeError.as_metadata calls and related tests.
        """
        payload: dict[str, Any] = {
            "error_source": self.source,
            "error_kind": self.kind,
        }
        if self.target:
            payload["error_target"] = self.target
        if self.detail:
            payload["error_detail"] = self.detail
        return payload


def sanitize_error_detail(detail: str) -> str:
    """Sanitize error detail for the surrounding runtime workflow.

    Inputs:
        Receives detail for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.error_normalization.sanitize_error_detail.
    """
    sanitized = _URL_CREDENTIAL_RE.sub(r"\1***:***@", str(detail or ""))
    return _AT_CREDENTIAL_RE.sub(r"\1***@", sanitized)


def normalize_planner_error(*, error_type: str, detail: str, llm_base_url: str) -> NormalizedRuntimeError | None:
    """Normalize planner error for the surrounding runtime workflow.

    Inputs:
        Receives error_type, detail, llm_base_url for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.error_normalization.normalize_planner_error.
    """
    sanitized_detail = sanitize_error_detail(detail)
    sanitized_target = sanitize_error_detail(llm_base_url)
    normalized_type = str(error_type or "").strip()
    lowered_detail = sanitized_detail.lower()

    if normalized_type in _LLM_TIMEOUT_TYPES:
        return NormalizedRuntimeError(
            message="LLM timeout error.",
            source="llm",
            kind="timeout",
            target=sanitized_target,
            detail=sanitized_detail,
        )

    if normalized_type in _LLM_CONNECTION_TYPES or lowered_detail == "connection error.":
        return NormalizedRuntimeError(
            message="LLM connection error.",
            source="llm",
            kind="connection",
            target=sanitized_target,
            detail=sanitized_detail,
        )

    if normalized_type == "JSONDecodeError":
        return NormalizedRuntimeError(
            message="Planner returned malformed JSON and the request was not executed.",
            source="planner",
            kind="malformed_json",
            detail=sanitized_detail,
        )

    if normalized_type == "ValidationError" and "ActionPlan" in sanitized_detail:
        return NormalizedRuntimeError(
            message="Planner produced an invalid action plan and the request was not executed.",
            source="planner",
            kind="invalid_action_plan",
            detail=sanitized_detail,
        )

    return None


def normalize_runtime_failure(
    *,
    reason: str,
    detail: str,
    step: dict[str, Any] | None,
    settings: Settings,
) -> NormalizedRuntimeError | None:
    """Normalize runtime failure for the surrounding runtime workflow.

    Inputs:
        Receives reason, detail, step, settings for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.error_normalization.normalize_runtime_failure.
    """
    if str(reason or "") != "tool_execution_failed":
        return None

    step_payload = dict(step or {})
    action = str(step_payload.get("action", "")).strip()
    args = dict(step_payload.get("args") or {})
    sanitized_detail = sanitize_error_detail(detail)
    lowered_detail = sanitized_detail.lower()

    if action == "sql.query":
        database_name = str(args.get("database") or settings.sql_default_database or "").strip()
        target = database_name or None
        if any(marker in lowered_detail for marker in _SQL_AUTH_MARKERS):
            return NormalizedRuntimeError(
                message=f"Database authentication error for {target!r}." if target else "Database authentication error.",
                source="sql",
                kind="authentication",
                target=target,
                detail=sanitized_detail,
            )
        if any(marker in lowered_detail for marker in _TIMEOUT_MARKERS):
            return NormalizedRuntimeError(
                message=f"Database timeout error for {target!r}." if target else "Database timeout error.",
                source="sql",
                kind="timeout",
                target=target,
                detail=sanitized_detail,
            )
        if any(marker in lowered_detail for marker in _CONNECTION_MARKERS):
            return NormalizedRuntimeError(
                message=f"Database connection error for {target!r}." if target else "Database connection error.",
                source="sql",
                kind="connection",
                target=target,
                detail=sanitized_detail,
            )

    if action == "shell.exec":
        node = str(args.get("node") or settings.resolved_default_node() or "").strip()
        target = node or None
        if any(marker in lowered_detail for marker in _GATEWAY_CONFIG_MARKERS):
            return NormalizedRuntimeError(
                message=f"Gateway configuration error for node {target!r}." if target else "Gateway configuration error.",
                source="gateway",
                kind="configuration",
                target=target,
                detail=sanitized_detail,
            )
        if "gateway response validation failed" in lowered_detail or "gateway response is not valid json" in lowered_detail:
            return NormalizedRuntimeError(
                message=f"Gateway response error for node {target!r}." if target else "Gateway response error.",
                source="gateway",
                kind="response",
                target=target,
                detail=sanitized_detail,
            )
        if "gateway request failed" in lowered_detail:
            kind = "timeout" if any(marker in lowered_detail for marker in _TIMEOUT_MARKERS) else "connection"
            label = "Gateway timeout error" if kind == "timeout" else "Gateway connection error"
            return NormalizedRuntimeError(
                message=f"{label} for node {target!r}." if target else f"{label}.",
                source="gateway",
                kind=kind,
                target=target,
                detail=sanitized_detail,
            )

    return None
