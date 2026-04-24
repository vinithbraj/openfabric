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
    message: str
    source: str
    kind: str
    target: str | None = None
    detail: str | None = None

    def as_metadata(self) -> dict[str, Any]:
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
    sanitized = _URL_CREDENTIAL_RE.sub(r"\1***:***@", str(detail or ""))
    return _AT_CREDENTIAL_RE.sub(r"\1***@", sanitized)


def normalize_planner_error(*, error_type: str, detail: str, llm_base_url: str) -> NormalizedRuntimeError | None:
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

    return None


def normalize_runtime_failure(
    *,
    reason: str,
    detail: str,
    step: dict[str, Any] | None,
    settings: Settings,
) -> NormalizedRuntimeError | None:
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
