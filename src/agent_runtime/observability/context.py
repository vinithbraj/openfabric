"""Request-scoped observability helpers."""

from __future__ import annotations

from typing import Any

from agent_runtime.observability.events import (
    EVENT_STAGE_COMPLETED,
    EVENT_STAGE_FAILED,
    EVENT_STAGE_STARTED,
    PipelineEvent,
)
from agent_runtime.observability.redaction import redact_event
from agent_runtime.observability.sinks import CallbackEventSink, EventSink, OpenWebUIEventSink


class ObservabilityContext:
    """Request-scoped event emission wrapper with redaction and sink fan-out."""

    def __init__(
        self,
        *,
        request_id: str,
        enabled: bool = False,
        debug: bool = False,
        sinks: list[EventSink] | None = None,
        redaction_policy: str = "standard",
    ) -> None:
        self.request_id = request_id
        self.enabled = enabled
        self.debug = debug
        self.sinks = list(sinks or [])
        self.redaction_policy = redaction_policy

    def emit(self, event: PipelineEvent) -> None:
        """Emit one event safely to every configured sink."""

        if not self.enabled:
            return
        if event.debug_only and not self.debug:
            return
        sanitized = redact_event(event, self.redaction_policy)
        for sink in list(self.sinks):
            try:
                sink.emit(sanitized)
            except Exception:
                continue

    def stage_started(
        self,
        stage: str,
        title: str,
        summary: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.info(stage, EVENT_STAGE_STARTED, title, summary, details or {})

    def stage_completed(
        self,
        stage: str,
        title: str,
        summary: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.info(stage, EVENT_STAGE_COMPLETED, title, summary, details or {})

    def stage_failed(
        self,
        stage: str,
        title: str,
        summary: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.error(stage, EVENT_STAGE_FAILED, title, summary, details or {})

    def info(
        self,
        stage: str,
        event_type: str,
        title: str,
        summary: str,
        details: dict[str, Any] | None = None,
        *,
        debug_only: bool = False,
    ) -> None:
        self.emit(
            PipelineEvent(
                request_id=self.request_id,
                level="info",
                stage=stage,
                event_type=event_type,
                title=title,
                summary=summary,
                details=details or {},
                debug_only=debug_only,
            )
        )

    def warning(
        self,
        stage: str,
        event_type: str,
        title: str,
        summary: str,
        details: dict[str, Any] | None = None,
        *,
        debug_only: bool = False,
    ) -> None:
        self.emit(
            PipelineEvent(
                request_id=self.request_id,
                level="warning",
                stage=stage,
                event_type=event_type,
                title=title,
                summary=summary,
                details=details or {},
                debug_only=debug_only,
            )
        )

    def error(
        self,
        stage: str,
        event_type: str,
        title: str,
        summary: str,
        details: dict[str, Any] | None = None,
        *,
        debug_only: bool = False,
    ) -> None:
        self.emit(
            PipelineEvent(
                request_id=self.request_id,
                level="error",
                stage=stage,
                event_type=event_type,
                title=title,
                summary=summary,
                details=details or {},
                debug_only=debug_only,
            )
        )


def build_observability_context(
    request_id: str,
    context: dict[str, Any] | None = None,
) -> ObservabilityContext:
    """Build one observability context from arbitrary request context."""

    payload = dict(context or {})
    config = payload.get("observability")
    sinks: list[EventSink] = []
    enabled = False
    debug = False
    redaction_policy = "standard"

    if isinstance(config, dict):
        enabled = bool(config.get("enabled", False))
        debug = bool(config.get("debug", False))
        redaction_policy = str(config.get("redaction_policy", "standard"))
        sink = config.get("sink")
        if isinstance(sink, EventSink):
            sinks.append(sink)
        many_sinks = config.get("sinks")
        if isinstance(many_sinks, list):
            sinks.extend(item for item in many_sinks if isinstance(item, EventSink))
        callback = config.get("callback")
        if callable(callback):
            sinks.append(OpenWebUIEventSink(callback))

    direct_sink = payload.get("event_sink")
    if isinstance(direct_sink, EventSink):
        sinks.append(direct_sink)
        enabled = True

    callback = payload.get("event_callback")
    if callable(callback):
        sinks.append(OpenWebUIEventSink(callback))
        enabled = True

    return ObservabilityContext(
        request_id=request_id,
        enabled=enabled,
        debug=debug,
        sinks=sinks,
        redaction_policy=redaction_policy,
    )


def observability_from_context(context: dict[str, Any] | None) -> ObservabilityContext | None:
    """Resolve an observability context from a generic runtime context payload."""

    if not isinstance(context, dict):
        return None
    observability = context.get("observability")
    if isinstance(observability, ObservabilityContext):
        return observability
    return None

