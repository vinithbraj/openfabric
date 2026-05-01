"""Observability event sinks."""

from __future__ import annotations

from typing import Callable

from agent_runtime.core.logging import get_logger
from agent_runtime.observability.events import PipelineEvent
from agent_runtime.observability.openwebui_format import format_event_for_openwebui


class EventSink:
    """Base class for pipeline event sinks."""

    def emit(self, event: PipelineEvent) -> None:  # pragma: no cover - interface only
        raise NotImplementedError


class InMemoryEventSink(EventSink):
    """Store emitted events in memory for tests or diagnostics."""

    def __init__(self) -> None:
        self.events: list[PipelineEvent] = []

    def emit(self, event: PipelineEvent) -> None:
        self.events.append(event)


class LoggingEventSink(EventSink):
    """Log pipeline events to the runtime logger."""

    def __init__(self, logger_name: str = "observability") -> None:
        self.logger = get_logger(logger_name)

    def emit(self, event: PipelineEvent) -> None:
        self.logger.info(event.model_dump_json())


class CallbackEventSink(EventSink):
    """Forward formatted event text to an arbitrary callback."""

    def __init__(
        self,
        callback: Callable[[str], None],
        formatter: Callable[[PipelineEvent], str] | None = None,
    ) -> None:
        self.callback = callback
        self.formatter = formatter or format_event_for_openwebui

    def emit(self, event: PipelineEvent) -> None:
        self.callback(self.formatter(event))


class OpenWebUIEventSink(CallbackEventSink):
    """Thin Open WebUI sink using the standard markdown formatter."""

    def __init__(self, callback: Callable[[str], None]) -> None:
        super().__init__(callback, formatter=format_event_for_openwebui)

