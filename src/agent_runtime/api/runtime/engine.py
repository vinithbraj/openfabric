"""Compatibility runtime engine that routes legacy surfaces into AgentRuntime.

Purpose:
    Preserve older run/session/CLI interfaces while using the current typed
    agent runtime as the real implementation.

Responsibilities:
    Extract prompt text from compatibility payloads, invoke ``AgentRuntime``,
    store in-memory session state, and expose lightweight compatibility events.

Data flow / Interfaces:
    Called by the FastAPI compatibility endpoints and CLI. Inputs are plain
    dictionaries containing fields such as ``task`` or ``prompt``. Outputs are
    JSON-serializable session dictionaries with ``final_output.content`` equal
    to the rendered runtime response.

Boundaries:
    This layer preserves compatibility envelopes, but planning, safety,
    confirmation, execution, and rendering all happen inside ``agent_runtime``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any
from uuid import uuid4

from agent_runtime.capabilities import build_default_registry
from agent_runtime.core.orchestrator import AgentRuntime
from agent_runtime.execution.engine import ExecutionEngine as AgentExecutionEngine
from agent_runtime.execution.result_store import InMemoryResultStore
from agent_runtime.llm import OpenAICompatLLMClient
from agent_runtime.output_pipeline.orchestrator import OutputPipelineOrchestrator
from agent_runtime.api.config import Settings, get_settings


def _utc_now() -> str:
    """Return an ISO timestamp for run and event metadata."""

    return datetime.now(UTC).isoformat()


def extract_prompt(input_payload: dict[str, Any] | None) -> str:
    """Extract the user prompt from a compatibility input payload."""

    payload = dict(input_payload or {})
    for key in ("task", "prompt", "query", "message", "text", "input"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in payload.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(payload.get("task") or payload.get("prompt") or "").strip()


def _resolve_agent_gateway_settings(configured_settings: Settings) -> tuple[str, str | None]:
    """Resolve gateway node and URL for the agent runtime bridge."""

    default_node = str(configured_settings.resolved_default_node() or "localhost").strip() or "localhost"
    try:
        gateway_url = configured_settings.resolve_gateway_url(default_node)
    except ValueError:
        gateway_url = "http://127.0.0.1:8787" if default_node == "localhost" else None
    return default_node, gateway_url


def _allow_shell_execution_from_settings(configured_settings: Settings) -> bool:
    """Resolve whether read-only shell capabilities should be enabled."""

    shell_mode = str(configured_settings.shell_mode or "read_only").strip().lower() or "read_only"
    return shell_mode != "disabled"


def build_agent_runtime(settings: Settings) -> AgentRuntime:
    """Build the default typed agent runtime for compatibility surfaces."""

    default_gateway_node, resolved_gateway_url = _resolve_agent_gateway_settings(settings)
    registry = build_default_registry()
    result_store = InMemoryResultStore()
    execution_engine = AgentExecutionEngine(
        registry,
        {
            "workspace_root": str(settings.workspace_root),
            "allow_shell_execution": _allow_shell_execution_from_settings(settings),
            "allow_network_operations": False,
            "gateway_default_node": default_gateway_node,
            "gateway_url": resolved_gateway_url,
            "gateway_endpoints": dict(settings.gateway_endpoints),
            "gateway_timeout_seconds": settings.gateway_timeout_seconds,
            "max_output_preview_bytes": settings.shell_max_output_chars,
            "max_rows_returned": max(1, int(settings.sql_row_limit or 100)),
            "max_files_listed": 1000,
            "stop_on_error": True,
        },
        result_store,
    )
    llm_client = OpenAICompatLLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.default_model,
        timeout_seconds=settings.llm_timeout_seconds,
        temperature=settings.default_temperature,
    )
    return AgentRuntime(
        llm_client=llm_client,
        registry=registry,
        execution_engine=execution_engine,
        output_orchestrator=OutputPipelineOrchestrator(),
    )


@dataclass
class CompatibilityEventStore:
    """Store compatibility run/session events in memory."""

    _events: list[dict[str, Any]] = field(default_factory=list)
    _next_id: int = 1
    _lock: Lock = field(default_factory=Lock)

    def append_event(self, session_id: str, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Append an event and return the stored event dictionary."""

        with self._lock:
            event = {
                "id": self._next_id,
                "session_id": session_id,
                "node_name": "agent_runtime",
                "event_type": event_type,
                "payload": dict(payload or {}),
                "created_at": _utc_now(),
            }
            self._next_id += 1
            self._events.append(event)
            return dict(event)

    def get_events_after(self, session_id: str, after_id: int | None = None) -> list[dict[str, Any]]:
        """Return events for a session after an optional event id."""

        after = int(after_id or 0)
        with self._lock:
            return [
                dict(event)
                for event in self._events
                if event["session_id"] == session_id and int(event["id"]) > after
            ]


class ExecutionEngine:
    """Compatibility engine backed by the typed agent runtime."""

    def __init__(
        self,
        settings: Settings | None = None,
        agent_runtime: AgentRuntime | Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.runtime = agent_runtime or build_agent_runtime(self.settings)
        self.store = CompatibilityEventStore()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def _public_session(self, session: dict[str, Any]) -> dict[str, Any]:
        """Return one external-facing session payload without private runtime state."""

        return {key: value for key, value in dict(session).items() if not key.startswith("_")}

    def _runtime_context(
        self,
        *,
        confirmation: bool = False,
        event_callback=None,
    ) -> dict[str, Any]:
        """Build the runtime context used by compatibility routes."""

        context: dict[str, Any] = {
            "workspace_root": str(self.settings.workspace_root),
            "confirmation": confirmation,
            "allow_full_output_access": False,
        }
        if callable(event_callback):
            context["event_callback"] = event_callback
            context["observability"] = {"enabled": True, "debug": False}
        return context

    def _session_status_from_runtime(self) -> str:
        """Map runtime state into one compatibility session status."""

        summary = getattr(self.runtime, "last_failure_summary", None)
        if isinstance(summary, dict) and summary.get("category") == "confirmation_required":
            return "awaiting_confirmation"
        return "completed"

    def validate_spec(self, spec_path: str) -> dict[str, Any]:
        """Return a compatibility validation result for a runtime spec path."""

        return {
            "valid": True,
            "spec_path": str(spec_path),
            "mode": "agent_runtime",
            "message": "Compatibility routes are backed by the typed agent runtime.",
        }

    def create_session(self, spec_path: str, input_payload: dict[str, Any] | None, **_: Any) -> dict[str, Any]:
        """Create a pending compatibility session."""

        session_id = uuid4().hex
        prompt = extract_prompt(input_payload)
        session = {
            "id": session_id,
            "spec_path": str(spec_path),
            "status": "pending",
            "input": dict(input_payload or {}),
            "prompt": prompt,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "final_output": None,
            "latest_snapshot": {},
            "_planning_trace": None,
        }
        with self._lock:
            self._sessions[session_id] = session
        self.store.append_event(session_id, "run.created", {"mode": "agent_runtime"})
        return self._public_session(session)

    def resume_session(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        """Resume a compatibility session through the typed runtime."""

        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            session = self._sessions[session_id]
            current_status = str(session.get("status") or "")
            confirmation = bool(kwargs.get("approve_dangerous") or kwargs.get("confirmation"))
            if current_status == "completed":
                return self._public_session(session)
            if current_status == "awaiting_confirmation" and not confirmation:
                return self._public_session(session)
            prompt = str(session.get("prompt") or "")
            planning_trace = session.get("_planning_trace")

        def _event_callback(text: str) -> None:
            if text:
                self.store.append_event(session_id, "agent_runtime.trace", {"content": text})

        if confirmation and planning_trace is not None and hasattr(self.runtime, "replay_from_trace"):
            content = self.runtime.replay_from_trace(
                planning_trace,
                context=self._runtime_context(confirmation=True, event_callback=_event_callback),
            )
        else:
            content = self.runtime.handle_request(
                prompt,
                context=self._runtime_context(confirmation=confirmation, event_callback=_event_callback),
            )

        status = self._session_status_from_runtime()
        final_output = {"kind": "text", "content": str(content or "")}
        latest_snapshot = {
            "session_id": session_id,
            "status": status,
            "final_output": final_output,
            "mode": "agent_runtime",
        }

        with self._lock:
            session = self._sessions[session_id]
            session["status"] = status
            session["updated_at"] = _utc_now()
            session["final_output"] = final_output
            session["latest_snapshot"] = latest_snapshot
            session["_planning_trace"] = (
                getattr(self.runtime, "last_planning_trace", None)
                if status == "awaiting_confirmation"
                else None
            )
            public_session = self._public_session(session)

        if status == "awaiting_confirmation":
            self.store.append_event(
                session_id,
                "confirmation.required",
                {"final_output": final_output},
            )
            self.store.append_event(
                session_id,
                "finalize.awaiting_confirmation",
                {"final_output": final_output},
            )
        else:
            self.store.append_event(
                session_id,
                "agent_runtime.completed",
                {"char_count": len(final_output["content"])},
            )
            self.store.append_event(
                session_id,
                "finalize.completed",
                {"final_output": final_output},
            )
        return public_session

    def trigger_session(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        """Compatibility alias for ``resume_session``."""

        return self.resume_session(session_id, **kwargs)

    def run_spec(self, spec_path: str, input_payload: dict[str, Any] | None) -> dict[str, Any]:
        """Create and complete a compatibility run in one call."""

        session = self.create_session(spec_path, input_payload)
        return self.resume_session(str(session["id"]))

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent in-memory compatibility sessions."""

        with self._lock:
            sessions = list(self._sessions.values())[-int(limit or 50) :]
            return [self._public_session(session) for session in reversed(sessions)]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Return one compatibility session plus its events."""

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            payload = self._public_session(session)
        payload["events"] = self.store.get_events_after(session_id)
        return {"session": payload, "latest_snapshot": dict(payload.get("latest_snapshot") or {})}

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Compatibility wrapper around ``list_sessions``."""

        return self.list_sessions(limit=limit)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Compatibility wrapper around ``get_session``."""

        return self.get_session(run_id)
