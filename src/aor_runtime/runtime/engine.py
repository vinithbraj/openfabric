"""Minimal echo runtime engine for the V10 reset.

Purpose:
    Provide a deliberately small pass-through runtime while the next architecture
    is designed from a clean base.

Responsibilities:
    Extract prompt text from supported request payloads, echo it as the final
    answer, and keep a lightweight in-memory record of runs and events for API
    compatibility.

Data flow / Interfaces:
    Called by the FastAPI app and CLI. Inputs are plain dictionaries containing
    fields such as ``task`` or ``prompt``. Outputs are JSON-serializable session
    dictionaries with ``final_output.content`` equal to the extracted prompt.

Boundaries:
    This module performs no planning, no LLM calls, no tool invocation, and no
    filesystem/database/shell access beyond in-memory bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any
from uuid import uuid4

from aor_runtime.config import Settings, get_settings


def _utc_now() -> str:
    """Return an ISO timestamp for run and event metadata.

    Used by:
        EchoEngine event/session creation.
    """

    return datetime.now(UTC).isoformat()


def extract_prompt(input_payload: dict[str, Any] | None) -> str:
    """Extract the user prompt from a runtime input payload.

    Inputs:
        A dictionary that may contain ``task``, ``prompt``, ``query``,
        ``message``, ``text``, or ``input``.

    Returns:
        The first non-empty string value found, stripped of leading/trailing
        whitespace. Non-string values are converted to strings only as a final
        compatibility fallback.

    Used by:
        ExecutionEngine, CLI commands, and API run/chat endpoints.
    """

    payload = dict(input_payload or {})
    for key in ("task", "prompt", "query", "message", "text", "input"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in payload.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(payload.get("task") or payload.get("prompt") or "").strip()


@dataclass
class EchoEventStore:
    """Store run events in memory for the compatibility API.

    Used by:
        Session and run event endpoints exposed from ``aor_runtime.api.app``.
    """

    _events: list[dict[str, Any]] = field(default_factory=list)
    _next_id: int = 1
    _lock: Lock = field(default_factory=Lock)

    def append_event(self, session_id: str, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Append an event and return the stored event dictionary.

        Used by:
            ExecutionEngine when creating or completing echo sessions.
        """

        with self._lock:
            event = {
                "id": self._next_id,
                "session_id": session_id,
                "node_name": "echo",
                "event_type": event_type,
                "payload": dict(payload or {}),
                "created_at": _utc_now(),
            }
            self._next_id += 1
            self._events.append(event)
            return dict(event)

    def get_events_after(self, session_id: str, after_id: int | None = None) -> list[dict[str, Any]]:
        """Return events for a session after an optional event id.

        Used by:
            API polling and streaming endpoints.
        """

        after = int(after_id or 0)
        with self._lock:
            return [
                dict(event)
                for event in self._events
                if event["session_id"] == session_id and int(event["id"]) > after
            ]


class ExecutionEngine:
    """Run the reset runtime in echo-only mode.

    Used by:
        FastAPI and CLI entrypoints that need a stable engine object.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Create an engine with in-memory run/session state.

        Used by:
            ``create_app`` and CLI command handlers.
        """

        self.settings = settings or get_settings()
        self.store = EchoEventStore()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def validate_spec(self, spec_path: str) -> dict[str, Any]:
        """Return a compatibility validation result for a runtime spec path.

        Used by:
            ``POST /compile`` and CLI diagnostics.
        """

        return {
            "valid": True,
            "spec_path": str(spec_path),
            "mode": "echo",
            "message": "V10 reset runtime accepts requests and echoes the prompt.",
        }

    def create_session(self, spec_path: str, input_payload: dict[str, Any] | None, **_: Any) -> dict[str, Any]:
        """Create a pending echo session.

        Inputs:
            ``spec_path`` is retained for compatibility; ``input_payload`` holds
            the prompt-like value to echo.

        Returns:
            A session dictionary that can be resumed or inspected.

        Used by:
            API session/run endpoints and streaming compatibility paths.
        """

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
        }
        with self._lock:
            self._sessions[session_id] = session
        self.store.append_event(session_id, "run.created", {"mode": "echo"})
        return dict(session)

    def resume_session(self, session_id: str, **_: Any) -> dict[str, Any]:
        """Complete a pending session by echoing its prompt.

        Used by:
            API session trigger/resume endpoints and run helpers.
        """

        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            session = self._sessions[session_id]
            prompt = str(session.get("prompt") or "")
            final_output = {"kind": "text", "content": prompt}
            session["status"] = "completed"
            session["updated_at"] = _utc_now()
            session["final_output"] = final_output
            session["latest_snapshot"] = {
                "session_id": session_id,
                "status": "completed",
                "final_output": final_output,
                "mode": "echo",
            }
            completed = dict(session)
        self.store.append_event(session_id, "echo.completed", {"char_count": len(prompt)})
        self.store.append_event(session_id, "finalize.completed", {"final_output": final_output})
        return completed

    def trigger_session(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        """Compatibility alias for ``resume_session``.

        Used by:
            Existing API routes that expose session triggers.
        """

        return self.resume_session(session_id, **kwargs)

    def run_spec(self, spec_path: str, input_payload: dict[str, Any] | None) -> dict[str, Any]:
        """Create and complete an echo run in one call.

        Used by:
            ``POST /runs`` and non-streaming OpenAI chat completions.
        """

        session = self.create_session(spec_path, input_payload)
        return self.resume_session(str(session["id"]))

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent in-memory echo sessions.

        Used by:
            ``GET /sessions`` and ``GET /runs``.
        """

        with self._lock:
            sessions = list(self._sessions.values())[-int(limit or 50) :]
            return [dict(session) for session in reversed(sessions)]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Return one session plus its events.

        Used by:
            Session and run inspection endpoints.
        """

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            payload = dict(session)
        payload["events"] = self.store.get_events_after(session_id)
        return {"session": payload, "latest_snapshot": dict(payload.get("latest_snapshot") or {})}

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Compatibility wrapper around ``list_sessions``.

        Used by:
            ``GET /runs``.
        """

        return self.list_sessions(limit=limit)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Compatibility wrapper around ``get_session``.

        Used by:
            ``GET /runs/{run_id}``.
        """

        return self.get_session(run_id)
