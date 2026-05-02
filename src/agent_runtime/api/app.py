"""FastAPI and OpenAI-compatible API for the typed agent runtime.

Purpose:
    Expose the modern chat surface and preserve older compatibility routes such
    as ``/runs`` and ``/sessions``.

Responsibilities:
    Parse HTTP payloads, bootstrap the typed agent runtime, preserve
    compatibility envelopes, and manage the generic confirmation handshake.
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent_runtime.core.orchestrator import AgentRuntime
from agent_runtime.api import __version__
from agent_runtime.api.app_config import APP_CONFIG_PATH_ENV
from agent_runtime.api.config import Settings, get_settings
from agent_runtime.api.constants import DEFAULT_COMPAT_SPEC_PATH
from agent_runtime.api.runtime.engine import ExecutionEngine, build_agent_runtime


class RunRequest(BaseModel):
    """Compatibility request for run and session endpoints."""

    spec_path: str = DEFAULT_COMPAT_SPEC_PATH
    input: dict[str, Any] = Field(default_factory=dict)


class SessionTriggerRequest(BaseModel):
    """Compatibility request for triggering or resuming a session."""

    trigger: str = "manual"
    max_cycles: int | None = None
    approve_dangerous: bool = False


class ValidateRequest(BaseModel):
    """Compatibility request for the old compile endpoint."""

    spec_path: str


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""

    role: str
    content: Any


class ChatCompletionsRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False


@dataclass
class PendingConfirmation:
    confirmation_id: str
    trace: Any
    prompt: str
    workspace_root: str
    created_at: float


class PendingConfirmationStore:
    """In-memory store for pause/resume confirmation handshakes."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self.ttl_seconds = max(60, int(ttl_seconds))
        self._items: dict[str, PendingConfirmation] = {}
        self._lock = threading.Lock()

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [
            key
            for key, item in self._items.items()
            if (now - item.created_at) > self.ttl_seconds
        ]
        for key in expired:
            self._items.pop(key, None)

    def create(self, *, trace: Any, prompt: str, workspace_root: str) -> PendingConfirmation:
        with self._lock:
            self._purge_expired()
            confirmation_id = f"confirm-{uuid.uuid4().hex[:10]}"
            item = PendingConfirmation(
                confirmation_id=confirmation_id,
                trace=trace,
                prompt=prompt,
                workspace_root=workspace_root,
                created_at=time.time(),
            )
            self._items[confirmation_id] = item
            return item

    def get(self, confirmation_id: str) -> PendingConfirmation | None:
        with self._lock:
            self._purge_expired()
            return self._items.get(confirmation_id)

    def pop(self, confirmation_id: str) -> PendingConfirmation | None:
        with self._lock:
            self._purge_expired()
            return self._items.pop(confirmation_id, None)


def json_dumps(payload: Any) -> str:
    """Serialize API payloads for JSON and SSE responses.

    Used by:
        OpenAI-compatible streaming chunks and event-stream endpoints.
    """

    return json.dumps(payload, default=str, separators=(",", ":"))


def _format_sse(event_name: str, payload: dict[str, Any] | list[Any] | str) -> str:
    """Format a named server-sent event.

    Used by:
        Session and run streaming compatibility endpoints.
    """

    encoded = payload if isinstance(payload, str) else json_dumps(payload)
    return f"event: {event_name}\ndata: {encoded}\n\n"


def _message_content_to_text(content: Any) -> str:
    """Convert OpenAI message content shapes into plain text.

    Used by:
        ``_messages_to_task`` when extracting the actionable prompt text.
    """

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and str(item.get("type") or "") == "text":
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _is_openwebui_meta_prompt(text: str) -> bool:
    """Identify OpenWebUI title/tag/follow-up helper prompts.

    Used by:
        Prompt extraction so assistant metadata requests do not replace the
        actual user message being answered.
    """

    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    has_chat_history = "<chat_history>" in lowered or "### chat history" in lowered
    meta_markers = (
        "suggest 3-5 relevant follow-up questions",
        "suggest relevant follow-up questions",
        "generate a concise title",
        "create a concise",
        "generate tags",
    )
    return (has_chat_history or lowered.startswith("### task:")) and any(marker in lowered for marker in meta_markers)


def _messages_to_task(messages: list[ChatMessage]) -> str:
    """Select the latest actionable chat prompt.

    Used by:
        ``POST /v1/chat/completions``.
    """

    for message in reversed(messages):
        text = _message_content_to_text(message.content)
        if message.role == "user" and text and not _is_openwebui_meta_prompt(text):
            return text
    for message in reversed(messages):
        text = _message_content_to_text(message.content)
        if text and not _is_openwebui_meta_prompt(text):
            return text
    return ""


_CONFIRMATION_DIRECTIVE_RE = re.compile(
    r"^\s*(?P<action>approve|confirm|proceed|cancel|reject)\b(?:[\s:]+`?(?P<token>confirm-[a-z0-9]+)`?)?\s*$",
    re.IGNORECASE,
)
_CONFIRMATION_TOKEN_RE = re.compile(r"Confirmation ID:\s*`?(confirm-[a-z0-9]+)`?", re.IGNORECASE)


def _parse_confirmation_directive(text: str) -> tuple[str, str | None] | None:
    """Return a typed confirmation directive when the user message is approval-like."""

    match = _CONFIRMATION_DIRECTIVE_RE.match(str(text or "").strip())
    if not match:
        return None
    action = str(match.group("action") or "").strip().lower()
    token = str(match.group("token") or "").strip().lower() or None
    return action, token


def _extract_confirmation_token_from_messages(messages: list[ChatMessage]) -> str | None:
    """Extract the most recent confirmation id mentioned by the assistant."""

    for message in reversed(messages):
        if message.role != "assistant":
            continue
        text = _message_content_to_text(message.content)
        match = _CONFIRMATION_TOKEN_RE.search(text)
        if match:
            return str(match.group(1)).strip().lower()
    return None


def _augment_confirmation_content(content: str, confirmation_id: str) -> str:
    """Append generic approval instructions to a confirmation-required response."""

    base = str(content or "").strip() or "## Confirmation Required"
    return "\n\n".join(
        [
            base,
            f"Confirmation ID: `{confirmation_id}`",
            "Reply with `approve` to continue or `cancel` to stop.",
        ]
    )


def _confirmation_required_from_runtime(runtime: Any) -> bool:
    """Return whether the runtime ended in a confirmation-required pause state."""

    summary = getattr(runtime, "last_failure_summary", None)
    return isinstance(summary, dict) and summary.get("category") == "confirmation_required"


def _chat_chunk(response_id: str, model: str, created: int, delta: dict[str, Any], finish_reason: str | None = None) -> str:
    """Build one OpenAI-compatible streaming chunk.

    Used by:
        Streaming chat completions.
    """

    payload = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json_dumps(payload)}\n\n"


def create_app(settings: Settings | None = None, agent_runtime: AgentRuntime | None = None) -> FastAPI:
    """Create the FastAPI application.

    Used by:
        ``aor serve`` and tests.
    """

    configured_settings = settings or get_settings(config_path=os.getenv(APP_CONFIG_PATH_ENV) or None)
    runtime = agent_runtime
    pending_confirmations = PendingConfirmationStore()
    if runtime is None:
        runtime = build_agent_runtime(configured_settings)
    engine = ExecutionEngine(configured_settings, agent_runtime=runtime)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Keep the public lifespan hook without background worker cleanup."""

        yield

    app = FastAPI(title="OpenFABRIC Agent Runtime", version=__version__, lifespan=lifespan)
    app.state.engine = engine
    app.state.settings = configured_settings
    app.state.agent_runtime = runtime

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """Report API health."""

        return {"status": "ok", "mode": "agent_runtime"}

    @app.post("/compile")
    def compile_spec(request: ValidateRequest) -> dict[str, Any]:
        """Return a compatibility compile/validation response."""

        return engine.validate_spec(request.spec_path)

    @app.post("/sessions")
    def create_session(request: RunRequest, run_immediately: bool = True) -> dict[str, Any]:
        """Create a compatibility session and optionally complete it immediately."""

        session = engine.create_session(request.spec_path, request.input, trigger="manual")
        if not run_immediately:
            return session
        return engine.resume_session(str(session["id"]), trigger="manual")

    @app.get("/sessions")
    def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
        """List compatibility sessions."""

        return engine.list_sessions(limit=limit)

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        """Inspect one compatibility session."""

        payload = engine.get_session(session_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return payload

    @app.get("/sessions/{session_id}/events")
    def get_session_events(session_id: str, after_id: int | None = None) -> list[dict[str, Any]]:
        """Return session events after an optional cursor."""

        if engine.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return engine.store.get_events_after(session_id, after_id=after_id)

    @app.get("/sessions/{session_id}/events/stream")
    def stream_session_events(session_id: str, after_id: int | None = None) -> StreamingResponse:
        """Stream existing session events and stop at finalization."""

        if engine.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")

        def event_stream():
            cursor = after_id
            for event in engine.store.get_events_after(session_id, after_id=cursor):
                cursor = int(event["id"])
                yield _format_sse(str(event["event_type"]), event)
                if str(event.get("event_type") or "") == "finalize.completed":
                    return

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/sessions/{session_id}/trigger")
    def trigger_session(session_id: str, request: SessionTriggerRequest) -> dict[str, Any]:
        """Trigger a pending compatibility session."""

        try:
            return engine.trigger_session(
                session_id,
                trigger=request.trigger,
                approve_dangerous=request.approve_dangerous,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/resume")
    def resume_session(session_id: str, request: SessionTriggerRequest) -> dict[str, Any]:
        """Resume a pending compatibility session."""

        try:
            return engine.resume_session(
                session_id,
                trigger=request.trigger,
                approve_dangerous=request.approve_dangerous,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/runs")
    def create_run(request: RunRequest) -> dict[str, Any]:
        """Run one prompt through the typed agent runtime in a compatibility envelope."""

        return engine.run_spec(request.spec_path, request.input)

    @app.post("/runs/stream")
    def create_run_stream(request: RunRequest) -> StreamingResponse:
        """Stream compatibility run events for one typed-runtime run."""

        session = engine.create_session(request.spec_path, request.input)
        final_state = engine.resume_session(str(session["id"]))

        def event_stream():
            for event in engine.store.get_events_after(str(final_state["id"])):
                yield _format_sse(str(event["event_type"]), event)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/runs")
    def list_runs(limit: int = 50) -> list[dict[str, Any]]:
        """List compatibility runs."""

        return engine.list_runs(limit=limit)

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        """Inspect one compatibility run."""

        payload = engine.get_run(run_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return payload

    @app.get("/v1/models")
    def openai_models() -> dict[str, Any]:
        """Return the OpenAI-compatible model list."""

        if not configured_settings.openai_compat_enabled:
            raise HTTPException(status_code=404, detail="OpenAI compatibility is disabled.")
        created = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": configured_settings.openai_compat_model_name,
                    "object": "model",
                    "created": created,
                    "owned_by": "openfabric",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def openai_chat_completions(request: ChatCompletionsRequest):
        """Handle chat completions through the new typed agent runtime."""

        if not configured_settings.openai_compat_enabled:
            raise HTTPException(status_code=404, detail="OpenAI compatibility is disabled.")
        task = _messages_to_task(request.messages)
        if not task:
            raise HTTPException(status_code=400, detail="No actionable user task found.")

        created = int(time.time())
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        directive = _parse_confirmation_directive(task)
        directive_token = _extract_confirmation_token_from_messages(request.messages)
        pending = None
        if directive is not None:
            action, token = directive
            confirmation_id = token or directive_token
            if confirmation_id:
                pending = pending_confirmations.get(confirmation_id)
            if action in {"cancel", "reject"}:
                if confirmation_id:
                    pending_confirmations.pop(confirmation_id)
                content = (
                    "## Confirmation Cancelled\n\nI cancelled the pending confirmation-gated action."
                    if confirmation_id
                    else "I couldn't find a pending confirmation to cancel."
                )
                return {
                    "id": response_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                }
            if action in {"approve", "confirm", "proceed"} and confirmation_id and pending is not None and not request.stream:
                pending_confirmations.pop(confirmation_id)
                content = runtime.replay_from_trace(
                    pending.trace,
                    context={
                        "workspace_root": pending.workspace_root,
                        "confirmation": True,
                        "allow_full_output_access": False,
                    },
                )
                return {
                    "id": response_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                }
            if action in {"approve", "confirm", "proceed"} and (confirmation_id is None or pending is None) and not request.stream:
                content = "I couldn't find a pending confirmation to approve. Please run the original request again."
                return {
                    "id": response_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                }
        if not request.stream:
            content = runtime.handle_request(
                task,
                context={
                    "workspace_root": str(configured_settings.workspace_root),
                    "confirmation": False,
                    "allow_full_output_access": False,
                },
            )
            if _confirmation_required_from_runtime(runtime):
                trace = getattr(runtime, "last_planning_trace", None)
                if trace is not None:
                    pending_item = pending_confirmations.create(
                        trace=trace,
                        prompt=task,
                        workspace_root=str(configured_settings.workspace_root),
                    )
                    content = _augment_confirmation_content(content, pending_item.confirmation_id)
            return {
                "id": response_id,
                "object": "chat.completion",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            }

        def event_stream():
            yield _chat_chunk(response_id, request.model, created, {"role": "assistant"})
            event_queue: queue.Queue[str] = queue.Queue()
            worker_done = threading.Event()
            outcome: dict[str, str] = {}

            def _event_callback(text: str) -> None:
                if text:
                    event_queue.put(text)

            def _worker() -> None:
                try:
                    if directive is not None:
                        action, token = directive
                        confirmation_id = token or directive_token
                        pending_item = (
                            pending_confirmations.pop(confirmation_id)
                            if confirmation_id and action in {"approve", "confirm", "proceed"}
                            else None
                        )
                        if action in {"approve", "confirm", "proceed"} and confirmation_id and pending_item is not None:
                            outcome["content"] = runtime.replay_from_trace(
                                pending_item.trace,
                                context={
                                    "workspace_root": pending_item.workspace_root,
                                    "confirmation": True,
                                    "allow_full_output_access": False,
                                    "event_callback": _event_callback,
                                    "observability": {
                                        "enabled": True,
                                        "debug": False,
                                    },
                                },
                            )
                        elif action in {"cancel", "reject"}:
                            if confirmation_id:
                                pending_confirmations.pop(confirmation_id)
                            outcome["content"] = (
                                "## Confirmation Cancelled\n\nI cancelled the pending confirmation-gated action."
                                if confirmation_id
                                else "I couldn't find a pending confirmation to cancel."
                            )
                        elif action in {"approve", "confirm", "proceed"}:
                            outcome["content"] = (
                                "I couldn't find a pending confirmation to approve. Please run the original request again."
                            )
                        else:
                            outcome["content"] = runtime.handle_request(
                                task,
                                context={
                                    "workspace_root": str(configured_settings.workspace_root),
                                    "confirmation": False,
                                    "allow_full_output_access": False,
                                    "event_callback": _event_callback,
                                    "observability": {
                                        "enabled": True,
                                        "debug": False,
                                    },
                                },
                            )
                    else:
                        outcome["content"] = runtime.handle_request(
                            task,
                            context={
                                "workspace_root": str(configured_settings.workspace_root),
                                "confirmation": False,
                                "allow_full_output_access": False,
                                "event_callback": _event_callback,
                                "observability": {
                                    "enabled": True,
                                    "debug": False,
                                },
                            },
                        )
                    if directive is None and _confirmation_required_from_runtime(runtime):
                        trace = getattr(runtime, "last_planning_trace", None)
                        if trace is not None:
                            pending_item = pending_confirmations.create(
                                trace=trace,
                                prompt=task,
                                workspace_root=str(configured_settings.workspace_root),
                            )
                            outcome["content"] = _augment_confirmation_content(
                                outcome["content"],
                                pending_item.confirmation_id,
                            )
                except Exception:
                    outcome["content"] = "I hit an internal error while handling that request."
                finally:
                    worker_done.set()

            worker = threading.Thread(target=_worker, daemon=True)
            worker.start()

            while not worker_done.is_set() or not event_queue.empty():
                try:
                    payload = event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                yield _chat_chunk(response_id, request.model, created, {"content": payload})

            worker.join(timeout=0.1)
            content = outcome.get("content", "")
            if content:
                yield _chat_chunk(response_id, request.model, created, {"content": content})
            yield _chat_chunk(response_id, request.model, created, {}, finish_reason="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app
