"""FastAPI and OpenAI-compatible echo API for the V10 reset.

Purpose:
    Preserve the external HTTP/OpenWebUI interface while the internal runtime is
    rebuilt from scratch.

Responsibilities:
    Expose health, model listing, run/session compatibility endpoints, and
    OpenAI-compatible chat completions that echo the latest user prompt.

Data flow / Interfaces:
    Receives HTTP JSON payloads, extracts the user prompt, delegates to the
    echo ``ExecutionEngine``, and returns JSON or SSE responses.

Boundaries:
    This API performs no tool execution and forwards no prompts to an LLM. The
    returned assistant content is exactly the prompt text selected from the
    request.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from aor_runtime import __version__
from aor_runtime.app_config import APP_CONFIG_PATH_ENV
from aor_runtime.config import Settings, get_settings
from aor_runtime.runtime.engine import ExecutionEngine


class RunRequest(BaseModel):
    """Compatibility request for run and session endpoints."""

    spec_path: str = "examples/general_purpose_assistant.yaml"
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
        ``_messages_to_task`` when extracting the prompt to echo.
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


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the echo FastAPI application.

    Used by:
        ``aor serve`` and tests.
    """

    configured_settings = settings or get_settings(config_path=os.getenv(APP_CONFIG_PATH_ENV) or None)
    engine = ExecutionEngine(configured_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Keep the public lifespan hook without background worker cleanup."""

        yield

    app = FastAPI(title="OpenFABRIC Echo Runtime", version=__version__, lifespan=lifespan)
    app.state.engine = engine
    app.state.settings = configured_settings

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """Report API health."""

        return {"status": "ok", "mode": "echo"}

    @app.post("/compile")
    def compile_spec(request: ValidateRequest) -> dict[str, Any]:
        """Return a compatibility compile/validation response."""

        return engine.validate_spec(request.spec_path)

    @app.post("/sessions")
    def create_session(request: RunRequest, run_immediately: bool = True) -> dict[str, Any]:
        """Create a session and optionally complete it immediately."""

        session = engine.create_session(request.spec_path, request.input, trigger="manual")
        if not run_immediately:
            return session
        return engine.resume_session(str(session["id"]), trigger="manual")

    @app.get("/sessions")
    def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
        """List echo sessions."""

        return engine.list_sessions(limit=limit)

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        """Inspect one echo session."""

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
        """Complete a pending echo session."""

        try:
            return engine.trigger_session(session_id, trigger=request.trigger)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/resume")
    def resume_session(session_id: str, request: SessionTriggerRequest) -> dict[str, Any]:
        """Resume a pending echo session."""

        try:
            return engine.resume_session(session_id, trigger=request.trigger)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/runs")
    def create_run(request: RunRequest) -> dict[str, Any]:
        """Run one prompt through the echo engine."""

        return engine.run_spec(request.spec_path, request.input)

    @app.post("/runs/stream")
    def create_run_stream(request: RunRequest) -> StreamingResponse:
        """Stream compatibility run events for an echo run."""

        session = engine.create_session(request.spec_path, request.input)
        final_state = engine.resume_session(str(session["id"]))

        def event_stream():
            for event in engine.store.get_events_after(str(final_state["id"])):
                yield _format_sse(str(event["event_type"]), event)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/runs")
    def list_runs(limit: int = 50) -> list[dict[str, Any]]:
        """List echo runs."""

        return engine.list_runs(limit=limit)

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        """Inspect one echo run."""

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
        """Echo the latest user prompt in OpenAI-compatible chat format."""

        if not configured_settings.openai_compat_enabled:
            raise HTTPException(status_code=404, detail="OpenAI compatibility is disabled.")
        task = _messages_to_task(request.messages)
        if not task:
            raise HTTPException(status_code=400, detail="No actionable user task found.")

        created = int(time.time())
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        if not request.stream:
            return {
                "id": response_id,
                "object": "chat.completion",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": task},
                        "finish_reason": "stop",
                    }
                ],
            }

        def event_stream():
            yield _chat_chunk(response_id, request.model, created, {"role": "assistant"})
            if task:
                yield _chat_chunk(response_id, request.model, created, {"content": task})
            yield _chat_chunk(response_id, request.model, created, {}, finish_reason="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app
