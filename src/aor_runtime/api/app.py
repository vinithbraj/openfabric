"""OpenFABRIC Runtime Module: aor_runtime.api.app

Purpose:
    Host the FastAPI application and OpenAI-compatible chat API.

Responsibilities:
    Build app lifespan, route requests into ExecutionEngine, stream OpenWebUI-safe progress, and expose model metadata.

Data flow / Interfaces:
    Receives HTTP request models and emits JSON/SSE/OpenAI-compatible responses using runtime sessions and active run handles.

Boundaries:
    Cancels active runs on disconnect/shutdown and keeps raw internal payloads out of normal chat responses.
"""

from __future__ import annotations

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
from aor_runtime.model_identity import is_accepted_openai_compat_model_name
from aor_runtime.runtime.engine import ExecutionEngine
from aor_runtime.runtime.openwebui_trace import OpenWebUITraceRenderer


class RunRequest(BaseModel):
    """Represent run request within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by RunRequest.

    Data flow / Interfaces:
        Instances are created and consumed by API and OpenWebUI integration code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.api.app.RunRequest and related tests.
    """
    spec_path: str
    input: dict[str, Any] = Field(default_factory=dict)


class SessionTriggerRequest(BaseModel):
    """Represent session trigger request within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SessionTriggerRequest.

    Data flow / Interfaces:
        Instances are created and consumed by API and OpenWebUI integration code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.api.app.SessionTriggerRequest and related tests.
    """
    trigger: str = "manual"
    max_cycles: int | None = None
    approve_dangerous: bool = False


class ValidateRequest(BaseModel):
    """Represent validate request within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ValidateRequest.

    Data flow / Interfaces:
        Instances are created and consumed by API and OpenWebUI integration code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.api.app.ValidateRequest and related tests.
    """
    spec_path: str


class ChatMessage(BaseModel):
    """Represent chat message within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ChatMessage.

    Data flow / Interfaces:
        Instances are created and consumed by API and OpenWebUI integration code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.api.app.ChatMessage and related tests.
    """
    role: str
    content: Any


class ChatCompletionsRequest(BaseModel):
    """Represent chat completions request within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ChatCompletionsRequest.

    Data flow / Interfaces:
        Instances are created and consumed by API and OpenWebUI integration code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.api.app.ChatCompletionsRequest and related tests.
    """
    model: str
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False


def _format_sse(event_name: str, payload: dict[str, Any] | list[Any] | str) -> str:
    """Handle the internal format sse helper path for this module.

    Inputs:
        Receives event_name, payload for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app._format_sse.
    """
    if isinstance(payload, str):
        encoded = payload
    else:
        encoded = json_dumps(payload)
    return f"event: {event_name}\ndata: {encoded}\n\n"


def json_dumps(payload: Any) -> str:
    """Json dumps for the surrounding runtime workflow.

    Inputs:
        Receives payload for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.json_dumps.
    """
    import json

    return json.dumps(payload, default=str)


def _latest_session_status(engine: ExecutionEngine, session_id: str) -> tuple[str | None, dict[str, Any] | None]:
    """Handle the internal latest session status helper path for this module.

    Inputs:
        Receives engine, session_id for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app._latest_session_status.
    """
    payload = engine.get_session(session_id)
    if payload is None:
        return None, None
    session = payload.get("session") or {}
    return str(session.get("status") or ""), payload


def _messages_to_task(messages: list[ChatMessage]) -> str:
    """Handle the internal messages to task helper path for this module.

    Inputs:
        Receives messages for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app._messages_to_task.
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


def _message_content_to_text(content: Any) -> str:
    """Handle the internal message content to text helper path for this module.

    Inputs:
        Receives content for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app._message_content_to_text.
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
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def _is_openwebui_meta_prompt(text: str) -> bool:
    """Handle the internal is openwebui meta prompt helper path for this module.

    Inputs:
        Receives text for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app._is_openwebui_meta_prompt.
    """
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    has_chat_history = "<chat_history>" in lowered or "### chat history" in lowered
    if "### task:" in lowered and has_chat_history:
        return True
    meta_markers = (
        "suggest 3-5 relevant follow-up questions",
        "suggest relevant follow-up questions",
        "generate a concise title",
        "create a concise",
        "generate tags",
    )
    if has_chat_history and any(marker in lowered for marker in meta_markers):
        return True
    if lowered.startswith("### task:") and any(marker in lowered for marker in meta_markers):
        return True
    return False


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create app for the surrounding runtime workflow.

    Inputs:
        Receives settings for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.create_app.
    """
    configured_settings = settings or get_settings(config_path=os.getenv(APP_CONFIG_PATH_ENV) or None)
    engine = ExecutionEngine(configured_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Asynchronously lifespan for the surrounding runtime workflow.

        Inputs:
            Receives app for this function; type hints and validators define accepted shapes.

        Returns:
            Returns the awaited result for the asynchronous runtime operation.

        Used by:
            Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.lifespan.
        """
        try:
            yield
        finally:
            engine.active_runs.cancel_all("server shutdown", wait_seconds=configured_settings.shutdown_grace_seconds)

    app = FastAPI(title="OpenFABRIC", version=__version__, lifespan=lifespan)
    app.state.engine = engine
    app.state.settings = configured_settings
    app.state.active_runs = engine.active_runs

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """Healthz for the surrounding runtime workflow.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.healthz.
        """
        return {"status": "ok"}

    @app.post("/compile")
    def compile_spec(request: ValidateRequest) -> dict[str, Any]:
        """Compile spec for the surrounding runtime workflow.

        Inputs:
            Receives request for this function; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.compile_spec.
        """
        compiled = engine.validate_spec(request.spec_path)
        return compiled.model_dump()

    @app.post("/sessions")
    def create_session(request: RunRequest, run_immediately: bool = True) -> dict[str, Any]:
        """Create session for the surrounding runtime workflow.

        Inputs:
            Receives request, run_immediately for this function; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.create_session.
        """
        session = engine.create_session(request.spec_path, request.input, trigger="manual")
        if not run_immediately:
            return session
        return engine.resume_session(session["id"], trigger="manual")

    @app.get("/sessions/{session_id}/events")
    def get_session_events(session_id: str, after_id: int | None = None) -> list[dict[str, Any]]:
        """Get session events for the surrounding runtime workflow.

        Inputs:
            Receives session_id, after_id for this function; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.get_session_events.
        """
        payload = engine.get_session(session_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return engine.store.get_events_after(session_id, after_id=after_id)

    @app.get("/sessions/{session_id}/events/stream")
    def stream_session_events(session_id: str, after_id: int | None = None) -> StreamingResponse:
        """Stream session events for the surrounding runtime workflow.

        Inputs:
            Receives session_id, after_id for this function; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.stream_session_events.
        """
        payload = engine.get_session(session_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Session not found")

        def event_stream():
            """Event stream for the surrounding runtime workflow.

            Inputs:
                Uses module or instance state; no caller-supplied data parameters are required.

            Returns:
                Returns None; side effects are limited to the local runtime operation described above.

            Used by:
                Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.event_stream.
            """
            cursor = after_id
            while True:
                emitted = False
                for event in engine.store.get_events_after(session_id, after_id=cursor):
                    emitted = True
                    cursor = int(event["id"])
                    yield _format_sse(str(event["event_type"]), event)
                    if str(event.get("event_type") or "") == "finalize.completed":
                        return
                status, _ = _latest_session_status(engine, session_id)
                if status in {"completed", "failed"} and not emitted:
                    return
                time.sleep(0.05)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/sessions/{session_id}/trigger")
    def trigger_session(session_id: str, request: SessionTriggerRequest) -> dict[str, Any]:
        """Trigger session for the surrounding runtime workflow.

        Inputs:
            Receives session_id, request for this function; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.trigger_session.
        """
        try:
            return engine.trigger_session(
                session_id,
                trigger=request.trigger,
                max_cycles=request.max_cycles,
                approve_dangerous=request.approve_dangerous,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/resume")
    def resume_session(session_id: str, request: SessionTriggerRequest) -> dict[str, Any]:
        """Resume session for the surrounding runtime workflow.

        Inputs:
            Receives session_id, request for this function; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.resume_session.
        """
        try:
            return engine.resume_session(
                session_id,
                trigger=request.trigger,
                max_cycles=request.max_cycles,
                approve_dangerous=request.approve_dangerous,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/sessions")
    def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
        """List sessions for the surrounding runtime workflow.

        Inputs:
            Receives limit for this function; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.list_sessions.
        """
        return engine.list_sessions(limit=limit)

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        """Get session for the surrounding runtime workflow.

        Inputs:
            Receives session_id for this function; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.get_session.
        """
        payload = engine.get_session(session_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return payload

    @app.post("/runs")
    def create_run(request: RunRequest) -> dict[str, Any]:
        """Create run for the surrounding runtime workflow.

        Inputs:
            Receives request for this function; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.create_run.
        """
        final_state = engine.run_spec(request.spec_path, request.input)
        return final_state

    @app.post("/runs/stream")
    def create_run_stream(request: RunRequest) -> StreamingResponse:
        """Create run stream for the surrounding runtime workflow.

        Inputs:
            Receives request for this function; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.create_run_stream.
        """
        session = engine.create_session(
            request.spec_path,
            request.input,
            trigger="manual",
            stream_shell_output=True,
        )
        session_id = str(session["id"])
        handle = engine.active_runs.start_background(
            session_id,
            lambda token: engine.resume_session(
                session_id,
                trigger="manual",
                stream_shell_output=True,
                runtime_context=engine.create_invocation_context(token),
            ),
        )
        worker = handle.thread
        outcome = handle.outcome

        def event_stream():
            """Event stream for the surrounding runtime workflow.

            Inputs:
                Uses module or instance state; no caller-supplied data parameters are required.

            Returns:
                Returns None; side effects are limited to the local runtime operation described above.

            Used by:
                Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.event_stream.
            """
            cursor: int | None = None
            completed = False
            try:
                while True:
                    emitted = False
                    for event in engine.store.get_events_after(session_id, after_id=cursor):
                        emitted = True
                        cursor = int(event["id"])
                        yield _format_sse(str(event["event_type"]), event)
                        if str(event.get("event_type") or "") == "finalize.completed":
                            completed = True
                            if worker is not None:
                                worker.join(timeout=0)
                            return
                    if not handle.is_alive:
                        if "error" in outcome:
                            payload = {"error": str(outcome["error"])}
                            yield _format_sse("stream.error", payload)
                        status, _ = _latest_session_status(engine, session_id)
                        if status in {"completed", "failed"} and not emitted:
                            completed = True
                            return
                    time.sleep(0.05)
            finally:
                if not completed and handle.is_alive:
                    handle.cancel("client disconnected")
                    handle.join(timeout=configured_settings.worker_join_timeout_seconds)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/runs")
    def list_runs(limit: int = 50) -> list[dict[str, Any]]:
        """List runs for the surrounding runtime workflow.

        Inputs:
            Receives limit for this function; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.list_runs.
        """
        return engine.list_runs(limit=limit)

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        """Get run for the surrounding runtime workflow.

        Inputs:
            Receives run_id for this function; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.get_run.
        """
        payload = engine.get_run(run_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return payload

    @app.get("/v1/models")
    def openai_models() -> dict[str, Any]:
        """Openai models for the surrounding runtime workflow.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.openai_models.
        """
        if not configured_settings.openai_compat_enabled:
            raise HTTPException(status_code=404, detail="OpenAI compatibility is disabled.")
        created = int(time.time())
        model_name = configured_settings.openai_compat_model_name
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": created,
                    "owned_by": "openfabric",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def openai_chat_completions(request: ChatCompletionsRequest):
        """Openai chat completions for the surrounding runtime workflow.

        Inputs:
            Receives request for this function; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.openai_chat_completions.
        """
        if not configured_settings.openai_compat_enabled:
            raise HTTPException(status_code=404, detail="OpenAI compatibility is disabled.")
        if not is_accepted_openai_compat_model_name(request.model, configured_settings.openai_compat_model_name):
            raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")

        spec_path = configured_settings.resolve_openai_compat_spec_path()
        task = _messages_to_task(request.messages)
        if not task:
            raise HTTPException(status_code=400, detail="No actionable user task found.")

        if not request.stream:
            final_state = engine.run_spec(str(spec_path), {"task": task})
            content = str((final_state.get("final_output") or {}).get("content") or "")
            created = int(time.time())
            response_id = f"chatcmpl-{uuid.uuid4().hex}"
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

        session = engine.create_session(
            str(spec_path),
            {"task": task},
            trigger="manual",
            stream_shell_output=True,
        )
        session_id = str(session["id"])
        handle = engine.active_runs.start_background(
            session_id,
            lambda token: engine.resume_session(
                session_id,
                trigger="manual",
                stream_shell_output=True,
                runtime_context=engine.create_invocation_context(token),
            ),
        )
        worker = handle.thread
        outcome = handle.outcome
        created = int(time.time())
        response_id = f"chatcmpl-{session_id}"

        def chunk(delta: dict[str, Any], *, finish_reason: str | None = None) -> str:
            """Chunk for the surrounding runtime workflow.

            Inputs:
                Receives delta, finish_reason for this function; type hints and validators define accepted shapes.

            Returns:
                Returns the computed value described by the function name and type hints.

            Used by:
                Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.chunk.
            """
            payload = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            return f"data: {json_dumps(payload)}\n\n"

        def event_stream():
            """Event stream for the surrounding runtime workflow.

            Inputs:
                Uses module or instance state; no caller-supplied data parameters are required.

            Returns:
                Returns None; side effects are limited to the local runtime operation described above.

            Used by:
                Used by API and OpenWebUI integration code paths that import or call aor_runtime.api.app.event_stream.
            """
            cursor: int | None = None
            visible_output: list[str] = []
            trace_renderer = OpenWebUITraceRenderer.from_settings(configured_settings)
            completed = False
            try:
                yield chunk({"role": "assistant"})
                while True:
                    emitted = False
                    for event in engine.store.get_events_after(session_id, after_id=cursor):
                        emitted = True
                        cursor = int(event["id"])
                        text = trace_renderer.render(event)
                        if text:
                            visible_output.append(text)
                            yield chunk({"content": text})
                        if str(event.get("event_type") or "") == "finalize.completed":
                            payload = engine.get_session(session_id) or {}
                            latest = dict(payload.get("latest_snapshot") or {})
                            final_output = dict(latest.get("final_output") or {})
                            final_content = str(final_output.get("content") or "")
                            if final_content.strip() and final_content.strip() != "".join(visible_output).strip():
                                yield chunk({"content": final_content})
                            yield chunk({}, finish_reason="stop")
                            yield "data: [DONE]\n\n"
                            completed = True
                            if worker is not None:
                                worker.join(timeout=0)
                            return
                    if not handle.is_alive:
                        if "error" in outcome:
                            yield chunk({"content": f"Failed: {outcome['error']}"}, finish_reason="stop")
                            yield "data: [DONE]\n\n"
                            completed = True
                            return
                        status, payload = _latest_session_status(engine, session_id)
                        if status in {"completed", "failed"} and not emitted:
                            latest = dict((payload or {}).get("latest_snapshot") or {})
                            final_output = dict(latest.get("final_output") or {})
                            final_content = str(final_output.get("content") or "")
                            if final_content.strip() and final_content.strip() != "".join(visible_output).strip():
                                yield chunk({"content": final_content})
                            yield chunk({}, finish_reason="stop")
                            yield "data: [DONE]\n\n"
                            completed = True
                            return
                    time.sleep(0.05)
            finally:
                if not completed and handle.is_alive:
                    handle.cancel("client disconnected")
                    handle.join(timeout=configured_settings.worker_join_timeout_seconds)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app
