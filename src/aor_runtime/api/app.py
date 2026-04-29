from __future__ import annotations

import os
import threading
import time
import uuid
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
    spec_path: str
    input: dict[str, Any] = Field(default_factory=dict)


class SessionTriggerRequest(BaseModel):
    trigger: str = "manual"
    max_cycles: int | None = None
    approve_dangerous: bool = False


class ValidateRequest(BaseModel):
    spec_path: str


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False


def _format_sse(event_name: str, payload: dict[str, Any] | list[Any] | str) -> str:
    if isinstance(payload, str):
        encoded = payload
    else:
        encoded = json_dumps(payload)
    return f"event: {event_name}\ndata: {encoded}\n\n"


def json_dumps(payload: Any) -> str:
    import json

    return json.dumps(payload, default=str)


def _start_background(target):
    outcome: dict[str, Any] = {}

    def runner() -> None:
        try:
            outcome["result"] = target()
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return thread, outcome


def _latest_session_status(engine: ExecutionEngine, session_id: str) -> tuple[str | None, dict[str, Any] | None]:
    payload = engine.get_session(session_id)
    if payload is None:
        return None, None
    session = payload.get("session") or {}
    return str(session.get("status") or ""), payload


def _messages_to_task(messages: list[ChatMessage]) -> str:
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
    app = FastAPI(title="OpenFABRIC", version=__version__)
    configured_settings = settings or get_settings(config_path=os.getenv(APP_CONFIG_PATH_ENV) or None)
    engine = ExecutionEngine(configured_settings)
    app.state.engine = engine
    app.state.settings = configured_settings

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/compile")
    def compile_spec(request: ValidateRequest) -> dict[str, Any]:
        compiled = engine.validate_spec(request.spec_path)
        return compiled.model_dump()

    @app.post("/sessions")
    def create_session(request: RunRequest, run_immediately: bool = True) -> dict[str, Any]:
        session = engine.create_session(request.spec_path, request.input, trigger="manual")
        if not run_immediately:
            return session
        return engine.resume_session(session["id"], trigger="manual")

    @app.get("/sessions/{session_id}/events")
    def get_session_events(session_id: str, after_id: int | None = None) -> list[dict[str, Any]]:
        payload = engine.get_session(session_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return engine.store.get_events_after(session_id, after_id=after_id)

    @app.get("/sessions/{session_id}/events/stream")
    def stream_session_events(session_id: str, after_id: int | None = None) -> StreamingResponse:
        payload = engine.get_session(session_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Session not found")

        def event_stream():
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
        return engine.list_sessions(limit=limit)

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        payload = engine.get_session(session_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return payload

    @app.post("/runs")
    def create_run(request: RunRequest) -> dict[str, Any]:
        final_state = engine.run_spec(request.spec_path, request.input)
        return final_state

    @app.post("/runs/stream")
    def create_run_stream(request: RunRequest) -> StreamingResponse:
        session = engine.create_session(
            request.spec_path,
            request.input,
            trigger="manual",
            stream_shell_output=True,
        )
        session_id = str(session["id"])
        worker, outcome = _start_background(
            lambda: engine.resume_session(session_id, trigger="manual", stream_shell_output=True)
        )

        def event_stream():
            cursor: int | None = None
            while True:
                emitted = False
                for event in engine.store.get_events_after(session_id, after_id=cursor):
                    emitted = True
                    cursor = int(event["id"])
                    yield _format_sse(str(event["event_type"]), event)
                    if str(event.get("event_type") or "") == "finalize.completed":
                        worker.join(timeout=0)
                        return
                if not worker.is_alive():
                    if "error" in outcome:
                        payload = {"error": str(outcome["error"])}
                        yield _format_sse("stream.error", payload)
                    status, _ = _latest_session_status(engine, session_id)
                    if status in {"completed", "failed"} and not emitted:
                        return
                time.sleep(0.05)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/runs")
    def list_runs(limit: int = 50) -> list[dict[str, Any]]:
        return engine.list_runs(limit=limit)

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        payload = engine.get_run(run_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return payload

    @app.get("/v1/models")
    def openai_models() -> dict[str, Any]:
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
        worker, outcome = _start_background(
            lambda: engine.resume_session(session_id, trigger="manual", stream_shell_output=True)
        )
        created = int(time.time())
        response_id = f"chatcmpl-{session_id}"

        def chunk(delta: dict[str, Any], *, finish_reason: str | None = None) -> str:
            payload = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            return f"data: {json_dumps(payload)}\n\n"

        def event_stream():
            cursor: int | None = None
            visible_output: list[str] = []
            trace_renderer = OpenWebUITraceRenderer.from_settings(configured_settings)
            yield chunk({"role": "assistant"})
            while True:
                emitted = False
                for event in engine.store.get_events_after(session_id, after_id=cursor):
                    emitted = True
                    cursor = int(event["id"])
                    text = trace_renderer.render(event)
                    if text:
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
                        worker.join(timeout=0)
                        return
                if not worker.is_alive():
                    if "error" in outcome:
                        yield chunk({"content": f"Failed: {outcome['error']}"}, finish_reason="stop")
                        yield "data: [DONE]\n\n"
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
                        return
                time.sleep(0.05)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app
