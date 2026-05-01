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
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent_runtime.capabilities import build_default_registry
from agent_runtime.core.orchestrator import AgentRuntime
from agent_runtime.execution.engine import ExecutionEngine as AgentExecutionEngine
from agent_runtime.execution.result_store import InMemoryResultStore
from agent_runtime.llm import OpenAICompatLLMClient
from agent_runtime.output_pipeline.orchestrator import OutputPipelineOrchestrator
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


def _resolve_agent_gateway_settings(configured_settings: Settings) -> tuple[str, str | None]:
    """Resolve gateway node and URL for the agent runtime bridge.

    Used by:
        ``create_app`` when building the gateway-backed execution config for the
        schema-driven agent runtime.
    """

    default_node = str(configured_settings.resolved_default_node() or "localhost").strip() or "localhost"
    try:
        gateway_url = configured_settings.resolve_gateway_url(default_node)
    except ValueError:
        gateway_url = "http://127.0.0.1:8787" if default_node == "localhost" else None
    return default_node, gateway_url


def _allow_shell_execution_from_settings(configured_settings: Settings) -> bool:
    """Resolve whether read-only shell capabilities should be enabled in the agent runtime."""

    shell_mode = str(configured_settings.shell_mode or "read_only").strip().lower() or "read_only"
    return shell_mode != "disabled"


def create_app(settings: Settings | None = None, agent_runtime: AgentRuntime | None = None) -> FastAPI:
    """Create the echo FastAPI application.

    Used by:
        ``aor serve`` and tests.
    """

    configured_settings = settings or get_settings(config_path=os.getenv(APP_CONFIG_PATH_ENV) or None)
    engine = ExecutionEngine(configured_settings)
    runtime = agent_runtime
    if runtime is None:
        default_gateway_node, resolved_gateway_url = _resolve_agent_gateway_settings(configured_settings)
        registry = build_default_registry()
        result_store = InMemoryResultStore()
        execution_engine = AgentExecutionEngine(
            registry,
            {
                "workspace_root": str(configured_settings.workspace_root),
                "allow_shell_execution": _allow_shell_execution_from_settings(configured_settings),
                "allow_network_operations": False,
                "gateway_default_node": default_gateway_node,
                "gateway_url": resolved_gateway_url,
                "gateway_endpoints": dict(configured_settings.gateway_endpoints),
                "gateway_timeout_seconds": configured_settings.gateway_timeout_seconds,
                "max_output_preview_bytes": configured_settings.shell_max_output_chars,
                "max_rows_returned": max(1, int(configured_settings.sql_row_limit or 100)),
                "max_files_listed": 1000,
                "stop_on_error": True,
            },
            result_store,
        )
        llm_client = OpenAICompatLLMClient(
            base_url=configured_settings.llm_base_url,
            api_key=configured_settings.llm_api_key,
            model=configured_settings.default_model,
            timeout_seconds=configured_settings.llm_timeout_seconds,
            temperature=configured_settings.default_temperature,
        )
        runtime = AgentRuntime(
            llm_client=llm_client,
            registry=registry,
            execution_engine=execution_engine,
            output_orchestrator=OutputPipelineOrchestrator(),
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Keep the public lifespan hook without background worker cleanup."""

        yield

    app = FastAPI(title="OpenFABRIC Echo Runtime", version=__version__, lifespan=lifespan)
    app.state.engine = engine
    app.state.settings = configured_settings
    app.state.agent_runtime = runtime

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
        """Handle chat completions through the new typed agent runtime."""

        if not configured_settings.openai_compat_enabled:
            raise HTTPException(status_code=404, detail="OpenAI compatibility is disabled.")
        task = _messages_to_task(request.messages)
        if not task:
            raise HTTPException(status_code=400, detail="No actionable user task found.")

        created = int(time.time())
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        if not request.stream:
            content = runtime.handle_request(
                task,
                context={
                    "workspace_root": str(configured_settings.workspace_root),
                    "confirmation": False,
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
