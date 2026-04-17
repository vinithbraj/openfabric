import argparse
import json
import os
import queue
import threading
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests

from runtime.engine import Engine
from runtime.loader import load_spec
from runtime.semantic_validator import validate_semantics


DEFAULT_MODEL_ID = "openfabric-planner"
DEFAULT_SPEC_PATH = "agent_library/specs/ops_assistant_llm.yml"


class ChatMessage(BaseModel):
    role: str
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False


class PlannerGateway:
    def __init__(self, spec_path: str, timeout_seconds: float | None = None):
        spec = load_spec(spec_path)
        validate_semantics(spec)
        self.engine = Engine(spec, global_timeout_seconds=timeout_seconds)
        self.engine.setup()
        self.lock = threading.Lock()

    def shutdown(self):
        self.engine.shutdown()

    def ask(self, question: str, on_event=None) -> str:
        collected: list[tuple[str, dict]] = []
        original_emit = self.engine.emit

        def collecting_emit(event_name: str, payload: dict, depth: int = 0):
            collected.append((event_name, payload))
            if on_event is not None:
                on_event(event_name, payload, depth)
            return original_emit(event_name, payload, depth)

        with self.lock:
            self.engine.emit = collecting_emit
            try:
                self.engine.emit("user.ask", {"question": question})
            finally:
                self.engine.emit = original_emit

        for event_name, payload in reversed(collected):
            if event_name == "answer.final" and isinstance(payload, dict):
                answer = payload.get("answer")
                if isinstance(answer, str) and answer.strip():
                    return answer

        for event_name, payload in reversed(collected):
            if event_name == "task.result" and isinstance(payload, dict):
                detail = payload.get("detail")
                if isinstance(detail, str) and detail.strip():
                    return detail

        return "The planner finished without producing a final answer."


gateway: PlannerGateway | None = None
model_id = os.getenv("OPENFABRIC_GATEWAY_MODEL", DEFAULT_MODEL_ID)
app = FastAPI(title="OpenFabric Open WebUI Gateway")


def _llm_config():
    api_key = os.getenv("LLM_OPS_API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy"
    base_url = os.getenv("LLM_OPS_BASE_URL") or "http://127.0.0.1:8000/v1"
    model = os.getenv("LLM_OPS_MODEL") or "deepseek-ai/deepseek-coder-6.7b-instruct"
    timeout_seconds = float(os.getenv("LLM_OPS_TIMEOUT_SECONDS", "300"))
    return api_key, base_url.rstrip("/"), model, timeout_seconds


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _question_from_messages(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            text = _content_to_text(message.content).strip()
            if text:
                return text
    raise HTTPException(status_code=400, detail="No user message found.")


def _is_openwebui_auxiliary_prompt(question: str) -> bool:
    normalized = question.strip().lower()
    has_chat_history = any(
        marker in normalized
        for marker in (
            "### chat history:",
            "chat history:",
            "<chat_history>",
        )
    )
    has_json_output_instruction = any(
        marker in normalized
        for marker in (
            '"follow_ups"',
            '"title"',
            '"tags"',
            "{ \"follow_ups\"",
            "{ \"title\"",
            "{ \"tags\"",
        )
    )
    if not has_chat_history and not has_json_output_instruction:
        return False
    auxiliary_markers = [
        "follow-up questions",
        "follow_ups",
        "generate a concise",
        "word title",
        "generate 1-3 broad tags",
        "subtopic tags",
        "categorizing the main themes",
    ]
    return any(marker in normalized for marker in auxiliary_markers)


def _is_openwebui_followup_prompt(question: str) -> bool:
    normalized = question.strip().lower()
    if '"follow_ups"' not in normalized and "follow_ups" not in normalized:
        return False
    return any(
        marker in normalized
        for marker in (
            "follow-up questions",
            "follow-up prompts",
            "follow_ups",
            "suggest 3-5 relevant",
        )
    )


def _upstream_chat_completion(request: ChatCompletionRequest):
    api_key, base_url, upstream_model, timeout_seconds = _llm_config()
    payload = request.model_dump()
    payload["model"] = upstream_model

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout_seconds,
            stream=request.stream,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Upstream LLM request failed: {exc}") from exc

    if request.stream:
        return StreamingResponse(
            response.iter_content(chunk_size=None),
            media_type=response.headers.get("content-type", "text/event-stream"),
        )
    return response.json()


def _chat_response(answer: str, model: str):
    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _stream_chunk(completion_id: str, created: int, model: str, content: str):
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": {"content": content}, "finish_reason": None}
        ],
    }


def _truncate_progress(value: str, limit: int = 600) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _trace_block(lines: list[str], *, separator: bool = True) -> str:
    clean_lines = [line.rstrip() for line in lines if line is not None]
    suffix = "\n\n---\n\n" if separator else "\n\n"
    return "\n".join(clean_lines).rstrip() + suffix


def _agent_label(agent: Any) -> str:
    return f"`{agent}`" if isinstance(agent, str) and agent else "`agent`"


def _task_label(task: Any) -> str:
    return str(task).strip() if isinstance(task, str) and task.strip() else "<no task>"


def _duration_label(value: Any) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    if value < 1000:
        return f"{value:.0f} ms"
    return f"{value / 1000:.2f} s"


def _format_stats(stats: Any) -> str | None:
    if not isinstance(stats, dict):
        return None
    labels = {
        "connect_ms": "connect",
        "schema_ms": "schema",
        "sql_generation_ms": "generate SQL",
        "query_ms": "query",
        "total_ms": "SQL total",
    }
    parts = []
    for key, label in labels.items():
        duration = _duration_label(stats.get(key))
        if duration:
            parts.append(f"{label}: `{duration}`")
    return ", ".join(parts) if parts else None


def _compact_fields(*items: tuple[str, Any]) -> str:
    parts = []
    for label, value in items:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        parts.append(f"**{label}:** {value}")
    return " · ".join(parts)


def _format_progress(event_name: str, payload: dict, depth: int) -> str | None:
    if event_name == "user.ask":
        return _trace_block(["**Planning workflow...**"], separator=False)
    if event_name == "plan.progress":
        lines = [
            f"**Plan:** {payload.get('message', 'Plan ready.')}".rstrip(),
        ]
        steps = payload.get("steps", [])
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                task = step.get("task")
                if not isinstance(task, str) or not task.strip():
                    continue
                target = step.get("target_agent")
                lines.append(
                    f"- `{step.get('id', 'step')}` via {_agent_label(target)}: {_task_label(task)}"
                )
                command = step.get("command")
                if isinstance(command, str) and command.strip():
                    lines.append(f"  Command: `{_truncate_progress(command, 220)}`")
        return _trace_block(lines)
    if event_name == "task.plan":
        step_id = payload.get("step_id")
        target = payload.get("target_agent")
        task = payload.get("task", "")
        if step_id or target:
            return None
        return _trace_block([f"**Planning:** {task}"])
    if event_name == "step.progress":
        stage = payload.get("stage", "")
        step_id = payload.get("step_id", "step")
        target = payload.get("target_agent")
        task = payload.get("task", "")
        if stage == "started":
            lines = [
                f"**Running `{step_id}`** · {_agent_label(target)}",
                f"Task: {_task_label(task)}",
            ]
            command = payload.get("command")
            if isinstance(command, str) and command.strip():
                lines.append(f"Command: `{_truncate_progress(command, 260)}`")
            sql = payload.get("sql")
            if isinstance(sql, str) and sql.strip():
                lines.append(f"SQL: `{_truncate_progress(sql, 300)}`")
            return _trace_block(lines)
        if stage == "completed":
            lines = [f"**Completed `{step_id}`** · {_agent_label(target)}"]
            duration = _duration_label(payload.get("duration_ms"))
            result = payload.get("result")
            sql = payload.get("sql")
            row_count = None
            stats_line = None
            if isinstance(result, dict):
                detail = result.get("detail")
                if isinstance(detail, str) and detail.strip():
                    lines.append(f"Detail: `{_truncate_progress(detail, 260)}`")
                command = result.get("command")
                if isinstance(command, str) and command.strip():
                    lines.append(f"Command: `{_truncate_progress(command, 260)}`")
                stdout = result.get("stdout")
                if isinstance(stdout, str) and stdout.strip():
                    lines.append(f"Output: `{_truncate_progress(stdout, 320)}`")
                stderr = result.get("stderr")
                if isinstance(stderr, str) and stderr.strip():
                    lines.append(f"Stderr: `{_truncate_progress(stderr, 220)}`")
                row_count = result.get("row_count")
                stats_line = _format_stats(result.get("stats"))
                nested_result = result.get("result")
                if not stats_line and isinstance(nested_result, dict):
                    stats_line = _format_stats(nested_result.get("stats"))
                if not sql:
                    sql = result.get("sql")
            summary = _compact_fields(
                ("Duration", f"`{duration}`" if duration else None),
                ("Rows", f"`{row_count}`" if row_count is not None else None),
            )
            if summary:
                lines.append(summary)
            if stats_line:
                lines.append(f"**Timing:** {stats_line}")
            if isinstance(sql, str) and sql.strip():
                lines.append(f"**SQL:** `{_truncate_progress(sql, 360)}`")
            return _trace_block(lines)
        if stage == "failed":
            error = payload.get("error") or payload.get("message") or "Step failed."
            lines = [
                f"**Failed `{step_id}`** · {_agent_label(target)}",
                f"**Error:** {error}",
            ]
            duration = _duration_label(payload.get("duration_ms"))
            if duration:
                lines.append(f"**Duration:** `{duration}`")
            result = payload.get("result")
            if isinstance(result, dict):
                detail = result.get("detail")
                if isinstance(detail, str) and detail.strip():
                    lines.append(f"**Detail:** `{_truncate_progress(detail, 260)}`")
                stats_line = _format_stats(result.get("stats"))
                nested_result = result.get("result")
                if not stats_line and isinstance(nested_result, dict):
                    stats_line = _format_stats(nested_result.get("stats"))
                if stats_line:
                    lines.append(f"**Timing:** {stats_line}")
            return _trace_block(lines)
    if event_name == "shell.result":
        command = payload.get("command", "")
        returncode = payload.get("returncode")
        stdout = payload.get("stdout", "")
        lines = [f"**Shell command completed** · exit `{returncode}`"]
        if isinstance(command, str) and command.strip():
            lines.append(f"Command: `{_truncate_progress(command, 220)}`")
        if isinstance(stdout, str) and stdout.strip():
            lines.append(f"Output: `{_truncate_progress(stdout, 300)}`")
        return _trace_block(lines)
    if event_name == "sql.result":
        sql = payload.get("sql", "")
        result = payload.get("result")
        row_count = None
        if isinstance(result, dict):
            row_count = result.get("row_count")
        lines = [
            f"**SQL query completed**"
            + (f" with `{row_count}` row(s)." if row_count is not None else ".")
        ]
        if isinstance(sql, str) and sql.strip():
            lines.append(f"SQL: `{_truncate_progress(sql, 300)}`")
        stats_line = _format_stats(payload.get("stats"))
        if stats_line:
            lines.append(f"Timing: {stats_line}")
        return _trace_block(lines)
    if event_name in {"file.content", "notify.result"}:
        return _trace_block([f"Completed {event_name}."])
    if event_name == "task.result":
        return None
    if event_name == "workflow.result":
        status = payload.get("status", "unknown")
        return _trace_block([f"**Workflow {status}.**"], separator=False)
    return None


def _stream_response(question: str, model: str):
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    events: queue.Queue[dict | None] = queue.Queue()

    def event(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=True)}\n\n"

    def run_planner():
        try:
            if gateway is None:
                raise RuntimeError("Planner gateway is not ready.")

            def on_event(event_name: str, payload: dict, depth: int):
                progress = _format_progress(event_name, payload, depth)
                if progress:
                    events.put(_stream_chunk(completion_id, created, model, progress))

            answer = gateway.ask(question, on_event=on_event)
            events.put(_stream_chunk(completion_id, created, model, answer))
        except Exception as exc:
            events.put(_stream_chunk(completion_id, created, model, f"Error: {type(exc).__name__}: {exc}"))
        finally:
            events.put(None)

    def generate():
        worker = threading.Thread(target=run_planner, daemon=True)
        worker.start()
        yield event(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
        )
        while True:
            item = events.get()
            if item is None:
                break
            yield event(item)
        yield event(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            }
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def _static_stream_response(answer: str, model: str):
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def event(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=True)}\n\n"

    def generate():
        yield event(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
        )
        if answer:
            yield event(_stream_chunk(completion_id, created, model, answer))
        yield event(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            }
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok", "model": model_id}


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "openfabric",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if gateway is None:
        raise HTTPException(status_code=503, detail="Planner gateway is not ready.")

    question = _question_from_messages(request.messages)
    selected_model = request.model or model_id

    if _is_openwebui_followup_prompt(question):
        empty_followups = '{"follow_ups":[]}'
        if request.stream:
            return _static_stream_response(empty_followups, selected_model)
        return _chat_response(empty_followups, selected_model)

    if _is_openwebui_auxiliary_prompt(question):
        return _upstream_chat_completion(request)

    if request.stream:
        return _stream_response(question, selected_model)
    answer = gateway.ask(question)
    return _chat_response(answer, selected_model)


def create_app(spec_path: str, timeout_seconds: float | None = None):
    global gateway
    gateway = PlannerGateway(spec_path, timeout_seconds=timeout_seconds)

    @app.on_event("shutdown")
    def shutdown_gateway():
        if gateway is not None:
            gateway.shutdown()

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=DEFAULT_SPEC_PATH)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8310)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    uvicorn.run(
        create_app(args.spec, timeout_seconds=args.timeout),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
