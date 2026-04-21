import argparse
import asyncio
import copy
import json
import os
import queue
import re
import threading
import time
import uuid
from typing import Any, Iterator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests

from agent_library.common import EventRequest, shared_llm_api_settings
from agent_library.agents.synthesizer import _build_prompt as _build_synthesis_prompt
from agent_library.agents.synthesizer import _fallback_answer as _fallback_synthesis_answer
from runtime.engine import Engine
from runtime.loader import load_spec
from runtime.semantic_validator import validate_semantics


DEFAULT_MODEL_ID = "openfabric-planner"
DEFAULT_SPEC_PATH = "agent_library/specs/ops_assistant_llm.yml"
SYNTHESIZER_AGENT_NAME = "synthesizer"
SYNTHESIZER_EVENTS = {
    "research.result",
    "task.result",
    "file.content",
    "shell.result",
    "sql.result",
    "notify.result",
    "clarification.required",
    "workflow.result",
}


class ClientCancelled(Exception):
    pass


class ChatMessage(BaseModel):
    role: str
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False


class PlannerGateway:
    def __init__(self, spec_path: str, timeout_seconds: float | None = None, selected_agents: list[str] | None = None):
        spec = load_spec(spec_path, selected_agents=selected_agents)
        validate_semantics(spec)
        self.engine = Engine(spec, global_timeout_seconds=timeout_seconds)
        self.engine.setup()
        self.lock = threading.Lock()

    def shutdown(self):
        self.engine.shutdown()

    def ask(self, question: str, on_event=None, disable_synthesizer: bool = False, should_cancel=None) -> str:
        collected: list[tuple[str, dict]] = []
        original_emit = self.engine.emit
        disabled_subscribers: dict[str, list[str]] = {}

        def collecting_emit(event_name: str, payload: dict, depth: int = 0):
            if should_cancel is not None and should_cancel():
                raise ClientCancelled("Client disconnected.")
            collected.append((event_name, payload))
            if on_event is not None:
                on_event(event_name, payload, depth)
            if should_cancel is not None and should_cancel():
                raise ClientCancelled("Client disconnected.")
            return original_emit(event_name, payload, depth)

        with self.lock:
            self.engine.emit = collecting_emit
            if disable_synthesizer:
                for event_name in SYNTHESIZER_EVENTS:
                    subscribers = list(self.engine.bus.subscribers.get(event_name, []))
                    if SYNTHESIZER_AGENT_NAME in subscribers:
                        disabled_subscribers[event_name] = subscribers
                        self.engine.bus.subscribers[event_name] = [
                            agent_name
                            for agent_name in subscribers
                            if agent_name != SYNTHESIZER_AGENT_NAME
                        ]
            try:
                if should_cancel is not None and should_cancel():
                    raise ClientCancelled("Client disconnected.")
                self.engine.emit("user.ask", {"question": question})
            finally:
                self.engine.emit = original_emit
                for event_name, subscribers in disabled_subscribers.items():
                    self.engine.bus.subscribers[event_name] = subscribers

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
    api_key, base_url, timeout_seconds, model = shared_llm_api_settings("gpt-4o-mini")
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


NEW_TASK_PATTERN = re.compile(r"^\s*(?:new\s+task\s*:|nt[;:])\s*", re.IGNORECASE)


def _is_new_task_request(question: str) -> bool:
    return bool(NEW_TASK_PATTERN.match(question))


def _strip_new_task_prefix(question: str) -> str:
    if not _is_new_task_request(question):
        return question
    stripped = NEW_TASK_PATTERN.sub("", question, count=1).strip()
    return stripped or question.strip()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _webui_context_enabled() -> bool:
    return _env_flag("ENABLE_CONTEXT", default=False)


def _compact_history_text(text: str, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _compact_assistant_history_text(text: str, limit: int = 900) -> str:
    text = re.sub(
        r"\*\*Planning workflow\.\.\.\*\*.*?\*\*Workflow\s+(?:completed|failed)\.\*\*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"\*\*(?:Plan|Running|Completed|Failed)[^*]*\*\*.*?(?=\n\n|$)", "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "Previous request completed."
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _latest_context_boundary(messages: list[ChatMessage]) -> int:
    boundary = 0
    for index, message in enumerate(messages[:-1]):
        if message.role != "user":
            continue
        text = _content_to_text(message.content).strip()
        if _is_new_task_request(text):
            boundary = index
    return boundary


def _planner_question_from_messages(messages: list[ChatMessage]) -> str:
    latest = _question_from_messages(messages)
    current = _strip_new_task_prefix(latest)
    if not _webui_context_enabled():
        return current
    if _is_new_task_request(latest):
        return current

    history: list[str] = []
    for message in messages[_latest_context_boundary(messages) : -1]:
        if message.role not in {"user", "assistant"}:
            continue
        text = _content_to_text(message.content).strip()
        if not text:
            continue
        if _is_openwebui_auxiliary_prompt(text) or _is_openwebui_followup_prompt(text):
            continue
        text = _strip_new_task_prefix(text)
        role = "User" if message.role == "user" else "Assistant"
        compact_text = _compact_assistant_history_text(text) if message.role == "assistant" else _compact_history_text(text)
        history.append(f"{role}: {compact_text}")

    history = history[-8:]
    if not history:
        return current

    return (
        "Use the recent conversation only to resolve references and follow-up requests. "
        "Execute only the current user request.\n\n"
        "Recent conversation:\n"
        + "\n".join(history)
        + "\n\nCurrent user request:\n"
        + current
    )


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


def _upstream_chat_completion(request: ChatCompletionRequest, client_request: Request):
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
        cancel_event = threading.Event()
        chunks: queue.Queue[bytes | None] = queue.Queue()

        def read_upstream():
            try:
                for chunk in response.iter_content(chunk_size=None):
                    if cancel_event.is_set():
                        break
                    if chunk:
                        chunks.put(chunk)
            finally:
                response.close()
                chunks.put(None)

        async def generate():
            reader = threading.Thread(target=read_upstream, daemon=True)
            reader.start()
            disconnected = False
            try:
                while True:
                    if await client_request.is_disconnected():
                        disconnected = True
                        cancel_event.set()
                        response.close()
                        break
                    try:
                        item = chunks.get(timeout=0.1)
                    except queue.Empty:
                        await asyncio.sleep(0.05)
                        continue
                    if item is None:
                        break
                    yield item
            finally:
                cancel_event.set()
                response.close()
            if disconnected:
                return

        return StreamingResponse(
            generate(),
            media_type=response.headers.get("content-type", "text/event-stream"),
            headers=_streaming_headers(),
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


def _stream_role_chunk(completion_id: str, created: int, model: str):
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    }


def _stream_stop_chunk(completion_id: str, created: int, model: str):
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": {}, "finish_reason": "stop"}
        ],
    }


def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=True)}\n\n"


def _streaming_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


def _stream_progress_enabled() -> bool:
    return _env_flag("OPENFABRIC_GATEWAY_STREAM_PROGRESS", default=True)


def _text_stream_parts(text: str, chunk_size: int = 96) -> Iterator[str]:
    if not text:
        return

    buffer = ""
    for part in re.findall(r"\S+\s*|\s+", text):
        if len(buffer) + len(part) > chunk_size and buffer:
            yield buffer
            buffer = part
        else:
            buffer += part

        while len(buffer) > chunk_size * 2:
            yield buffer[:chunk_size]
            buffer = buffer[chunk_size:]

    if buffer:
        yield buffer


def _synthesis_llm_config():
    api_key, base_url, timeout_seconds, model = shared_llm_api_settings("gpt-4o-mini")
    return api_key, base_url.rstrip("/"), model, timeout_seconds


def _shell_summary_payload(payload: dict, include_excerpts: bool = True) -> dict:
    command = payload.get("command")
    returncode = payload.get("returncode")
    stdout = _clean_terminal_output(payload.get("stdout", ""))
    stderr = _clean_terminal_output(payload.get("stderr", ""))
    summary = {
        "command": command,
        "returncode": returncode,
        "status": "success" if returncode == 0 else "failed",
        "stdout_available": bool(stdout),
        "stderr_available": bool(stderr),
    }
    if include_excerpts and stdout:
        summary["stdout_excerpt"] = _truncate_progress(stdout, 500)
    if include_excerpts and stderr:
        summary["stderr_excerpt"] = _truncate_progress(stderr, 300)
    return summary


def _payload_for_synthesis(event_name: str, payload: dict) -> dict:
    sanitized = copy.deepcopy(payload)
    presentation = sanitized.get("presentation")
    presentation_format = presentation.get("format") if isinstance(presentation, dict) else None
    include_shell_excerpts = presentation_format != "plain"
    if event_name == "shell.result":
        return _shell_summary_payload(sanitized, include_excerpts=include_shell_excerpts)
    if event_name != "workflow.result":
        return sanitized

    steps = sanitized.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict) or step.get("event") != "shell.result":
                continue
            step_payload = step.get("payload")
            if isinstance(step_payload, dict):
                summary = _shell_summary_payload(
                    step_payload,
                    include_excerpts=include_shell_excerpts,
                )
                step["payload"] = summary
                step["result"] = summary
                step["emitted"] = []
    if isinstance(sanitized.get("result"), str):
        sanitized["result"] = "<shell output omitted; appended after the final answer>"
    return sanitized


def _gateway_step_field(step: dict[str, Any], field: str) -> Any:
    if field in step:
        return step.get(field)
    payload = step.get("payload")
    if isinstance(payload, dict) and field in payload:
        return payload.get(field)
    evidence = step.get("evidence")
    if isinstance(evidence, dict):
        evidence_payload = evidence.get("payload")
        if isinstance(evidence_payload, dict) and field in evidence_payload:
            return evidence_payload.get(field)
    return None


def _flatten_gateway_steps(steps: Any) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    if not isinstance(steps, list):
        return flattened
    for step in steps:
        if not isinstance(step, dict):
            continue
        flattened.append(step)
        flattened.extend(_flatten_gateway_steps(step.get("steps")))
    return flattened


def _workflow_has_grounded_shell_output(payload: dict[str, Any]) -> bool:
    for step in _flatten_gateway_steps(payload.get("steps")):
        if step.get("status") != "completed":
            continue
        event = step.get("event")
        if event not in {"shell.result", "slurm.result", "file.content"}:
            continue
        stdout = (
            _gateway_step_field(step, "stdout")
            or _gateway_step_field(step, "stdout_excerpt")
            or _gateway_step_field(step, "content")
        )
        if isinstance(stdout, str) and stdout.strip():
            return True
    return False


def _workflow_requests_internal_steps(payload: dict[str, Any]) -> bool:
    presentation = payload.get("presentation")
    return isinstance(presentation, dict) and bool(presentation.get("include_internal_steps"))


def _should_use_gateway_direct_fallback(event_name: str, payload: dict) -> bool:
    if event_name in {"task.result", "clarification.required", "shell.result", "file.content"}:
        return True
    if event_name == "workflow.result":
        return _workflow_has_grounded_shell_output(payload) or _workflow_requests_internal_steps(payload)
    return False


def _stream_synthesis_parts(
    event_name: str,
    payload: dict,
    cancel_event: threading.Event | None = None,
    response_holder: dict[str, Any] | None = None,
) -> Iterator[str]:
    raw_req = EventRequest(event=event_name, payload=copy.deepcopy(payload))
    req = EventRequest(event=event_name, payload=_payload_for_synthesis(event_name, payload))
    if cancel_event is not None and cancel_event.is_set():
        return
    if _should_use_gateway_direct_fallback(event_name, payload):
        direct_answer = _fallback_synthesis_answer(raw_req)
        if isinstance(direct_answer, str) and direct_answer.strip():
            yield from _text_stream_parts(direct_answer.strip(), chunk_size=32)
            return

    api_key, base_url, synthesis_model, timeout_seconds = _synthesis_llm_config()
    prompt = (
        _build_synthesis_prompt(req)
        + "\n\nGateway streaming requirements:\n"
        "- Do not include planner progress, task/status/result tables, or workflow traces.\n"
        "- Do not include a 'How it was done' section unless the user explicitly requested implementation details.\n"
        "- Do not include a 'SQL used' section unless SQL queries are present in the source event.\n"
        "- Do not include raw command/stdout/stderr blocks; the gateway appends those separately in a command inspection section.\n"
        "- For shell workflows, give only the concise user-facing outcome; do not echo stdout_excerpt or stderr_excerpt verbatim.\n"
        "- Never output HTML collapsible-section tags.\n"
    )
    yielded = False
    response = None

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": synthesis_model,
                "messages": [
                    {"role": "system", "content": "You create polished final answers for users."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "stream": True,
            },
            timeout=timeout_seconds,
            stream=True,
        )
        if response_holder is not None:
            response_holder["response"] = response
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if cancel_event is not None and cancel_event.is_set():
                response.close()
                return
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices")
            if not isinstance(choices, list):
                continue
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                content = delta.get("content")
                if isinstance(content, str) and content:
                    yielded = True
                    yield content
        if yielded:
            return
    except Exception as exc:
        import traceback
        if not yielded:
            # Only log when we haven't streamed anything yet — if we partially
            # yielded and then hit an error the client already received data.
            print(
                f"[gateway] synthesis stream error (falling back to local answer): "
                f"{exc!r}\n{traceback.format_exc()}",
                flush=True,
            )
        if yielded:
            return
        if cancel_event is not None and cancel_event.is_set():
            return
    finally:
        if response_holder is not None:
            response_holder["response"] = None
        try:
            response.close()
        except Exception:
            pass

    yield from _text_stream_parts(_fallback_synthesis_answer(req), chunk_size=48)


def _truncate_progress(value: str, limit: int = 600) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _trace_block(lines: list[str], *, separator: bool = True) -> str:
    clean_lines = [line.rstrip() for line in lines if line is not None]
    suffix = "\n\n---\n\n" if separator else "\n\n"
    return "\n".join(clean_lines).rstrip() + suffix


def _blockquote_lines(text: str) -> list[str]:
    clean = str(text or "").strip()
    if not clean:
        return []
    return [f"> {line}" if line else ">" for line in clean.splitlines()]


def _trace_tone(label: str) -> str:
    label_lc = label.strip().lower()
    if any(token in label_lc for token in ("failed", "error")):
        return "failure"
    if any(token in label_lc for token in ("validation", "retry", "clarification", "trying")):
        return "validation"
    if any(token in label_lc for token in ("running", "completed step", "shell result", "sql result", "workflow complete", "completed")):
        return "execution"
    if any(token in label_lc for token in ("plan", "planning", "replanning")):
        return "planning"
    if "thinking" in label_lc:
        return "thinking"
    return "neutral"


def _trace_badge(tone: str) -> str:
    return {
        "thinking": "🟣",
        "planning": "🟦",
        "execution": "🟩",
        "validation": "🟨",
        "failure": "🟥",
        "neutral": "⬜",
    }.get(tone, "⬜")


def _trace_title(label: str, summary: str | None = None) -> str:
    title = f"### {_trace_badge(_trace_tone(label))} {label}"
    if isinstance(summary, str) and summary.strip():
        return f"{title}\n\n" + "\n".join(_blockquote_lines(summary.strip()))
    return title


def _trace_kv_lines(*items: tuple[str, Any]) -> list[str]:
    lines: list[str] = []
    for label, value in items:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        lines.append(f"**{label.title()}:** {text}")
    return lines


def _trace_snippet(label: str, value: Any, limit: int = 700) -> list[str]:
    text = _clean_terminal_output(value) if isinstance(value, str) else str(value or "").strip()
    if not text:
        return []
    text = _truncate_progress(text, limit)
    label_lc = label.lower()
    if any(token in label_lc for token in ("command", "output", "sql")):
        return _format_console_section(label, text)
    return _format_callout_section(label, text)


def _trace_bullets(items: list[str], *, prefix: str = "- ") -> list[str]:
    return [f"{prefix}{item}" for item in items if isinstance(item, str) and item.strip()]


def _narration_block(text: str) -> str:
    return _trace_block(_blockquote_lines(text))


def _narration_for_event(event_name: str, payload: dict, state: dict[str, Any]) -> str | None:
    if event_name == "user.ask":
        return _narration_block("Mapping the request into a workflow and choosing the best path to try first.")
    if event_name == "plan.progress":
        options = payload.get("options")
        if isinstance(options, list) and len(options) > 1:
            return _narration_block(
                f"{len(options)} viable workflow options were found. The strongest path will run first, with fallback only if validation says it missed the mark."
            )
        steps = payload.get("steps")
        if isinstance(steps, list) and steps:
            return _narration_block(
                f"The request was broken into {len(steps)} planned step(s) so execution stays structured and easier to recover if something fails."
            )
        return None
    if event_name == "step.progress":
        stage = payload.get("stage")
        step_id = str(payload.get("step_id") or "step")
        task = _task_label(payload.get("task", ""))
        target = payload.get("target_agent")
        if stage == "started":
            state["last_running_step"] = step_id
            return _narration_block(
                f"Running {step_id} with {_agent_label(target)}.\nFocus: {_truncate_progress(task, 180)}"
            )
        if stage == "completed":
            detail = None
            result = payload.get("result")
            if isinstance(result, dict):
                detail = result.get("detail")
            detail_text = f"\nResult: {_truncate_progress(detail, 180)}" if isinstance(detail, str) and detail.strip() else ""
            return _narration_block(f"{step_id} completed successfully, so its output can be used for whatever depends on it next.{detail_text}")
        if stage == "failed":
            reason = payload.get("error") or payload.get("message")
            reason_text = f"\nIssue: {_truncate_progress(str(reason), 180)}" if reason else ""
            return _narration_block(f"{step_id} did not complete cleanly. That becomes a signal to inspect, replan, or fall back.{reason_text}")
        if stage == "replanning":
            return _narration_block("The current step was too coarse or blocked, so it is being decomposed into a smaller plan before trying again.")
    if event_name == "validation.progress":
        stage = payload.get("stage")
        option_id = payload.get("option_id")
        if stage == "option_started":
            return _narration_block(f"Trying {option_id or 'this option'} and validating it against the original request before accepting it.")
        if stage == "started":
            return _narration_block("Execution finished for this option. Now checking whether it actually answered the request, not just whether the steps ran.")
        if stage == "failed":
            reason = payload.get("reason")
            reason_text = f"\nValidator note: {_truncate_progress(str(reason), 180)}" if reason else ""
            return _narration_block(f"This option executed, but it still does not fully satisfy the request, so it will not be accepted yet.{reason_text}")
        if stage == "retrying":
            return _narration_block("Using what was learned from the failed attempt and trying the next workflow option.")
        if stage == "passed":
            return _narration_block("This attempt satisfies the request closely enough to promote it to the final answer.")
    if event_name == "clarification.required":
        return _narration_block("A point was reached where guessing would be risky, so the exact clarification needed is surfaced instead of fabricating a result.")
    if event_name == "workflow.result":
        status = payload.get("status")
        attempts = payload.get("attempts")
        selected = payload.get("selected_option")
        if isinstance(attempts, list) and len(attempts) > 1 and isinstance(selected, dict):
            return _narration_block(
                f"Multiple workflow options were tested, and {selected.get('id', 'the selected option')} was the first one that validated successfully."
            )
        if status == "completed":
            return _narration_block("The execution path is complete, so presentation now shifts from orchestration to the final user-facing answer.")
        if status == "failed":
            return _narration_block("None of the attempted workflows validated successfully, so the failure state is surfaced instead of pretending the task succeeded.")
    return None


def _agent_label(agent: Any) -> str:
    return f"**{agent}**" if isinstance(agent, str) and agent else "**agent**"


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
        "total_ms": "task took",
    }
    parts = []
    for key, label in labels.items():
        duration = _duration_label(stats.get(key))
        if duration:
            if key == "total_ms":
                parts.append(f"{label} `{duration}`")
            else:
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


ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean_terminal_output(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = ANSI_PATTERN.sub("", value)
    text = text.replace("\r", "\n")
    while "\b" in text:
        text = re.sub(r".\x08", "", text)
        text = text.replace("\b", "")
    text = CONTROL_PATTERN.sub("", text)
    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        compact = line.strip()
        if not compact:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        line = re.sub(r"([:\s])[-\\/|.\s]+(done)$", r"\1\2", line)
        line = re.sub(r":done$", ": done", line)
        compact = line.strip()
        if re.fullmatch(r"[-\\/|.\s]+", compact):
            continue
        cleaned_lines.append(line)
    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()
    return "\n".join(cleaned_lines).strip()


def _output_excerpt(value: Any, limit: int = 900) -> str:
    text = _clean_terminal_output(value)
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) > 18:
        text = "\n".join([*lines[:8], "...", *lines[-8:]])
    return _truncate_progress(text, limit)


def _format_preformatted_markdown(value: Any) -> list[str]:
    text = "" if value is None else str(value)
    if not text:
        return [">"]
    return [(f"> {line}" if line else ">") for line in text.splitlines()]


def _format_console_section(label: str, value: Any) -> list[str]:
    text = "" if value is None else str(value).rstrip()
    if not text:
        return []
    return [f"**{label}:**", "", "```text", text, "```"]


def _format_callout_section(label: str, value: Any) -> list[str]:
    text = "" if value is None else str(value).strip()
    if not text:
        return []
    if "\n" not in text:
        return [f"**{label}:** `{text}`"]
    return _format_console_section(label, text)


def _candidate_file_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or "\n" in text:
        return None
    if text.startswith(("http://", "https://", "app://", "file://")):
        return None
    candidate = text.strip("`")
    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {"'", '"'}:
        candidate = candidate[1:-1].strip()
    if not candidate or candidate.endswith(("/", "\\")):
        return None
    normalized = os.path.expanduser(candidate)
    has_separator = "/" in normalized or "\\" in normalized
    if not has_separator and not normalized.startswith("."):
        return None
    basename = os.path.basename(normalized.replace("\\", "/"))
    if not basename or basename in {".", ".."}:
        return None
    absolute_candidate = os.path.abspath(normalized)
    if not os.path.isfile(absolute_candidate):
        return None
    return candidate


def _markdown_link_for_file_path(value: Any) -> str | None:
    candidate = _candidate_file_path(value)
    if not candidate:
        return None
    absolute_target = os.path.abspath(os.path.expanduser(candidate))
    target = f"<{absolute_target}>" if " " in absolute_target else absolute_target
    label = candidate.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
    return f"[{label}]({target})"


def _extract_file_paths_from_text(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    lines = [line.strip() for line in _clean_terminal_output(value).splitlines() if line.strip()]
    if not lines or len(lines) > 20:
        return []
    linked = []
    for line in lines:
        link = _markdown_link_for_file_path(line)
        if not link:
            return []
        linked.append(link)
    return linked


def _command_output_details(command: str, stdout: Any, stderr: Any, returncode: Any = None) -> str:
    linked_paths = _extract_file_paths_from_text(stdout)
    lines = _format_console_section("Command", command.strip())
    if returncode is not None:
        lines.extend(["", f"**Exit code:** `{returncode}`"])
    clean_stdout = _clean_terminal_output(stdout) or "<empty>"
    if linked_paths:
        lines.extend(["", "**Files**"])
        lines.extend([f"- {path}" for path in linked_paths])
    lines.extend([""])
    lines.extend(_format_console_section("Output", clean_stdout))
    clean_stderr = _clean_terminal_output(stderr)
    if clean_stderr:
        lines.extend([""])
        lines.extend(_format_console_section("Error output", clean_stderr))
    return "\n".join(lines)


def _shell_details_for_event(event_name: str, payload: dict) -> str:
    shell_payloads = []
    if event_name == "shell.result":
        shell_payloads.append(payload)
    elif event_name == "workflow.result":
        steps = payload.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict) or step.get("event") != "shell.result":
                    continue
                step_payload = step.get("payload")
                if isinstance(step_payload, dict):
                    shell_payloads.append(step_payload)

    details = []
    for shell_payload in shell_payloads:
        command = shell_payload.get("command")
        if not isinstance(command, str) or not command.strip():
            continue
        details.append(
            _command_output_details(
                command,
                shell_payload.get("stdout", ""),
                shell_payload.get("stderr", ""),
                shell_payload.get("returncode"),
            )
        )
    return "\n\n".join(details)


def _shell_status(returncode: Any) -> str:
    return "success" if returncode == 0 else "failed"


def _append_collapsed_command(lines: list[str], command: str, summary: str = "Command"):
    lines.extend(_format_callout_section(summary, command.strip()))


def _append_shell_result(lines: list[str], result: dict, duration: str | None = None):
    command = result.get("command")
    returncode = result.get("returncode")
    stdout = _output_excerpt(result.get("stdout"), 900)
    stderr = _output_excerpt(result.get("stderr"), 600)
    status = _shell_status(returncode)
    summary = _compact_fields(
        ("Status", f"`{status}`"),
        ("Exit", f"`{returncode}`" if returncode is not None else None),
        ("Duration", f"`{duration}`" if duration else None),
    )
    if summary:
        lines.append(summary)
    if isinstance(command, str) and command.strip():
        lines.extend(_format_callout_section("Command", command.strip()))
    if stdout:
        lines.extend([""])
        lines.extend(_format_callout_section("Output", stdout))
    if stderr:
        warning_label = "**Warnings**" if returncode == 0 else "**Error output**"
        lines.extend([""])
        lines.extend(_format_callout_section(warning_label.replace("**", ""), stderr))


def _format_progress(event_name: str, payload: dict, depth: int) -> str | None:
    if event_name == "user.ask":
        return _trace_block([_trace_title("Thinking", "Planning the workflow and selecting an execution path.")], separator=False)
    if event_name == "plan.progress":
        lines = [_trace_title("Plan", payload.get("message", "Plan ready."))]
        options = payload.get("options", [])
        if isinstance(options, list) and options:
            lines.append("**Workflow options**")
            for option in options:
                if not isinstance(option, dict):
                    continue
                option_id = option.get("id", "option")
                label = option.get("label", option_id)
                reason = option.get("reason")
                step_count = option.get("step_count")
                summary = f"**{option_id}** {label}"
                if step_count is not None:
                    summary += f" · {step_count} step(s)"
                if isinstance(reason, str) and reason.strip():
                    summary += f" · {_truncate_progress(reason, 180)}"
                lines.append(f"- {summary}")
        if isinstance(options, list) and options:
            selected_option_id = payload.get("selected_option_id")
            if isinstance(selected_option_id, str) and selected_option_id.strip():
                lines.append(f"**Selected option:** {selected_option_id}")
        steps = payload.get("steps", [])
        if isinstance(steps, list):
            lines.append("**Planned steps**")
            for step in steps:
                if not isinstance(step, dict):
                    continue
                task = step.get("task")
                if not isinstance(task, str) or not task.strip():
                    continue
                target = step.get("target_agent")
                lines.append(
                    f"- **{step.get('id', 'step')}** via {_agent_label(target)}: {_task_label(task)}"
                )
        return _trace_block(lines)
    if event_name == "validation.progress":
        stage = payload.get("stage", "")
        option_id = payload.get("option_id", "option")
        option_label = payload.get("option_label")
        label = f"**{option_id}**"
        if isinstance(option_label, str) and option_label.strip():
            label += f" {option_label.strip()}"
        if stage == "option_started":
            lines = [_trace_title("Trying workflow option", str(label))]
            reason = payload.get("reason")
            if isinstance(reason, str) and reason.strip():
                lines.extend(_trace_kv_lines(("reason", _truncate_progress(reason, 240))))
            return _trace_block(lines)
        if stage == "started":
            return _trace_block([_trace_title("Validation", f"Checking whether {label} satisfied the request.")])
        if stage == "passed":
            lines = [_trace_title("Validation passed", str(label))]
            reason = payload.get("reason")
            if isinstance(reason, str) and reason.strip():
                lines.extend(_trace_kv_lines(("reason", _truncate_progress(reason, 240))))
            trace = payload.get("trace")
            if isinstance(trace, list):
                lines.append("**Checks**")
                lines.extend(_trace_bullets([_truncate_progress(item, 240) for item in trace[:4] if isinstance(item, str) and item.strip()]))
            return _trace_block(lines)
        if stage == "failed":
            lines = [_trace_title("Validation failed", str(label))]
            reason = payload.get("reason")
            if isinstance(reason, str) and reason.strip():
                lines.extend(_trace_kv_lines(("reason", _truncate_progress(reason, 240))))
            trace = payload.get("trace")
            if isinstance(trace, list):
                lines.append("**Checks**")
                lines.extend(_trace_bullets([_truncate_progress(item, 240) for item in trace[:4] if isinstance(item, str) and item.strip()]))
            return _trace_block(lines)
        if stage == "retrying":
            message = payload.get("message") or "Trying the next workflow option."
            return _trace_block([_trace_title("Retrying", message)])
    if event_name == "task.plan":
        step_id = payload.get("step_id")
        target = payload.get("target_agent")
        task = payload.get("task", "")
        if step_id or target:
            return None
        return _trace_block([_trace_title("Planning", task)])
    if event_name == "step.progress":
        stage = payload.get("stage", "")
        step_id = payload.get("step_id", "step")
        target = payload.get("target_agent")
        task = payload.get("task", "")
        if stage == "started":
            lines = [_trace_title("Running step", f"`{step_id}` via {_agent_label(target)}")]
            lines.extend(_trace_kv_lines(("task", _task_label(task))))
            sql = payload.get("sql")
            if isinstance(sql, str) and sql.strip():
                lines.extend(_trace_snippet("SQL snippet", sql, 500))
            command = payload.get("command")
            if isinstance(command, str) and command.strip():
                lines.extend(_trace_snippet("Command snippet", command, 500))
            return _trace_block(lines)
        if stage == "replanning":
            lines = [_trace_title("Replanning step", f"`{step_id}` via {_agent_label(target)}")]
            lines.extend(_trace_kv_lines(("task", _task_label(task))))
            error = payload.get("error")
            if isinstance(error, str) and error.strip():
                lines.extend(_trace_kv_lines(("reason", f"`{_truncate_progress(error, 260)}`")))
            return _trace_block(lines)
        if stage == "completed":
            lines = [_trace_title("Completed step", f"`{step_id}` via {_agent_label(target)}")]
            duration = _duration_label(payload.get("duration_ms"))
            result = payload.get("result")
            sql = payload.get("sql")
            row_count = None
            stats_line = None
            if isinstance(result, dict):
                detail = result.get("detail")
                if isinstance(detail, str) and detail.strip():
                    lines.extend(_trace_kv_lines(("detail", f"`{_truncate_progress(detail, 260)}`")))
                row_count = result.get("row_count")
                stats_line = _format_stats(result.get("stats"))
                nested_result = result.get("result")
                if not stats_line and isinstance(nested_result, dict):
                    stats_line = _format_stats(nested_result.get("stats"))
                if not sql:
                    sql = result.get("sql")
                if "command" in result or "returncode" in result:
                    returncode = result.get("returncode")
                    summary = _compact_fields(
                        ("Status", f"`{_shell_status(returncode)}`"),
                        ("Exit", f"`{returncode}`" if returncode is not None else None),
                        ("Duration", f"`{duration}`" if duration else None),
                    )
                    if summary:
                        lines.append(summary)
            summary = _compact_fields(
                ("Duration", f"`{duration}`" if duration and not (isinstance(result, dict) and ("command" in result or "returncode" in result)) else None),
                ("Rows", f"`{row_count}`" if row_count is not None else None),
            )
            if summary:
                lines.append(summary)
            if stats_line:
                lines.append(f"`timing` {stats_line}")
            
            local_cmd = payload.get("local_reduction_command")
            if isinstance(local_cmd, str) and local_cmd.strip():
                lines.extend(_trace_snippet("Local reduction command", local_cmd, 500))
                
            if isinstance(sql, str) and sql.strip():
                lines.extend(_trace_snippet("SQL", sql, 700))
            command = payload.get("command")
            if not command and isinstance(result, dict):
                command = result.get("command")
            if isinstance(command, str) and command.strip():
                lines.extend(_trace_snippet("Command", command, 700))

            # Raw Output Visibility
            raw_stdout = payload.get("stdout")
            raw_stderr = payload.get("stderr")
            if isinstance(result, dict):
                raw_stdout = raw_stdout or result.get("stdout")
                raw_stderr = raw_stderr or result.get("stderr")
                if not raw_stdout and isinstance(result.get("slurm"), dict):
                    raw_stdout = result["slurm"].get("stdout")
                    raw_stderr = raw_stderr or result["slurm"].get("stderr")

            clean_stdout = _clean_terminal_output(raw_stdout)
            if clean_stdout:
                lines.extend(_trace_snippet("Output", clean_stdout, 1200))

            clean_stderr = _clean_terminal_output(raw_stderr)
            if clean_stderr:
                lines.extend(_trace_snippet("Error output", clean_stderr, 900))
            return _trace_block(lines)
        if stage == "failed":
            error = payload.get("error") or payload.get("message") or "Step failed."
            lines = [_trace_title("Failed step", f"`{step_id}` via {_agent_label(target)}")]
            lines.extend(_trace_kv_lines(("error", error)))
            duration = _duration_label(payload.get("duration_ms"))
            if duration:
                lines.extend(_trace_kv_lines(("duration", f"`{duration}`")))
            result = payload.get("result")
            if isinstance(result, dict):
                detail = result.get("detail")
                if isinstance(detail, str) and detail.strip():
                    lines.extend(_trace_kv_lines(("detail", f"`{_truncate_progress(detail, 260)}`")))
                stats_line = _format_stats(result.get("stats"))
                nested_result = result.get("result")
                if not stats_line and isinstance(nested_result, dict):
                    stats_line = _format_stats(nested_result.get("stats"))
                if stats_line:
                    lines.append(f"`timing` {stats_line}")
            return _trace_block(lines)
    if event_name == "shell.result":
        command = payload.get("command", "")
        returncode = payload.get("returncode")
        lines = [_trace_title("Shell result", f"`{_shell_status(returncode)}`")]
        _append_shell_result(lines, payload)
        return _trace_block(lines)
    if event_name == "sql.result":
        sql = payload.get("sql", "")
        result = payload.get("result")
        row_count = None
        if isinstance(result, dict):
            row_count = result.get("row_count")
        lines = [_trace_title("SQL result", f"Completed query{f' with `{row_count}` row(s)' if row_count is not None else ''}.")]
        if isinstance(sql, str) and sql.strip():
            lines.extend(_trace_snippet("SQL", sql, 700))
        stats_line = _format_stats(payload.get("stats"))
        if stats_line:
            lines.append(f"`timing` {stats_line}")
        return _trace_block(lines)
    if event_name in {"file.content", "notify.result"}:
        return _trace_block([_trace_title("Completed", event_name)])
    if event_name == "task.result":
        return None
    if event_name == "clarification.required":
        question = payload.get("question") or payload.get("detail") or "More information is required."
        return _trace_block([_trace_title("Clarification needed", question)], separator=False)
    if event_name == "workflow.result":
        status = payload.get("status", "unknown")
        selected = payload.get("selected_option")
        lines = [_trace_title("Workflow complete", f"Status: `{status}`")]
        if isinstance(selected, dict):
            option_id = selected.get("id")
            label = selected.get("label")
            if isinstance(option_id, str) and option_id.strip():
                summary = f"`selected option` `{option_id}`"
                if isinstance(label, str) and label.strip():
                    summary += f" {label.strip()}"
                lines.append(summary)
        validation = payload.get("validation")
        if isinstance(validation, dict):
            reason = validation.get("reason")
            if isinstance(reason, str) and reason.strip():
                lines.extend(_trace_kv_lines(("validator", f"`{_truncate_progress(reason, 260)}`")))
        return _trace_block(lines, separator=False)
    return None


def _stream_response(client_request: Request, question: str, model: str):
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    events: queue.Queue[dict | None] = queue.Queue()
    cancel_event = threading.Event()
    response_holder: dict[str, Any] = {"response": None}

    def run_planner():
        final_answer_streamed = False
        narration_state: dict[str, Any] = {}

        def put_text(content: str):
            for part in _text_stream_parts(content):
                events.put(_stream_chunk(completion_id, created, model, part))

        try:
            if gateway is None:
                raise RuntimeError("Planner gateway is not ready.")

            def on_event(event_name: str, payload: dict, depth: int):
                nonlocal final_answer_streamed
                if event_name == "answer.final" and isinstance(payload, dict):
                    answer = payload.get("answer")
                    if isinstance(answer, str) and answer:
                        events.put(_stream_chunk(completion_id, created, model, answer))
                        final_answer_streamed = True
                    return

                if (
                    not final_answer_streamed
                    and event_name in SYNTHESIZER_EVENTS
                    and isinstance(payload, dict)
                ):
                    for part in _stream_synthesis_parts(
                        event_name,
                        payload,
                        cancel_event=cancel_event,
                        response_holder=response_holder,
                    ):
                        if cancel_event.is_set():
                            break
                        events.put(_stream_chunk(completion_id, created, model, part))
                    if cancel_event.is_set():
                        raise ClientCancelled("Client disconnected.")
                    shell_details = None
                    if not _should_use_gateway_direct_fallback(event_name, payload):
                        shell_details = _shell_details_for_event(event_name, payload)
                    if shell_details:
                        events.put(
                            _stream_chunk(
                                completion_id,
                                created,
                                model,
                                "\n\n" + shell_details,
                            )
                        )
                    final_answer_streamed = True
                    return

                narration = _narration_for_event(event_name, payload, narration_state)
                if narration and _stream_progress_enabled():
                    events.put(_stream_chunk(completion_id, created, model, narration))

                progress = _format_progress(event_name, payload, depth)
                if progress and _stream_progress_enabled():
                    events.put(_stream_chunk(completion_id, created, model, progress))

            answer = gateway.ask(
                question,
                on_event=on_event,
                disable_synthesizer=True,
                should_cancel=cancel_event.is_set,
            )
            if not final_answer_streamed:
                put_text(answer)
        except ClientCancelled:
            pass
        except Exception as exc:
            put_text(f"Error: {type(exc).__name__}: {exc}")
        finally:
            response = response_holder.get("response")
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
            events.put(None)

    async def generate():
        worker = threading.Thread(target=run_planner, daemon=True)
        worker.start()
        yield _sse_event(_stream_role_chunk(completion_id, created, model))
        disconnected = False
        while True:
            if await client_request.is_disconnected():
                disconnected = True
                cancel_event.set()
                response = response_holder.get("response")
                if response is not None:
                    try:
                        response.close()
                    except Exception:
                        pass
                break
            try:
                item = events.get(timeout=0.1)
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            if item is None:
                break
            yield _sse_event(item)
        if disconnected:
            return
        yield _sse_event(_stream_stop_chunk(completion_id, created, model))
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=_streaming_headers(),
    )


def _static_stream_response(answer: str, model: str):
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def generate():
        yield _sse_event(_stream_role_chunk(completion_id, created, model))
        if answer:
            for part in _text_stream_parts(answer):
                yield _sse_event(_stream_chunk(completion_id, created, model, part))
        yield _sse_event(_stream_stop_chunk(completion_id, created, model))
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=_streaming_headers(),
    )


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
def chat_completions(request: ChatCompletionRequest, client_request: Request):
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
        return _upstream_chat_completion(request, client_request)

    planner_question = _planner_question_from_messages(request.messages)

    if request.stream:
        return _stream_response(client_request, planner_question, selected_model)
    answer = gateway.ask(planner_question)
    return _chat_response(answer, selected_model)


def create_app(spec_path: str, timeout_seconds: float | None = None, selected_agents: list[str] | None = None):
    global gateway
    gateway = PlannerGateway(spec_path, timeout_seconds=timeout_seconds, selected_agents=selected_agents)

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
    parser.add_argument(
        "--agent",
        action="append",
        dest="selected_agents",
        help="Start only the named agent(s). Repeat the flag or pass a comma-separated list. Matches agent name, template name, or argument instance name.",
    )
    parser.add_argument(
        "--enable-context",
        action="store_true",
        help="Include prior Open WebUI chat history in planner requests. Disabled by default.",
    )
    args = parser.parse_args()
    if args.enable_context:
        os.environ["ENABLE_CONTEXT"] = "1"

    uvicorn.run(
        create_app(args.spec, timeout_seconds=args.timeout, selected_agents=args.selected_agents),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
