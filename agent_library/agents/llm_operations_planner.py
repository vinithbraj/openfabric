import json
import os
import re
from typing import Any, Dict, List

import requests
from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse

app = FastAPI()

SUPPORTED_EVENT_SCHEMAS = {
    "file.read": '{"path":"relative/file/path"}',
    "shell.exec": '{"command":"..."}',
    "notify.send": '{"channel":"console","message":"..."}',
    "task.result": '{"detail":"..."}',
}
DEFAULT_ALLOWED_EVENTS = set(SUPPORTED_EVENT_SCHEMAS.keys())
CAPABILITIES = {"agents": [], "available_events": sorted(DEFAULT_ALLOWED_EVENTS)}


def _extract_filepath(question: str):
    match = re.search(r"(?:read|open)\s+([./a-zA-Z0-9_-]+\.[a-zA-Z0-9]+)", question)
    return match.group(1) if match else None


def _extract_command(question: str):
    match = re.search(r"(?:run|execute)\s+`([^`]+)`", question)
    return match.group(1) if match else None


def _fallback_plan(question: str, allowed_events) -> List[Dict[str, Any]]:
    question_lc = question.lower()
    emits: List[Dict[str, Any]] = []

    filepath = _extract_filepath(question)
    if filepath and "file.read" in allowed_events:
        emits.append({"event": "file.read", "payload": {"path": filepath}})

    command = _extract_command(question)
    if command and "shell.exec" in allowed_events:
        emits.append({"event": "shell.exec", "payload": {"command": command}})

    if ("notify" in question_lc or "alert" in question_lc) and "notify.send" in allowed_events:
        emits.append(
            {
                "event": "notify.send",
                "payload": {
                    "channel": "console",
                    "message": f"Notification requested: {question}",
                },
            }
        )

    if not emits and "task.result" in allowed_events:
        emits.append(
            {
                "event": "task.result",
                "payload": {
                    "detail": "No operations detected. Use phrases like "
                    "'read <file>', 'run `<command>`', or 'notify ...'."
                },
            }
        )

    return emits or [{"event": "task.result", "payload": {"detail": "No available tools to execute the request."}}]


def _format_allowed_event_schemas(allowed_events):
    lines = []
    for event in sorted(allowed_events):
        schema = SUPPORTED_EVENT_SCHEMAS.get(event)
        if schema:
            lines.append(f"- {event} -> {schema}")
    return "\n".join(lines) if lines else '- task.result -> {"detail":"No tools available"}'


def _build_prompt(question: str, allowed_events, capabilities: dict) -> str:
    agent_names = [item.get("name") for item in capabilities.get("agents", []) if item.get("name")]
    available_agents = ", ".join(agent_names) if agent_names else "unknown"
    return (
        "You are a strict planning agent for an operations assistant.\n"
        "Return ONLY JSON with this exact shape: "
        '{"emits":[{"event":"...","payload":{...}}]}.\n'
        f"Discovered runtime agents: {available_agents}.\n"
        "Allowed events and payload schemas:\n"
        f"{_format_allowed_event_schemas(allowed_events)}\n"
        "Prefer actionable tool events when relevant. Do not invent extra keys.\n"
        f'User request: "{question}"'
    )


def _extract_json_object(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]


def _validate_emits(raw: Any, allowed_events) -> List[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    emits = raw.get("emits", [])
    if not isinstance(emits, list):
        return []

    valid: List[Dict[str, Any]] = []
    for item in emits:
        if not isinstance(item, dict):
            continue
        event = item.get("event")
        payload = item.get("payload")
        if event not in allowed_events or not isinstance(payload, dict):
            continue
        valid.append({"event": event, "payload": payload})
    return valid


def _llm_plan(question: str, allowed_events, capabilities):
    base_url = os.getenv("LLM_OPS_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("LLM_OPS_API_KEY")
    model = os.getenv("LLM_OPS_MODEL", "gpt-4o-mini")
    timeout_seconds = float(os.getenv("LLM_OPS_TIMEOUT_SECONDS", "10"))

    if not api_key:
        raise RuntimeError("LLM_OPS_API_KEY is not set")

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You produce strict JSON only."},
            {
                "role": "user",
                "content": _build_prompt(question, allowed_events, capabilities),
            },
        ],
        "temperature": 0,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    json_blob = _extract_json_object(content)
    if not json_blob:
        return []
    parsed = json.loads(json_blob)
    return _validate_emits(parsed, allowed_events)


def _derive_available_events(agents: List[Dict[str, Any]]) -> List[str]:
    available = set()
    for agent in agents:
        for event in agent.get("subscribes_to", []):
            if event in SUPPORTED_EVENT_SCHEMAS and event != "user.ask":
                available.add(event)
    return sorted(available or DEFAULT_ALLOWED_EVENTS)


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event == "system.capabilities":
        agents = req.payload.get("agents", [])
        if isinstance(agents, list):
            CAPABILITIES["agents"] = agents
            CAPABILITIES["available_events"] = _derive_available_events(agents)
        return {"emits": []}

    if req.event != "user.ask":
        return {"emits": []}

    question = req.payload["question"]
    allowed_events = set(CAPABILITIES.get("available_events", DEFAULT_ALLOWED_EVENTS))
    try:
        emits = _llm_plan(question, allowed_events, CAPABILITIES)
        if emits:
            return {"emits": emits}
    except Exception:
        pass

    return {"emits": _fallback_plan(question, allowed_events)}
