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
    "task.plan": '{"task":"..."}',
    "task.result": '{"detail":"..."}',
}
DEFAULT_ALLOWED_EVENTS = set(SUPPORTED_EVENT_SCHEMAS.keys())
CAPABILITIES = {"agents": [], "available_events": sorted(DEFAULT_ALLOWED_EVENTS)}


def _debug_enabled() -> bool:
    return os.getenv("LLM_OPS_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _debug_log(message: str):
    if _debug_enabled():
        print(f"[LLM_OPS_DEBUG] {message}")


def _extract_filepath(question: str):
    match = re.search(r"(?:read|open)\s+([./a-zA-Z0-9_-]+\.[a-zA-Z0-9]+)", question)
    return match.group(1) if match else None


def _extract_command(question: str):
    match = re.search(r"(?:run|execute)\s+`([^`]+)`", question)
    return match.group(1) if match else None


def _extract_task_plan(question: str):
    numbers = re.findall(r"-?\d+(?:\.\d+)?", question)
    question_lc = question.lower()
    has_arithmetic_intent = any(
        token in question_lc
        for token in (
            "sum",
            "add",
            "plus",
            "subtract",
            "minus",
            "multiply",
            "multipy",
            "mutiply",
            "times",
            "divide",
            "divided by",
        )
    ) or any(symbol in question for symbol in ("+", "-", "*", "/"))
    if len(numbers) >= 2 and has_arithmetic_intent:
        return question
    return None


def _fallback_plan(question: str, allowed_events) -> List[Dict[str, Any]]:
    question_lc = question.lower()
    emits: List[Dict[str, Any]] = []

    filepath = _extract_filepath(question)
    if filepath and "file.read" in allowed_events:
        emits.append({"event": "file.read", "payload": {"path": filepath}})

    command = _extract_command(question)
    if command and "shell.exec" in allowed_events:
        emits.append({"event": "shell.exec", "payload": {"command": command}})

    task = _extract_task_plan(question)
    if task and "task.plan" in allowed_events:
        emits.append({"event": "task.plan", "payload": {"task": task}})

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


def _format_discovered_agents(capabilities: dict) -> str:
    lines = []
    for item in capabilities.get("agents", []):
        name = item.get("name")
        if not name:
            continue
        description = item.get("description", "").strip() or "No description provided."
        method_list = item.get("methods", [])
        if isinstance(method_list, list) and method_list:
            method_parts = []
            for method in method_list:
                if not isinstance(method, dict):
                    continue
                method_name = method.get("name", "unnamed_method")
                method_event = method.get("event", "unknown_event")
                method_when = method.get("when", "").strip()
                method_text = f"{method_name} -> {method_event}"
                if method_when:
                    method_text += f" ({method_when})"
                method_parts.append(method_text)
            methods_text = "; ".join(method_parts) if method_parts else "No methods provided."
        else:
            methods_text = "No methods provided."
        lines.append(f"- {name}: {description} Methods: {methods_text}")
    return "\n".join(lines) if lines else "- unknown: No discovered agents"


def _build_prompt(question: str, allowed_events, capabilities: dict) -> str:
    return (
        "You are a strict planning agent for an operations assistant.\n"
        "Return ONLY JSON with this exact shape: "
        '{"emits":[{"event":"...","payload":{...}}]}.\n'
        "Discovered runtime agents and responsibilities:\n"
        f"{_format_discovered_agents(capabilities)}\n"
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


def _prefer_calculator_for_arithmetic(question: str, emits: List[Dict[str, Any]], allowed_events) -> List[Dict[str, Any]]:
    task = _extract_task_plan(question)
    if not task or "task.plan" not in allowed_events:
        return emits

    # For arithmetic user intent, force calculator path even if LLM suggests shell commands.
    return [{"event": "task.plan", "payload": {"task": task}}]


def _llm_plan(question: str, allowed_events, capabilities):
    base_url = os.getenv("LLM_OPS_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("LLM_OPS_API_KEY")
    model = os.getenv("LLM_OPS_MODEL", "gpt-4o-mini")
    timeout_seconds = float(os.getenv("LLM_OPS_TIMEOUT_SECONDS", "10"))

    if not api_key:
        raise RuntimeError("LLM_OPS_API_KEY is not set")

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    user_prompt = _build_prompt(question, allowed_events, capabilities)
    messages = [
        {"role": "system", "content": "You produce strict JSON only."},
        {"role": "user", "content": user_prompt},
    ]

    _debug_log("Constructed planner prompt:")
    _debug_log(user_prompt)
    _debug_log("Messages sent to LLM:")
    _debug_log(json.dumps(messages, indent=2))

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    _debug_log("Raw LLM response content:")
    _debug_log(content)
    json_blob = _extract_json_object(content)
    if not json_blob:
        _debug_log("No JSON object found in LLM response content.")
        return []
    _debug_log("Extracted JSON object from LLM response:")
    _debug_log(json_blob)
    parsed = json.loads(json_blob)
    valid_emits = _validate_emits(parsed, allowed_events)
    valid_emits = _prefer_calculator_for_arithmetic(question, valid_emits, allowed_events)
    _debug_log("Validated emits from LLM response:")
    _debug_log(json.dumps(valid_emits, indent=2))
    return valid_emits


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
