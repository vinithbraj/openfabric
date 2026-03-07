import json
import os
import re
from typing import Any, Dict, List

import requests
from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse

app = FastAPI()

AGENT_METADATA = {
    "description": "LLM planning agent that maps user requests into allowed tool events.",
    "routing_notes": [
        "Choose exactly the tool event that best matches user intent.",
        "For file discovery/search/list intents, prefer shell.exec with a find/ls command.",
        "Use file.read only when the user asks to open/read a specific file path.",
    ],
    "methods": [
        {
            "name": "plan_file_read",
            "event": "file.read",
            "when": "Use for reading/opening files from user request.",
            "intent_tags": ["read_file", "open_file"],
        },
        {
            "name": "plan_cli_exec",
            "event": "shell.exec",
            "when": "Use for shell command execution requests.",
            "intent_tags": ["cli_exec", "file_search", "workspace_inspection"],
        },
        {
            "name": "plan_notification",
            "event": "notify.send",
            "when": "Use for notify/alert requests.",
            "intent_tags": ["notify", "alert"],
        },
        {
            "name": "plan_arithmetic_task",
            "event": "task.plan",
            "when": "Use for arithmetic operations such as add/subtract/multiply/divide.",
            "intent_tags": ["math", "arithmetic"],
        },
        {
            "name": "planner_fallback",
            "event": "task.result",
            "when": "Use only when no actionable tool event applies.",
        },
    ],
}

SUPPORTED_EVENT_SCHEMAS = {
    "file.read": '{"path":"relative/file/path"}',
    "shell.exec": '{"command":"..."}',
    "notify.send": '{"channel":"console","message":"..."}',
    "task.plan": '{"task":"..."}',
    "task.result": '{"detail":"..."}',
}
EVENT_ALIASES = {
    "shell_command": "shell.exec",
    "shell command": "shell.exec",
    "shell.run": "shell.exec",
    "run.command": "shell.exec",
    "read_file": "file.read",
    "file.open": "file.read",
    "plan.task": "task.plan",
}
REQUIRED_PAYLOAD_KEYS = {
    "file.read": {"path"},
    "shell.exec": {"command"},
    "notify.send": {"channel", "message"},
    "task.plan": {"task"},
    "task.result": {"detail"},
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


def _extract_find_command(question: str):
    question_lc = question.lower()
    if not any(token in question_lc for token in ("find", "search", "locate")):
        return None
    if "file" not in question_lc:
        return None

    name_match = re.search(
        r"(?:named|called)\s+['\"]?([a-zA-Z0-9._-]+)['\"]?",
        question,
        re.IGNORECASE,
    )
    if name_match:
        filename = name_match.group(1)
        return f"find . -iname '*{filename}*'"

    token_match = re.search(
        r"(?:for|matching)\s+['\"]?([a-zA-Z0-9._-]+)['\"]?",
        question,
        re.IGNORECASE,
    )
    if token_match:
        token = token_match.group(1)
        return f"find . -iname '*{token}*'"

    return "find . -type f"


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

    discover_command = _extract_find_command(question)
    if discover_command and "shell.exec" in allowed_events:
        emits.append({"event": "shell.exec", "payload": {"command": discover_command}})

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
        routing_notes = item.get("routing_notes", [])
        notes_text = ""
        if isinstance(routing_notes, list) and routing_notes:
            safe_notes = [note for note in routing_notes if isinstance(note, str)]
            if safe_notes:
                notes_text = f" Routing notes: {' | '.join(safe_notes)}."
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
                extras = []
                for key in sorted(method.keys()):
                    if key in {"name", "event", "when"}:
                        continue
                    value = method.get(key)
                    if isinstance(value, str):
                        extras.append(f"{key}: {value}")
                    elif isinstance(value, list):
                        list_items = [item for item in value if isinstance(item, str)]
                        if list_items:
                            extras.append(f"{key}: {', '.join(list_items)}")
                if extras:
                    method_text += f" [{'; '.join(extras)}]"
                method_parts.append(method_text)
            methods_text = "; ".join(method_parts) if method_parts else "No methods provided."
        else:
            methods_text = "No methods provided."
        lines.append(f"- {name}: {description}{notes_text} Methods: {methods_text}")
    return "\n".join(lines) if lines else "- unknown: No discovered agents"


def _build_prompt(question: str, allowed_events, capabilities: dict) -> str:
    allowed_event_names = ", ".join(sorted(allowed_events))
    return (
        "You are a strict planning agent for an operations assistant.\n"
        "Return ONLY JSON with this exact shape: "
        '{"emits":[{"event":"...","payload":{...}}]}.\n'
        f"Valid event names are EXACTLY: {allowed_event_names}.\n"
        "Never use method names or intent tags as event values.\n"
        "Discovered runtime agents and responsibilities:\n"
        f"{_format_discovered_agents(capabilities)}\n"
        "Allowed events and payload schemas:\n"
        f"{_format_allowed_event_schemas(allowed_events)}\n"
        "Follow method routing hints: prefer methods whose intent_tags/examples match the request and avoid anti_patterns.\n"
        "Only include notify.send when the user explicitly asks to notify/alert/remind.\n"
        "Prefer actionable tool events when relevant. Do not invent extra keys.\n"
        f'User request: "{question}"'
    )


def _extract_json_object(text: str):
    start = text.find("{")
    if start == -1:
        return None
    end = text.rfind("}")
    if end == -1 or end < start:
        return text[start:]
    return text[start : end + 1]


def _repair_json_blob(blob: str):
    # Apply conservative repairs for common LLM formatting mistakes.
    repaired = blob.strip()
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    normalized_chars = []
    stack = []
    in_string = False
    escaped = False
    for ch in repaired:
        if escaped:
            normalized_chars.append(ch)
            escaped = False
            continue
        if ch == "\\":
            normalized_chars.append(ch)
            escaped = True
            continue
        if ch == '"':
            normalized_chars.append(ch)
            in_string = not in_string
            continue
        if in_string:
            normalized_chars.append(ch)
            continue
        if ch in "{[":
            normalized_chars.append(ch)
            stack.append(ch)
        elif ch == "}":
            if stack and stack[-1] == "{":
                normalized_chars.append(ch)
                stack.pop()
            else:
                continue
        elif ch == "]":
            if stack and stack[-1] == "[":
                normalized_chars.append(ch)
                stack.pop()
            else:
                continue
        else:
            normalized_chars.append(ch)

    repaired = "".join(normalized_chars)
    closing = {"{": "}", "[": "]"}
    while stack:
        repaired += closing[stack.pop()]
    return repaired


def _parse_planner_json(content: str):
    json_blob = _extract_json_object(content)
    if not json_blob:
        return None, None
    for candidate in (json_blob, _repair_json_blob(json_blob)):
        try:
            return json_blob, json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return json_blob, None


def _validate_emits(raw: Any, allowed_events) -> List[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    emits = raw.get("emits", [])
    if not isinstance(emits, list):
        return []

    method_aliases = {}
    for agent in CAPABILITIES.get("agents", []):
        for method in agent.get("methods", []):
            if not isinstance(method, dict):
                continue
            method_name = method.get("name")
            method_event = method.get("event")
            if isinstance(method_name, str) and isinstance(method_event, str):
                method_aliases[method_name] = method_event

    valid: List[Dict[str, Any]] = []
    for item in emits:
        if not isinstance(item, dict):
            continue
        event = item.get("event")
        payload = item.get("payload")
        if isinstance(event, str):
            event = EVENT_ALIASES.get(event, event)
            event = method_aliases.get(event, event)
        if not isinstance(payload, dict):
            continue
        payload = _normalize_payload(event, payload)
        required_keys = REQUIRED_PAYLOAD_KEYS.get(event, set())
        if not required_keys.issubset(set(payload.keys())):
            continue
        if event not in allowed_events or not isinstance(payload, dict):
            continue
        valid.append({"event": event, "payload": payload})
    return valid


def _normalize_payload(event: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    if event == "shell.exec":
        if "command" not in normalized and isinstance(normalized.get("cmd"), str):
            normalized["command"] = normalized["cmd"]
    elif event == "task.plan":
        if "task" not in normalized:
            operation = normalized.get("operation")
            operands = normalized.get("operands")
            if isinstance(operation, str) and isinstance(operands, list):
                tokens = [str(item) for item in operands]
                normalized["task"] = f"{operation} {' '.join(tokens)}".strip()
    elif event == "file.read":
        path = normalized.get("path")
        if isinstance(path, str) and path.startswith("/"):
            normalized["path"] = path.lstrip("/")
    elif event == "notify.send":
        if "channel" not in normalized:
            normalized["channel"] = "console"
        if "message" not in normalized and isinstance(normalized.get("detail"), str):
            normalized["message"] = normalized["detail"]
    return normalized


def _prefer_calculator_for_arithmetic(question: str, emits: List[Dict[str, Any]], allowed_events) -> List[Dict[str, Any]]:
    task = _extract_task_plan(question)
    if not task or "task.plan" not in allowed_events:
        return emits

    # For arithmetic user intent, force calculator path even if LLM suggests shell commands.
    return [{"event": "task.plan", "payload": {"task": task}}]


def _prune_unrequested_notify(question: str, emits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    question_lc = question.lower()
    asked_notify = any(token in question_lc for token in ("notify", "alert", "remind"))
    if asked_notify:
        return emits
    has_non_notify = any(item.get("event") != "notify.send" for item in emits)
    if not has_non_notify:
        return emits
    return [item for item in emits if item.get("event") != "notify.send"]


def _llm_plan(question: str, allowed_events, capabilities):
    api_key = os.getenv("LLM_OPS_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_OPS_BASE_URL")
    model = os.getenv("LLM_OPS_MODEL")
    timeout_seconds = float(os.getenv("LLM_OPS_TIMEOUT_SECONDS", "10"))

    if not api_key:
        raise RuntimeError("LLM_OPS_API_KEY is not set")

    # If a real OpenAI key is present, prefer OpenAI endpoint over local defaults.
    if api_key.startswith("sk-") and api_key.lower() != "dummy":
        base_url = "https://api.openai.com/v1"
        if not model:
            model = "gpt-4.1"
    elif not base_url:
        base_url = "https://api.openai.com/v1"
    if not model:
        model = "gpt-4o-mini"

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
    json_blob, parsed = _parse_planner_json(content)
    if not json_blob:
        _debug_log("No JSON object found in LLM response content.")
        return []
    _debug_log("Extracted JSON object from LLM response:")
    _debug_log(json_blob)
    if parsed is None:
        _debug_log("Could not parse planner JSON after repair attempts.")
        return []
    valid_emits = _validate_emits(parsed, allowed_events)
    valid_emits = _prune_unrequested_notify(question, valid_emits)
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
        _debug_log("LLM planning returned no valid emits.")
    except Exception as exc:
        _debug_log(f"LLM planning failed. Error: {type(exc).__name__}: {exc}")

    if "task.result" in allowed_events:
        return {
            "emits": [
                {
                    "event": "task.result",
                    "payload": {
                        "detail": "Planner could not produce a valid LLM plan. "
                        "Check LLM response format and retry."
                    },
                }
            ]
        }
    return {"emits": []}
