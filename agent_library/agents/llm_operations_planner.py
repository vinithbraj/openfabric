import json
import os
import re
from typing import Any, Dict, List

import requests
from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse
from runtime.console import log_debug

app = FastAPI()

AGENT_METADATA = {
    "description": "LLM planner that decides whether a request is processable by discovered system capabilities.",
    "capability_domains": ["planning", "routing", "operations"],
    "action_verbs": ["plan", "route", "assess"],
    "side_effect_policy": "read_only",
    "safety_enforced_by_agent": True,
    "routing_notes": [
        "Only decide if request is processable by discovered capabilities.",
        "If processable, emit task.plan with original task for broadcast execution.",
        "If not processable, emit task.result with reason.",
    ],
    "methods": [
        {
            "name": "assess_processable_request",
            "event": "task.plan",
            "when": "When request can be handled by at least one discovered agent capability.",
            "intent_tags": ["processable", "capability_match"],
        },
        {
            "name": "reject_unprocessable_request",
            "event": "task.result",
            "when": "When request cannot be handled by discovered capabilities.",
            "intent_tags": ["unprocessable"],
        },
    ],
}

SUPPORTED_EVENT_SCHEMAS = {
    "task.plan": '{"task":"..."}',
    "task.result": '{"detail":"..."}',
}
DEFAULT_ALLOWED_EVENTS = set(SUPPORTED_EVENT_SCHEMAS.keys())
CAPABILITIES = {"agents": [], "available_events": sorted(DEFAULT_ALLOWED_EVENTS)}


def _debug_enabled() -> bool:
    return os.getenv("LLM_OPS_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _debug_log(message: str):
    if _debug_enabled():
        log_debug("LLM_OPS_DEBUG", message)


def _format_discovered_agents(capabilities: dict) -> str:
    lines = []
    for item in capabilities.get("agents", []):
        name = item.get("name")
        if not name:
            continue
        description = item.get("description", "").strip() or "No description provided."
        domains = item.get("capability_domains", [])
        verbs = item.get("action_verbs", [])
        domain_text = ", ".join(entry for entry in domains if isinstance(entry, str)) if isinstance(domains, list) else ""
        verb_text = ", ".join(entry for entry in verbs if isinstance(entry, str)) if isinstance(verbs, list) else ""
        methods = []
        method_list = item.get("methods", [])
        if isinstance(method_list, list):
            for method in method_list:
                if not isinstance(method, dict):
                    continue
                method_name = method.get("name")
                method_event = method.get("event")
                if isinstance(method_name, str) and isinstance(method_event, str):
                    methods.append(f"{method_name}->{method_event}")
        method_text = ", ".join(methods) if methods else "none"
        lines.append(
            f"- {name}: {description} Domains[{domain_text or 'none'}] "
            f"Verbs[{verb_text or 'none'}] Methods[{method_text}]"
        )
    return "\n".join(lines) if lines else "- unknown: No discovered agents"


def _build_prompt(question: str, capabilities: dict) -> str:
    return (
        "You are the routing planner for an operations assistant.\n"
        "Task: decide whether the request is processable by at least one discovered agent capability.\n"
        "Output JSON only with exact keys: "
        '{"processable":true|false,"confidence":0..1,"reason":"short reason"}\n'
        "No markdown, no extra keys, no prose outside JSON.\n"
        "Decision policy:\n"
        "1) processable=true if any discovered agent can attempt the request.\n"
        "2) For operational shell requests (find/list/search/grep/git/commit/ps/ports), default to processable=true "
        "when shell capabilities exist.\n"
        "3) Tolerate minor typos and informal phrasing.\n"
        "4) Do NOT mark false only because the request might fail at execution time.\n"
        "5) processable=false only when no discovered capability can attempt it.\n"
        "Calibration examples:\n"
        '- "find all files with extension sh in the current directory" -> {"processable":true,"confidence":0.96,"reason":"shell file search"}\n'
        '- "grep TODO in this repo" -> {"processable":true,"confidence":0.95,"reason":"shell text search"}\n'
        '- "add 12 and 30" -> {"processable":true,"confidence":0.98,"reason":"calculator"}\n'
        '- "open Readme.md" -> {"processable":true,"confidence":0.97,"reason":"filesystem read"}\n'
        '- "commit all git changes" -> {"processable":true,"confidence":0.95,"reason":"shell git commit capability"}\n'
        '- "book a flight to NYC" -> {"processable":false,"confidence":0.98,"reason":"no travel booking capability"}\n'
        "Discovered runtime agents:\n"
        f"{_format_discovered_agents(capabilities)}\n"
        "Choose confidence based on capability match clarity.\n"
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


def _parse_decision(raw: Any):
    if not isinstance(raw, dict):
        return None
    processable = raw.get("processable")
    confidence = raw.get("confidence")
    reason = raw.get("reason", "")
    if not isinstance(processable, bool):
        return None
    if not isinstance(confidence, (int, float)):
        return None
    confidence = max(0.0, min(1.0, float(confidence)))
    if not isinstance(reason, str):
        reason = ""
    return {"processable": processable, "confidence": confidence, "reason": reason.strip()}


def _llm_decide(question: str, capabilities):
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
    user_prompt = _build_prompt(question, capabilities)
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
    if not response.ok:
        _debug_log(f"Planner LLM HTTP error status: {response.status_code}")
        _debug_log("Planner LLM HTTP error body:")
        _debug_log(response.text[:4000])
        response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    _debug_log("Raw LLM response content:")
    _debug_log(content)
    json_blob, parsed = _parse_planner_json(content)
    if not json_blob:
        _debug_log("No JSON object found in LLM response content.")
        return None
    _debug_log("Extracted JSON object from LLM response:")
    _debug_log(json_blob)
    if parsed is None:
        _debug_log("Could not parse planner JSON after repair attempts.")
        return None
    decision = _parse_decision(parsed)
    if decision is None:
        _debug_log("Parsed planner JSON missing valid processable/reason fields.")
        return None
    _debug_log("Planner decision:")
    _debug_log(json.dumps(decision, indent=2))
    return decision


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
        decision = _llm_decide(question, CAPABILITIES)
        if decision is not None:
            if decision["processable"] and "task.plan" in allowed_events:
                return {"emits": [{"event": "task.plan", "payload": {"task": question}}]}
            if "task.result" in allowed_events:
                reason = decision["reason"] or "No matching capability found."
                return {"emits": [{"event": "task.result", "payload": {"detail": reason}}]}
            return {"emits": []}
        _debug_log("LLM planner decision was invalid.")
    except Exception as exc:
        _debug_log(f"LLM planning failed. Error: {type(exc).__name__}: {exc}")

    if "task.result" in allowed_events:
        return {
            "emits": [
                {
                    "event": "task.result",
                    "payload": {
                        "detail": "Planner could not determine if the request is processable. "
                        "Check LLM connectivity/response format and retry."
                    },
                }
            ]
        }
    return {"emits": []}
