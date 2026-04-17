import json
import os
import re
from typing import Any, List

from fastapi import FastAPI
import requests

from agent_library.common import EventRequest, EventResponse
from runtime.console import log_debug, log_raw

app = FastAPI()

AGENT_METADATA = {
    "description": "Performs basic calculator operations from task plans using LLM-selected function invocation.",
    "capability_domains": ["math", "arithmetic"],
    "action_verbs": ["add", "subtract", "multiply", "divide", "compute", "calculate"],
    "side_effect_policy": "read_only",
    "safety_enforced_by_agent": True,
    "routing_notes": [
        "Only handle arithmetic requests that map to add/subtract/multiply/divide.",
        "Use an LLM preprocessing step to choose the calculator function and operands.",
        "If request does not map to calculator capabilities, emit nothing.",
    ],
    "methods": [
        {
            "name": "compute_with_llm_selected_function",
            "event": "task.plan",
            "when": "Uses LLM to select add/subtract/multiply/divide and operands from task text.",
            "intent_tags": ["math", "arithmetic"],
            "examples": ["add 12 and 30", "what is 144 / 12"],
            "anti_patterns": ["find files named vinith", "read README.md"],
        }
    ],
}


CALCULATOR_CAPABILITIES = {
    "functions": [
        {"name": "add", "signature": "add(operands: number[])", "notes": "Sum all operands."},
        {
            "name": "subtract",
            "signature": "subtract(operands: number[])",
            "notes": "Subtract operands from left to right.",
        },
        {
            "name": "multiply",
            "signature": "multiply(operands: number[])",
            "notes": "Multiply operands from left to right.",
        },
        {
            "name": "divide",
            "signature": "divide(operands: number[])",
            "notes": "Divide operands from left to right. Division by zero is invalid.",
        },
    ]
}
FUNCTIONS = {item["name"] for item in CALCULATOR_CAPABILITIES["functions"]}


def _debug_enabled() -> bool:
    return os.getenv("LLM_OPS_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _debug_log(message: str):
    if _debug_enabled():
        log_debug("CALC_DEBUG", message)


def _normalize_number(value: float):
    return int(value) if value.is_integer() else value


def _extract_json_object(text: str):
    start = text.find("{")
    if start == -1:
        return None
    end = text.rfind("}")
    if end == -1 or end < start:
        return text[start:]
    return text[start : end + 1]


def _repair_json_blob(blob: str):
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


def _parse_json(content: str):
    json_blob = _extract_json_object(content)
    if not json_blob:
        return None
    for candidate in (json_blob, _repair_json_blob(json_blob)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _build_preprocess_prompt(task: str):
    return (
        "You are a strict calculator preprocessor.\n"
        "Do NOT solve math. Do NOT compute results. Do NOT infer missing numbers from world knowledge.\n"
        "Treat the request as a single atomic calculator step, not a multi-step workflow.\n"
        "Your job is only to map text to a calculator function and explicit numeric operands found in the request.\n"
        "Decide if the request can be handled ONLY by these calculator functions:\n"
        f"{json.dumps(CALCULATOR_CAPABILITIES, indent=2)}\n"
        "Return ONLY JSON with this exact shape:\n"
        '{"processable":true|false,"function":"add|subtract|multiply|divide|null","operands":[number,...],"reason":"short reason"}\n'
        "Rules:\n"
        "- If unprocessable, set processable=false, function=null, operands=[].\n"
        "- If processable, set processable=true and provide a valid function with at least two numeric operands.\n"
        "- Use only numeric operands explicitly present in the user request.\n"
        "- If the request has words like 'times two', convert only explicit number words one..ten to digits.\n"
        "- If fewer than two explicit operands are available, mark unprocessable.\n"
        "- If the request contains multiple chained operations, mark unprocessable instead of guessing.\n"
        "- Never call external knowledge (e.g., planet sizes, constants not stated by user).\n"
        "- operands MUST be a JSON array of bare numbers only, for example [50,5].\n"
        "- Never return operands as objects, strings, or nested arrays.\n"
        "- function MUST be one of add/subtract/multiply/divide or null.\n"
        "Valid examples:\n"
        '{"processable":true,"function":"add","operands":[12,30],"reason":"explicit add request"}\n'
        '{"processable":true,"function":"divide","operands":[50,5],"reason":"explicit equal split"}\n'
        '{"processable":false,"function":null,"operands":[],"reason":"not enough explicit numbers"}\n'
        "Invalid examples (DO NOT OUTPUT):\n"
        '{"processable":true,"function":"divide","operands":[{"value":"50"},{"value":"5"}],"reason":"..."}\n'
        '{"processable":true,"function":"multiply","operands":["50","5"],"reason":"..."}\n'
        "- No markdown, no prose, no extra keys.\n"
        f'User request: "{task}"'
    )


def _llm_preprocess(task: str):
    api_key = os.getenv("LLM_OPS_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_OPS_BASE_URL")
    model = os.getenv("LLM_OPS_MODEL")
    timeout_seconds = float(os.getenv("LLM_OPS_TIMEOUT_SECONDS", "300"))

    if not api_key:
        raise RuntimeError("LLM_OPS_API_KEY is not set")

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
    user_prompt = _build_preprocess_prompt(task)
    messages = [
        {"role": "system", "content": "You produce strict JSON only."},
        {"role": "user", "content": user_prompt},
    ]
    payload = {"model": model, "messages": messages, "temperature": 0}

    _debug_log("Constructed calculator preprocessing prompt:")
    _debug_log(user_prompt)

    response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    log_raw("CALC_LLM_RAW", content)
    _debug_log("Raw preprocessing response:")
    _debug_log(content)
    return _parse_json(content)


def _parse_decision(raw: Any):
    if not isinstance(raw, dict):
        return None

    processable = raw.get("processable")
    function = raw.get("function")
    operands = raw.get("operands")
    reason = raw.get("reason", "")

    if not isinstance(processable, bool):
        return None
    if function is not None and not isinstance(function, str):
        return None
    if not isinstance(operands, list):
        return None
    if not isinstance(reason, str):
        reason = ""

    parsed_operands: List[float] = []
    for value in operands:
        if not isinstance(value, (int, float)):
            return None
        parsed_operands.append(float(value))

    if not processable:
        return {
            "processable": False,
            "function": None,
            "operands": [],
            "reason": reason.strip(),
        }

    if function not in FUNCTIONS:
        return None
    if len(parsed_operands) < 2:
        return None

    return {
        "processable": True,
        "function": function,
        "operands": parsed_operands,
        "reason": reason.strip(),
    }


def _execute(function: str, operands: List[float]):
    if function == "add":
        return ("sum", _normalize_number(sum(operands)))

    if function == "subtract":
        value = operands[0]
        for num in operands[1:]:
            value -= num
        return ("difference", _normalize_number(value))

    if function == "multiply":
        value = operands[0]
        for num in operands[1:]:
            value *= num
        return ("product", _normalize_number(value))

    if function == "divide":
        value = operands[0]
        for num in operands[1:]:
            if num == 0:
                return ("error", "Cannot divide by zero")
            value /= num
        return ("quotient", _normalize_number(value))

    return ("error", "Unknown calculator function")


def _format_result(function: str, operands: List[float], kind: str, value: Any):
    operand_text = ", ".join(str(_normalize_number(num)) for num in operands)
    if kind == "error":
        return str(value)
    return f"Computed {kind} via {function}([{operand_text}]): {value}"


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event != "task.plan":
        return {"emits": []}

    task = req.payload["task"]
    if not isinstance(task, str):
        return {"emits": []}

    try:
        decision_raw = _llm_preprocess(task)
        decision = _parse_decision(decision_raw)
    except Exception as exc:
        _debug_log(f"Calculator preprocessing failed: {type(exc).__name__}: {exc}")
        return {"emits": []}

    if decision is None:
        _debug_log("Calculator preprocessing returned invalid JSON shape.")
        return {"emits": []}

    if not decision["processable"]:
        return {"emits": []}

    kind, value = _execute(decision["function"], decision["operands"])
    detail = _format_result(decision["function"], decision["operands"], kind, value)
    result_value = value if kind != "error" else None
    return {"emits": [{"event": "task.result", "payload": {"detail": detail, "result": result_value}}]}
