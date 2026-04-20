import json
import os
from typing import Any

import requests
from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse
from runtime.console import log_debug

app = FastAPI()

AGENT_METADATA = {
    "description": "Validates whether a workflow attempt actually satisfied the user's request and explains retry decisions.",
    "capability_domains": ["validation", "quality_control", "workflow_guardrails"],
    "action_verbs": ["validate", "verify", "gate"],
    "side_effect_policy": "read_only",
    "safety_enforced_by_agent": True,
    "methods": [
        {
            "name": "validate_workflow_attempt",
            "event": "validation.result",
            "when": "Checks whether an execution attempt answered the user's request and whether another workflow option should be tried.",
        }
    ],
}


def _debug_enabled() -> bool:
    return os.getenv("LLM_OPS_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _debug_log(message: str):
    if _debug_enabled():
        log_debug("VALIDATOR_DEBUG", message)


def _parse_validation_decision(raw: Any):
    if not isinstance(raw, dict):
        return None
    valid = raw.get("valid")
    reason = raw.get("reason", "")
    retry_recommended = raw.get("retry_recommended")
    if not isinstance(valid, bool):
        return None
    if retry_recommended is None:
        retry_recommended = not valid
    if not isinstance(retry_recommended, bool):
        return None
    if not isinstance(reason, str):
        reason = ""
    missing_requirements = raw.get("missing_requirements", [])
    if not isinstance(missing_requirements, list):
        missing_requirements = []
    trace = raw.get("trace", [])
    if not isinstance(trace, list):
        trace = []
    return {
        "valid": valid,
        "reason": reason.strip(),
        "retry_recommended": retry_recommended,
        "missing_requirements": [item for item in missing_requirements if isinstance(item, str) and item.strip()],
        "trace": [item for item in trace if isinstance(item, str) and item.strip()],
    }


def _extract_json_object(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]


def _llm_validate(payload: dict):
    api_key = os.getenv("VALIDATOR_API_KEY") or os.getenv("LLM_OPS_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("VALIDATOR_BASE_URL") or os.getenv("LLM_OPS_BASE_URL")
    model = os.getenv("VALIDATOR_MODEL") or os.getenv("LLM_OPS_MODEL")
    timeout_seconds = float(os.getenv("VALIDATOR_TIMEOUT_SECONDS", os.getenv("LLM_OPS_TIMEOUT_SECONDS", "300")))
    if not api_key:
        raise RuntimeError("VALIDATOR_API_KEY is not set")
    if api_key.startswith("sk-") and api_key.lower() != "dummy":
        base_url = "https://api.openai.com/v1"
        if not model:
            model = "gpt-4.1-mini"
    elif not base_url:
        base_url = "https://api.openai.com/v1"
    if not model:
        model = "gpt-4o-mini"
    prompt = (
        "You are a strict workflow validator.\n"
        "Judge whether the workflow attempt actually satisfied the original user request.\n"
        "Ignore polish and formatting; focus on substantive correctness and completeness.\n"
        "If the request is not fully satisfied, recommend retrying another workflow option.\n"
        'Return JSON only with keys: {"valid":true|false,"reason":"short reason","retry_recommended":true|false,"missing_requirements":["..."],"trace":["short validation check 1","short validation check 2"]}\n'
        "Workflow payload:\n"
        f"{json.dumps(payload, ensure_ascii=True, indent=2, default=str)}"
    )
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You produce strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    _debug_log("Raw validator response:")
    _debug_log(content)
    blob = _extract_json_object(content)
    if not blob:
        return None
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return _parse_validation_decision(parsed)


def _heuristic_validate(payload: dict):
    status = str(payload.get("workflow_status") or "").strip()
    error = payload.get("error")
    result = payload.get("result")
    steps = payload.get("steps", [])
    completed_steps = [
        step for step in steps
        if isinstance(step, dict) and step.get("status") == "completed"
    ] if isinstance(steps, list) else []
    has_material_result = result not in (None, "", [], {})
    valid = status == "completed" and (has_material_result or bool(completed_steps))
    trace = [
        f"Checked workflow status: {status or 'unknown'}.",
        f"Observed {len(completed_steps)} completed step(s).",
    ]
    if error:
        trace.append(f"Observed execution error: {error}")
    if valid:
        trace.append("The attempt completed with usable output, so it passes validation.")
    else:
        trace.append("The attempt is incomplete or failed to produce enough output, so it should not be accepted.")
    return {
        "valid": valid,
        "reason": "The workflow output satisfies the request." if valid else str(error or "The workflow did not clearly satisfy the request."),
        "retry_recommended": not valid,
        "missing_requirements": [],
        "trace": trace,
    }


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event != "validation.request":
        return {"emits": []}
    try:
        result = _llm_validate(req.payload)
    except Exception as exc:
        _debug_log(f"Validator LLM call failed. Error: {type(exc).__name__}: {exc}")
        result = None
    if result is None:
        result = _heuristic_validate(req.payload)
    return {"emits": [{"event": "validation.result", "payload": result}]}
