import json
import os
from typing import Any

import requests
from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse, shared_llm_api_settings
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
            "event": "validation.request",
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
    verdict = raw.get("verdict")
    if not isinstance(valid, bool):
        return None
    if retry_recommended is None:
        retry_recommended = not valid
    if not isinstance(retry_recommended, bool):
        return None
    if not isinstance(reason, str):
        reason = ""
    if not isinstance(verdict, str) or verdict.strip().lower() not in {"valid", "invalid", "uncertain"}:
        verdict = "valid" if valid else "invalid"
    missing_requirements = raw.get("missing_requirements", [])
    if not isinstance(missing_requirements, list):
        missing_requirements = []
    trace = raw.get("trace", [])
    if not isinstance(trace, list):
        trace = []
    return {
        "valid": valid,
        "verdict": verdict.strip().lower(),
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
    api_key, base_url, timeout_seconds, model = shared_llm_api_settings("gpt-4o-mini")
    prompt = (
        "You are a strict workflow validator.\n"
        "Judge whether the workflow attempt actually satisfied the original user request.\n"
        "Ignore polish and formatting; focus on substantive correctness and completeness.\n"
        "If the request is not fully satisfied, recommend retrying another workflow option.\n"
        'Return JSON only with keys: {"valid":true|false,"verdict":"valid|invalid|uncertain","reason":"short reason","retry_recommended":true|false,"missing_requirements":["..."],"trace":["short validation check 1","short validation check 2"]}\n'
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
    task_shape = str(payload.get("task_shape") or "").strip().lower() or "lookup"
    task_text = str(payload.get("task") or "").strip().lower()
    error = payload.get("error")
    result = payload.get("result")
    steps = payload.get("steps", [])
    completed_steps = [
        step for step in steps
        if isinstance(step, dict) and step.get("status") == "completed"
    ] if isinstance(steps, list) else []
    has_material_result = result not in (None, "", [], {})
    verdict = "uncertain"
    reason = "The workflow outcome needs semantic review."
    missing_requirements = []
    if status != "completed":
        verdict = "invalid"
        reason = str(error or "The workflow did not complete successfully.")
    elif task_shape == "count":
        if isinstance(result, (int, float)) or (isinstance(result, str) and result.strip().replace(".", "", 1).isdigit()):
            verdict = "valid"
            reason = "Detected a scalar count result."
        else:
            verdict = "uncertain"
            reason = "Count task completed, but the result is not yet clearly reduced to a scalar."
            missing_requirements = ["scalar count"]
    elif task_shape == "boolean_check":
        if _looks_like_boolean_result(task_text, result):
            verdict = "valid"
            reason = "Detected a boolean-like result."
        else:
            verdict = "invalid"
            reason = "Boolean check did not produce a boolean-like result."
            missing_requirements = ["boolean result"]
    elif task_shape == "save_artifact":
        if isinstance(result, str) and result.strip() and ("/" in result or "\\" in result or "." in result):
            verdict = "valid"
            reason = "Detected an artifact path."
        else:
            verdict = "invalid"
            reason = "Save task did not return an artifact path."
            missing_requirements = ["artifact path"]
    elif task_shape in {"list", "compare"}:
        collection_reason = _collection_result_reason(result)
        if collection_reason:
            verdict = "valid"
            reason = collection_reason
        elif has_material_result:
            verdict = "uncertain"
            reason = "List/compare task has output, but the structure is ambiguous."
            missing_requirements = ["collection-like result"]
        else:
            verdict = "invalid"
            reason = "List/compare task did not produce collection output."
            missing_requirements = ["non-empty collection"]
    elif task_shape in {"lookup", "command_execution", "schema_summary", "summarize_dataset"}:
        if has_material_result or bool(completed_steps):
            verdict = "uncertain" if not has_material_result else "valid"
            reason = "Detected material output." if has_material_result else "Completed steps exist, but semantic sufficiency is still unclear."
            if verdict == "uncertain":
                missing_requirements = ["clear reduced output"]
        else:
            verdict = "invalid"
            reason = str(error or "No usable output detected.")
    valid = verdict == "valid"
    trace = [
        f"Checked workflow status: {status or 'unknown'}.",
        f"Checked task shape: {task_shape}.",
        f"Observed {len(completed_steps)} completed step(s).",
    ]
    if error:
        trace.append(f"Observed execution error: {error}")
    if verdict == "valid":
        trace.append("The attempt completed with usable output, so it passes validation.")
    elif verdict == "uncertain":
        trace.append("The attempt may be correct, but deterministic checks could not confirm it.")
    else:
        trace.append("The attempt is incomplete or failed to produce enough output, so it should not be accepted.")
    return {
        "valid": valid,
        "verdict": verdict,
        "reason": reason,
        "retry_recommended": verdict == "invalid",
        "missing_requirements": missing_requirements,
        "trace": trace,
    }


def _looks_like_boolean_result(task_text: str, result: Any) -> bool:
    if isinstance(result, bool):
        return True
    if not isinstance(result, str):
        return False
    compact = result.strip().lower()
    if compact in {"true", "false", "yes", "no", "0", "1"}:
        return True
    return bool(compact) and any(token in task_text for token in {"installed", "available", "exists", "exist"}) and ("/" in compact or "\\" in compact)


def _collection_result_reason(result: Any) -> str:
    if isinstance(result, list) and result:
        return "Detected a non-empty list result."
    if isinstance(result, dict) and isinstance(result.get("rows"), list) and result["rows"]:
        return "Detected non-empty structured rows."
    if (
        isinstance(result, dict)
        and isinstance(result.get("result"), dict)
        and isinstance(result["result"].get("rows"), list)
        and result["result"]["rows"]
    ):
        return "Detected non-empty structured rows nested in the workflow result."
    if isinstance(result, str):
        lines = [line for line in result.splitlines() if line.strip()]
        if len(lines) > 1:
            return "Detected non-empty multi-line shell output compatible with a list result."
        if lines:
            return "Detected a non-empty singleton result compatible with a list task."
    return ""


def _reduced_validation_payload(payload: dict):
    steps = payload.get("steps", [])
    reduced_steps = []
    if isinstance(steps, list):
        for step in steps[:6]:
            if not isinstance(step, dict):
                continue
            reduced_steps.append(
                {
                    "id": step.get("id"),
                    "task": step.get("task"),
                    "status": step.get("status"),
                    "event": step.get("event"),
                    "error": step.get("error"),
                    "detail": step.get("payload", {}).get("detail") if isinstance(step.get("payload"), dict) else None,
                }
            )
    return {
        "task": payload.get("task", ""),
        "task_shape": payload.get("task_shape", ""),
        "attempt": payload.get("attempt"),
        "total_attempts": payload.get("total_attempts"),
        "option_id": payload.get("option_id"),
        "option_label": payload.get("option_label"),
        "workflow_status": payload.get("workflow_status", ""),
        "error": payload.get("error"),
        "result": payload.get("result"),
        "steps": reduced_steps,
    }


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event != "validation.request":
        return {"emits": []}
    heuristic = _heuristic_validate(req.payload)
    result = heuristic
    llm_budget_remaining = req.payload.get("validation_llm_budget_remaining")
    can_use_llm = not isinstance(llm_budget_remaining, (int, float)) or llm_budget_remaining > 0
    if heuristic.get("verdict") == "uncertain" and can_use_llm:
        try:
            llm_result = _llm_validate(_reduced_validation_payload(req.payload))
        except Exception as exc:
            _debug_log(f"Validator LLM call failed. Error: {type(exc).__name__}: {exc}")
            llm_result = None
        if llm_result is not None:
            result = llm_result
    return {"emits": [{"event": "validation.result", "payload": result}]}
