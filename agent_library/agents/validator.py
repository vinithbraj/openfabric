import json
import os
import re
from typing import Any

import requests
from web_compat import FastAPI

from agent_library.common import EventRequest, EventResponse, shared_llm_api_settings, with_node_envelope
from agent_library.template import agent_api, agent_descriptor, emit, noop
from runtime.console import log_debug

app = FastAPI()

AGENT_DESCRIPTOR = agent_descriptor(
    name="validator",
    role="validator",
    description="Validates whether a workflow attempt actually satisfied the user's request and explains retry decisions.",
    capability_domains=["validation", "quality_control", "workflow_guardrails"],
    action_verbs=["validate", "verify", "gate"],
    side_effect_policy="read_only",
    safety_enforced_by_agent=True,
    routing_notes=[
        "Use after step execution or workflow completion to decide whether the result should be accepted, retried, or replanned.",
        "Uses deterministic heuristics first and an LLM only when the verdict remains uncertain and budget allows.",
    ],
    apis=[
        agent_api(
            name="validate_workflow_attempt",
            trigger_event="validation.request",
            emits=["validation.result"],
            summary="Validates step-level or workflow-level execution results.",
            when="Checks whether an execution attempt answered the user's request and whether another workflow option should be tried.",
            deterministic=False,
            side_effect_level="read_only",
        )
    ],
)
AGENT_METADATA = AGENT_DESCRIPTOR


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
    task_text = str(payload.get("task") or "").strip().lower()
    task_shape = _infer_step_task_shape(task_text, str(payload.get("task_shape") or ""))
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
    elif _task_requires_count_and_state(task_text):
        has_count = _structured_count_like(result)
        has_state = _contains_state_evidence(result)
        if has_count and has_state:
            verdict = "valid"
            reason = "Detected both count and state information."
        else:
            verdict = "invalid"
            reason = "The workflow did not include both the requested count and state information."
            if not has_count:
                missing_requirements.append("count")
            if not has_state:
                missing_requirements.append("state breakdown")
    elif _task_requires_count_and_identifiers(task_text):
        evidence_candidates = _workflow_evidence_candidates(payload, result)
        has_count = any(_structured_count_like(candidate) for candidate in evidence_candidates)
        has_identifiers = any(_contains_identifier_evidence(candidate) for candidate in evidence_candidates)
        if has_count and has_identifiers:
            verdict = "valid"
            reason = "Detected both count and identifier information."
        else:
            verdict = "invalid"
            reason = "The workflow did not include both the requested count and identifier list."
            if not has_count:
                missing_requirements.append("count")
            if not has_identifiers:
                missing_requirements.append("identifier list")
    elif task_shape == "count":
        if _structured_count_like(result):
            verdict = "valid"
            reason = "Detected count-like output."
        else:
            verdict = "uncertain"
            reason = "Count task completed, but the result is not yet clearly count-like."
            missing_requirements = ["count-like result"]
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
    if _task_requires_count_and_state(task_text):
        trace.append("Detected compound count/state intent from the workflow task.")
    if _task_requires_count_and_identifiers(task_text):
        trace.append("Detected compound count/identifier intent from the workflow task.")
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


def _infer_step_task_shape(step_task: str, fallback_shape: str) -> str:
    text = str(step_task or "").strip().lower()
    if any(token in text for token in ("save ", "write ", "create ")) and any(
        token in text for token in (" file", " path", ".txt", ".csv", ".json", ".md")
    ):
        return "save_artifact"
    if any(token in text for token in ("how many", "count ", "number of", "total ")) or text.startswith("count "):
        return "count"
    if any(token in text for token in ("whether", "check whether", "if any", "exists", "is there", "does ", "do any", "has ")) and not any(
        token in text for token in ("list ", "show ", "display ")
    ):
        return "boolean_check"
    if any(token in text for token in ("schema", "schemas", "relationship", "relationships", "foreign key", "column ", "columns")):
        return "schema_summary"
    if any(token in text for token in ("compare", "difference", "versus", " vs ")):
        return "compare"
    if any(token in text for token in ("summarize", "summary", "overview")):
        return "summarize_dataset"
    if any(token in text for token in ("list ", "show ", "find ", "which ", "display ", "sample rows")):
        return "list"
    return str(fallback_shape or "").strip().lower() or "lookup"


def _looks_like_numeric_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        compact = value.strip()
        return bool(compact) and compact.replace(".", "", 1).isdigit()
    return False


def _structured_count_like(result: Any) -> bool:
    if _looks_like_numeric_scalar(result):
        return True
    if isinstance(result, str):
        lower = result.lower()
        if ("total" in lower or "count" in lower or "matching" in lower) and re.search(r"\d", lower):
            return True
    if isinstance(result, list) and len(result) == 1:
        row = result[0]
        if isinstance(row, dict):
            numeric_values = [value for value in row.values() if _looks_like_numeric_scalar(value)]
            if len(numeric_values) == 1:
                return True
    if isinstance(result, dict):
        for key in ("count", "total", "matching_jobs", "matching job", "total_nodes", "jobs_considered"):
            if key in result and _looks_like_numeric_scalar(result.get(key)):
                return True
        rows = result.get("rows")
        if _structured_count_like(rows):
            return True
        nested = result.get("result")
        if nested is not None and nested is not result and _structured_count_like(nested):
            return True
        reduced = result.get("reduced_result") or result.get("refined_answer")
        if isinstance(reduced, str):
            lower = reduced.lower()
            if ("total" in lower or "count" in lower or "matching" in lower) and re.search(r"\d", lower):
                return True
    return False


def _contains_state_evidence(result: Any) -> bool:
    try:
        text = json.dumps(result, ensure_ascii=True, default=str).lower()
    except TypeError:
        text = str(result).lower()
    return any(
        token in text
        for token in (
            "state",
            "status",
            "idle",
            "mixed",
            "allocated",
            "running",
            "pending",
            "failed",
            "completed",
            "down",
            "drain",
        )
    )


def _contains_identifier_evidence(result: Any) -> bool:
    if isinstance(result, str):
        lines = [line.strip() for line in result.splitlines() if line.strip()]
        if len(lines) >= 2 and all(re.fullmatch(r"[A-Za-z0-9_.:-]+", line) for line in lines):
            return True
        if any("job id" in line.lower() for line in lines):
            return True
        return False
    if isinstance(result, list):
        return any(_contains_identifier_evidence(item) for item in result)
    if isinstance(result, dict):
        for key in ("job_id", "job_ids", "ids", "identifier", "identifiers"):
            candidate = result.get(key)
            if candidate not in (None, "", [], {}):
                if _contains_identifier_evidence(candidate):
                    return True
        excerpt = result.get("excerpt")
        if isinstance(excerpt, str) and _contains_identifier_evidence(excerpt):
            return True
        rows = result.get("rows")
        if isinstance(rows, list) and rows:
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for key, candidate in row.items():
                    if "id" in str(key).lower() and candidate not in (None, "", [], {}):
                        return True
        nested = result.get("result")
        if nested is not None and nested is not result and _contains_identifier_evidence(nested):
            return True
        reduced = result.get("reduced_result") or result.get("refined_answer")
        if isinstance(reduced, str) and _contains_identifier_evidence(reduced):
            return True
    return False


def _looks_like_collection_output(result: Any) -> bool:
    if isinstance(result, list):
        return True
    if isinstance(result, dict):
        if isinstance(result.get("rows"), list):
            return True
        nested = result.get("result")
        if nested is not None and nested is not result:
            return _looks_like_collection_output(nested)
        columns = result.get("columns")
        if isinstance(columns, list):
            return True
    if isinstance(result, str):
        lines = [line for line in result.splitlines() if line.strip()]
        return bool(lines)
    return False


def _looks_like_schema_output(result: Any) -> bool:
    if isinstance(result, dict):
        if any(key in result for key in ("schema", "schemas", "tables", "columns")):
            return True
        rows = result.get("rows")
        if isinstance(rows, list):
            return True
        nested = result.get("result")
        if nested is not None and nested is not result and _looks_like_schema_output(nested):
            return True
    try:
        text = json.dumps(result, ensure_ascii=True, default=str).lower()
    except TypeError:
        text = str(result).lower()
    return any(token in text for token in ("schema", "table", "column", "relationship", "foreign"))


def _has_material_output(result: Any) -> bool:
    if result not in (None, "", [], {}):
        return True
    if isinstance(result, dict):
        nested = result.get("result")
        if nested is not None and nested is not result:
            return _has_material_output(nested)
    return False


def _task_requires_count_and_state(step_task: str) -> bool:
    text = str(step_task or "").lower()
    return any(token in text for token in ("how many", "count ", "number of", "total ")) and any(
        token in text for token in (" state", " states", "status", "breakdown")
    )


def _task_requires_count_and_identifiers(task_text: str) -> bool:
    text = str(task_text or "").lower()
    has_count = any(token in text for token in ("how many", "count ", "number of", "total "))
    has_identifier_request = "job id" in text or bool(re.search(r"\bids\b", text))
    return has_count and has_identifier_request


def _workflow_evidence_candidates(payload: dict, result: Any) -> list[Any]:
    candidates: list[Any] = []
    if result not in (None, "", [], {}):
        candidates.append(result)
    steps = payload.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict) or step.get("status") != "completed":
                continue
            for key in ("result", "value", "summary", "payload"):
                candidate = step.get(key)
                if candidate not in (None, "", [], {}):
                    candidates.append(candidate)
    available_context = payload.get("available_context")
    if isinstance(available_context, dict):
        step_results = available_context.get("__step_results__")
        if isinstance(step_results, list):
            for step in step_results:
                if not isinstance(step, dict) or step.get("status") != "completed":
                    continue
                for key in ("value", "result", "summary", "evidence"):
                    candidate = step.get(key)
                    if candidate not in (None, "", [], {}):
                        candidates.append(candidate)
    return candidates


def _heuristic_validate_step(payload: dict):
    status = str(payload.get("workflow_status") or "").strip()
    step_task = str(payload.get("step_task") or payload.get("task") or "").strip()
    task_shape = _infer_step_task_shape(step_task, str(payload.get("task_shape") or ""))
    result = payload.get("result")
    step_value = payload.get("step_value")
    candidate = step_value if step_value not in (None, "", [], {}) else result
    event = str(payload.get("step_event") or "").strip()
    trace = [
        f"Checked step status: {status or 'unknown'}.",
        f"Checked step task shape: {task_shape}.",
        f"Checked step event: {event or 'unknown'}.",
    ]
    if status != "completed":
        trace.append("The step did not complete successfully, so it cannot be accepted.")
        return {
            "valid": False,
            "verdict": "invalid",
            "reason": "The step did not complete successfully.",
            "retry_recommended": True,
            "missing_requirements": ["completed step"],
            "trace": trace,
        }

    if _task_requires_count_and_state(step_task):
        has_count = _structured_count_like(candidate)
        has_state = _contains_state_evidence(candidate)
        trace.append(f"Count evidence detected: {has_count}.")
        trace.append(f"State evidence detected: {has_state}.")
        if has_count and has_state:
            trace.append("The step output contains both count and state information.")
            return {
                "valid": True,
                "verdict": "valid",
                "reason": "The step output contains both count and state information.",
                "retry_recommended": False,
                "missing_requirements": [],
                "trace": trace,
            }
        missing = []
        if not has_count:
            missing.append("count")
        if not has_state:
            missing.append("state breakdown")
        trace.append("The step output missed required combined count/state information.")
        return {
            "valid": False,
            "verdict": "invalid",
            "reason": "The step output did not include both the requested count and state information.",
            "retry_recommended": True,
            "missing_requirements": missing,
            "trace": trace,
        }

    if task_shape == "save_artifact":
        valid = isinstance(candidate, str) and candidate.strip() and ("/" in candidate or "\\" in candidate or "." in candidate)
        trace.append(f"Artifact path detected: {valid}.")
        return {
            "valid": valid,
            "verdict": "valid" if valid else "invalid",
            "reason": "Detected an artifact path." if valid else "The step did not return an artifact path.",
            "retry_recommended": not valid,
            "missing_requirements": [] if valid else ["artifact path"],
            "trace": trace,
        }

    if task_shape == "count":
        valid = _structured_count_like(candidate)
        trace.append(f"Count evidence detected: {valid}.")
        return {
            "valid": valid,
            "verdict": "valid" if valid else "invalid",
            "reason": "Detected count-like output." if valid else "The step did not produce count-like output.",
            "retry_recommended": not valid,
            "missing_requirements": [] if valid else ["count-like output"],
            "trace": trace,
        }

    if task_shape == "boolean_check":
        valid = _looks_like_boolean_result(step_task.lower(), candidate)
        trace.append(f"Boolean evidence detected: {valid}.")
        return {
            "valid": valid,
            "verdict": "valid" if valid else "invalid",
            "reason": "Detected boolean-like output." if valid else "The step did not produce boolean-like output.",
            "retry_recommended": not valid,
            "missing_requirements": [] if valid else ["boolean-like output"],
            "trace": trace,
        }

    if task_shape == "schema_summary":
        valid = _looks_like_schema_output(candidate)
        trace.append(f"Schema evidence detected: {valid}.")
        return {
            "valid": valid,
            "verdict": "valid" if valid else "invalid",
            "reason": "Detected schema-oriented output." if valid else "The step did not produce schema-oriented output.",
            "retry_recommended": not valid,
            "missing_requirements": [] if valid else ["schema/table/column summary"],
            "trace": trace,
        }

    if task_shape in {"list", "compare"}:
        valid = _looks_like_collection_output(candidate)
        trace.append(f"Collection evidence detected: {valid}.")
        return {
            "valid": valid,
            "verdict": "valid" if valid else "invalid",
            "reason": "Detected collection-like output." if valid else "The step did not produce collection-like output.",
            "retry_recommended": not valid,
            "missing_requirements": [] if valid else ["collection-like output"],
            "trace": trace,
        }

    if task_shape == "summarize_dataset":
        valid = isinstance(candidate, str) and bool(candidate.strip())
        trace.append(f"Summary text detected: {valid}.")
        return {
            "valid": valid,
            "verdict": "valid" if valid else "invalid",
            "reason": "Detected summary text." if valid else "The step did not produce summary text.",
            "retry_recommended": not valid,
            "missing_requirements": [] if valid else ["summary text"],
            "trace": trace,
        }

    valid = _has_material_output(candidate)
    trace.append(f"Material output detected: {valid}.")
    return {
        "valid": valid,
        "verdict": "valid" if valid else "invalid",
        "reason": "Detected material output." if valid else "The step did not produce usable output.",
        "retry_recommended": not valid,
        "missing_requirements": [] if valid else ["usable output"],
        "trace": trace,
    }


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
    if str(payload.get("validation_scope") or "").strip().lower() == "step":
        return {
            "validation_scope": "step",
            "task": payload.get("task", ""),
            "original_task": payload.get("original_task", ""),
            "task_shape": payload.get("task_shape", ""),
            "workflow_status": payload.get("workflow_status", ""),
            "step_id": payload.get("step_id"),
            "step_task": payload.get("step_task"),
            "step_target_agent": payload.get("step_target_agent"),
            "step_event": payload.get("step_event"),
            "step_value": payload.get("step_value"),
            "result": payload.get("result"),
        }
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
@with_node_envelope("validator", "validator")
def handle_event(req: EventRequest):
    if req.event != "validation.request":
        return noop()
    if str(req.payload.get("validation_scope") or "").strip().lower() == "step":
        heuristic = _heuristic_validate_step(req.payload)
    else:
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
    return emit("validation.result", result)
