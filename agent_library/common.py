import subprocess
import os
import json
from dataclasses import dataclass
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

from pydantic import BaseModel


class EventRequest(BaseModel):
    event: str
    payload: Dict


class EmittedEvent(BaseModel):
    event: str
    payload: Dict


class EventResponse(BaseModel):
    emits: List[EmittedEvent]


@dataclass(frozen=True)
class TaskPlanContext:
    step_task: str
    original_task: str
    target_agent: str
    previous_step_result: Dict[str, Any]
    prior_step_results: List[Dict[str, Any]]
    dependency_results: List[Dict[str, Any]]

    @property
    def structured_context(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"current_step": self.step_task}
        if self.original_task and self.original_task.strip() != self.step_task.strip():
            payload["original_task"] = self.original_task.strip()
        if self.previous_step_result:
            payload["previous_step_result"] = self.previous_step_result
        if self.prior_step_results:
            payload["prior_step_results"] = self.prior_step_results
        if self.dependency_results:
            payload["dependency_results"] = self.dependency_results
        return payload

    @property
    def classification_task(self) -> str:
        return self.step_task

    @property
    def execution_task(self) -> str:
        sections = [f"Current workflow step: {self.step_task.strip()}"]
        if self.original_task and self.original_task.strip() != self.step_task.strip():
            sections.extend(["Original user request:", self.original_task.strip()])
        context_payload = self.structured_context
        if context_payload:
            sections.extend(["Structured workflow context JSON:", _to_json(context_payload)])
        return "\n".join(sections)

    def targets(self, agent_name_or_family: str) -> bool:
        target = self.target_agent.strip()
        candidate = agent_name_or_family.strip()
        if not target or not candidate:
            return False
        return target == candidate or target.startswith(f"{candidate}_")


def task_plan_context(payload: Dict[str, Any]) -> TaskPlanContext:
    step_task = payload.get("task")
    original_task = payload.get("original_task")
    target_agent = payload.get("target_agent")
    previous_step_result = payload.get("previous_step_result")
    prior_step_results = payload.get("prior_step_results")
    dependency_results = payload.get("dependency_results")
    return TaskPlanContext(
        step_task=step_task.strip() if isinstance(step_task, str) else "",
        original_task=original_task.strip() if isinstance(original_task, str) else "",
        target_agent=target_agent.strip() if isinstance(target_agent, str) else "",
        previous_step_result=previous_step_result if isinstance(previous_step_result, dict) else {},
        prior_step_results=[item for item in prior_step_results if isinstance(item, dict)] if isinstance(prior_step_results, list) else [],
        dependency_results=[item for item in dependency_results if isinstance(item, dict)] if isinstance(dependency_results, list) else [],
    )


def _to_json(value: Any) -> str:
    return json_dumps(value)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, indent=2, default=str)


@lru_cache(maxsize=16)
def _list_openai_compatible_models(base_url: str, api_key: str, timeout_seconds: float) -> tuple[str, ...]:
    models_url = f"{base_url.rstrip('/')}/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(models_url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, HTTPError, TimeoutError, ValueError, json.JSONDecodeError):
        return ()

    data = payload.get("data")
    if not isinstance(data, list):
        return ()
    model_ids: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id.strip():
            model_ids.append(model_id.strip())
    return tuple(model_ids)


def _discover_openai_compatible_model(base_url: str, api_key: str, timeout_seconds: float) -> str | None:
    model_ids = _list_openai_compatible_models(base_url, api_key, timeout_seconds)
    return model_ids[0] if model_ids else None


def shared_llm_api_settings(
    default_model: str = "gpt-4o-mini",
    *,
    timeout_seconds: float | None = None,
) -> tuple[str, str, float, str]:
    api_key = os.getenv("LLM_OPS_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_OPS_BASE_URL")
    explicit_model = os.getenv("LLM_OPS_MODEL")
    timeout = timeout_seconds
    if timeout is None:
        raw_timeout = os.getenv("LLM_OPS_TIMEOUT_SECONDS", "300")
        try:
            timeout = float(raw_timeout)
        except ValueError:
            timeout = 300.0

    if api_key and api_key.startswith("sk-") and api_key.lower() != "dummy":
        model = explicit_model or default_model
        return api_key, "https://api.openai.com/v1", float(timeout), model

    resolved_base_url = (base_url or "http://127.0.0.1:8000/v1").rstrip("/")
    resolved_api_key = api_key or "dummy"
    available_models = _list_openai_compatible_models(
        resolved_base_url,
        resolved_api_key,
        float(timeout),
    )
    discovered_model = available_models[0] if available_models else None
    normalized_explicit_model = explicit_model.strip() if isinstance(explicit_model, str) and explicit_model.strip() else None
    if normalized_explicit_model and available_models and normalized_explicit_model not in available_models:
        normalized_explicit_model = discovered_model
    model = normalized_explicit_model or discovered_model or default_model
    return resolved_api_key, resolved_base_url, float(timeout), model


@dataclass(frozen=True)
class LocalReductionResult:
    output: str = ""
    command: str = ""
    attempts: int = 0
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return bool(self.output and self.command)


def _nonempty(value: Any) -> bool:
    return value not in (None, "", [], {})


def _derived_node_status(role: str, emitted_payload: Dict[str, Any]) -> str:
    status = emitted_payload.get("status")
    if isinstance(status, str) and status.strip():
        return status.strip()
    if role == "validator":
        verdict = emitted_payload.get("verdict")
        if isinstance(verdict, str) and verdict.strip():
            return verdict.strip()
        if emitted_payload.get("valid") is True:
            return "valid"
        if emitted_payload.get("valid") is False:
            return "invalid"
    if role == "reducer":
        if _nonempty(emitted_payload.get("reduced_result")):
            return "completed"
        if _nonempty(emitted_payload.get("error")):
            return "failed"
        return "noop"
    return "completed"


def build_node_envelope(
    request_event: str,
    request_payload: Dict[str, Any] | None,
    emitted_event: str,
    emitted_payload: Dict[str, Any] | None,
    *,
    agent_name: str,
    role: str,
) -> Dict[str, Any]:
    request_payload = request_payload if isinstance(request_payload, dict) else {}
    emitted_payload = emitted_payload if isinstance(emitted_payload, dict) else {}
    request_node = request_payload.get("node") if isinstance(request_payload.get("node"), dict) else {}
    emitted_node = emitted_payload.get("node") if isinstance(emitted_payload.get("node"), dict) else {}
    instruction = request_payload.get("instruction") if isinstance(request_payload.get("instruction"), dict) else {}
    reduction_request = request_payload.get("reduction_request") if isinstance(request_payload.get("reduction_request"), dict) else {}

    operation = None
    if isinstance(instruction.get("operation"), str) and instruction.get("operation").strip():
        operation = instruction.get("operation").strip()
    elif isinstance(request_payload.get("operation"), str) and request_payload.get("operation").strip():
        operation = request_payload.get("operation").strip()
    elif isinstance(request_payload.get("validation_scope"), str) and request_payload.get("validation_scope").strip():
        operation = f"validate_{request_payload.get('validation_scope').strip()}"
    elif isinstance(reduction_request.get("kind"), str) and reduction_request.get("kind").strip():
        operation = reduction_request.get("kind").strip()
    elif request_event == "data.reduce":
        operation = "reduce_step_output"

    if isinstance(request_payload.get("validation_scope"), str) and request_payload.get("validation_scope").strip():
        scope = request_payload.get("validation_scope").strip()
    elif _nonempty(request_payload.get("step_id")) or _nonempty(emitted_payload.get("step_id")):
        scope = "step"
    else:
        scope = "workflow"

    envelope = {
        "agent": agent_name,
        "role": role,
        "request_event": request_event,
        "emitted_event": emitted_event,
        "run_id": request_payload.get("run_id") or emitted_payload.get("run_id") or request_node.get("run_id") or emitted_node.get("run_id"),
        "attempt": request_payload.get("attempt") or emitted_payload.get("attempt") or request_node.get("attempt") or emitted_node.get("attempt"),
        "step_id": request_payload.get("step_id") or emitted_payload.get("step_id") or request_node.get("step_id") or emitted_node.get("step_id"),
        "task": request_payload.get("task") or emitted_payload.get("task") or request_node.get("task") or emitted_node.get("task"),
        "original_task": request_payload.get("original_task") or emitted_payload.get("original_task") or request_node.get("original_task") or emitted_node.get("original_task"),
        "target_agent": request_payload.get("target_agent") or emitted_payload.get("target_agent") or request_node.get("target_agent") or emitted_node.get("target_agent"),
        "scope": scope,
        "operation": operation or request_node.get("operation") or emitted_node.get("operation"),
        "status": _derived_node_status(role, emitted_payload),
    }
    sanitized = {key: value for key, value in envelope.items() if _nonempty(value)}
    merged = dict(request_node)
    merged.update(emitted_node)
    merged.update(sanitized)
    return merged


def attach_node_envelopes(
    response: Dict[str, Any],
    request_event: str,
    request_payload: Dict[str, Any] | None,
    *,
    agent_name: str,
    role: str,
) -> Dict[str, Any]:
    if not isinstance(response, dict):
        return response
    emits = response.get("emits")
    if not isinstance(emits, list):
        return response

    updated_emits: List[Dict[str, Any]] = []
    for item in emits:
        if not isinstance(item, dict):
            updated_emits.append(item)
            continue
        event_name = item.get("event")
        payload = item.get("payload")
        if not isinstance(event_name, str) or not isinstance(payload, dict):
            updated_emits.append(item)
            continue
        updated_payload = dict(payload)
        updated_payload["node"] = build_node_envelope(
            request_event,
            request_payload,
            event_name,
            updated_payload,
            agent_name=agent_name,
            role=role,
        )
        updated_emits.append({"event": event_name, "payload": updated_payload})
    updated_response = dict(response)
    updated_response["emits"] = updated_emits
    return updated_response


def with_node_envelope(agent_name: str, role: str):
    def decorator(func):
        @wraps(func)
        def wrapper(req: EventRequest, *args, **kwargs):
            response = func(req, *args, **kwargs)
            payload = req.payload if isinstance(req.payload, dict) else {}
            return attach_node_envelopes(
                response,
                req.event,
                payload,
                agent_name=agent_name,
                role=role,
            )

        return wrapper

    return decorator


def serialize_for_stdin(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple, int, float, bool)):
        return json_dumps(value)
    return str(value)


def run_local_reducer_loop(
    input_data: Any,
    generate_command: Callable[[str, str], str],
    *,
    max_attempts: int = 3,
    timeout_seconds: float = 30.0,
    validate_output: Callable[[str], bool] | None = None,
) -> LocalReductionResult:
    stdin_text = serialize_for_stdin(input_data)
    previous_command = ""
    previous_error = ""

    for attempt in range(1, max_attempts + 1):
        command = generate_command(previous_command, previous_error)
        if not isinstance(command, str) or not command.strip():
            break
        command = command.strip()
        previous_command = command

        try:
            completed = subprocess.run(
                command,
                input=stdin_text,
                capture_output=True,
                text=True,
                shell=True,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            previous_error = f"{type(exc).__name__}: {exc}"
            continue

        if completed.returncode != 0:
            previous_error = completed.stderr.strip() or completed.stdout.strip() or f"Reducer exited with {completed.returncode}"
            continue

        output = completed.stdout.strip()
        if validate_output and not validate_output(output):
            previous_error = "Reducer produced empty or invalid output."
            continue

        return LocalReductionResult(output=output, command=command, attempts=attempt)

    return LocalReductionResult(command=previous_command, attempts=max_attempts, error=previous_error)
