import subprocess
from typing import Any

from web_compat import FastAPI

from agent_library.common import EventRequest, EventResponse, serialize_for_stdin, with_node_envelope
from agent_library.reduction import execute_reduction_request, looks_like_safe_reducer_command
from agent_library.template import agent_api, agent_descriptor, emit, noop

app = FastAPI()

AGENT_DESCRIPTOR = agent_descriptor(
    name="data_reducer",
    role="reducer",
    description=(
        "Reduces raw executor outputs into smaller validated summaries using a concrete local "
        "reduction command or a provided fallback reduced result."
    ),
    capability_domains=["data_reduction", "result_normalization", "workflow_graph"],
    action_verbs=["reduce", "summarize", "normalize"],
    side_effect_policy="read_only_local_processing",
    safety_enforced_by_agent=True,
    routing_notes=[
        "Use after executor output is available and a reduction_request or local_reduction_command exists.",
        "Falls back to any existing reduced result when local reduction cannot safely produce output.",
    ],
    apis=[
        agent_api(
            name="reduce_step_output",
            trigger_event="data.reduce",
            emits=["data.reduced"],
            summary="Reduces raw step output into a normalized result for downstream validation and synthesis.",
            when="Runs a local reducer command against raw step output and emits a normalized reduced result.",
            deterministic=True,
            side_effect_level="read_only",
        )
    ],
)
AGENT_METADATA = AGENT_DESCRIPTOR

def _emit_reduced_result(
    payload: dict[str, Any],
    *,
    reduced_result: Any,
    strategy: str,
    command: str = "",
    attempts: int = 0,
    error: str = "",
) -> dict[str, Any]:
    result_payload = {
        "step_id": payload.get("step_id"),
        "task": payload.get("task", ""),
        "original_task": payload.get("original_task", ""),
        "target_agent": payload.get("target_agent", ""),
        "source_event": payload.get("source_event", ""),
        "detail": "Reduced step output." if reduced_result not in (None, "", [], {}) else "No reduced output produced.",
        "reduced_result": reduced_result,
        "strategy": strategy,
        "attempts": attempts,
    }
    if isinstance(command, str) and command.strip():
        result_payload["local_reduction_command"] = command.strip()
    if error:
        result_payload["error"] = error
    return emit("data.reduced", result_payload)


def _run_reduction_command(command: str, input_data: Any) -> tuple[str, str]:
    stdin_text = serialize_for_stdin(input_data)
    completed = subprocess.run(
        command,
        input=stdin_text,
        capture_output=True,
        text=True,
        shell=True,
        timeout=30.0,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip() or f"Reducer exited with status {completed.returncode}."
        raise RuntimeError(error)
    return completed.stdout.strip(), completed.stderr.strip()


@app.post("/handle", response_model=EventResponse)
@with_node_envelope("data_reducer", "reducer")
def handle_event(req: EventRequest):
    if req.event != "data.reduce":
        return noop()

    payload = req.payload if isinstance(req.payload, dict) else {}
    command = str(payload.get("local_reduction_command") or "").strip()
    existing_reduced_result = payload.get("existing_reduced_result")
    input_data = payload.get("input_data")
    reduction_request = payload.get("reduction_request") if isinstance(payload.get("reduction_request"), dict) else None

    if isinstance(reduction_request, dict):
        reduction = execute_reduction_request(reduction_request, input_data)
        if reduction.reduced_result not in (None, "", [], {}):
            return _emit_reduced_result(
                payload,
                reduced_result=reduction.reduced_result,
                strategy=reduction.strategy or "reduction_request",
                command=reduction.local_reduction_command,
                attempts=reduction.attempts,
            )
        if existing_reduced_result not in (None, "", [], {}):
            return _emit_reduced_result(
                payload,
                reduced_result=existing_reduced_result,
                strategy="existing_reduced_result_fallback",
                command=reduction.local_reduction_command,
                attempts=reduction.attempts,
                error=reduction.error,
            )
        return _emit_reduced_result(
            payload,
            reduced_result=None,
            strategy=reduction.strategy or "reduction_request",
            command=reduction.local_reduction_command,
            attempts=reduction.attempts,
            error=reduction.error or "Reduction request produced no output.",
        )

    if command:
        if not looks_like_safe_reducer_command(command):
            if existing_reduced_result not in (None, "", [], {}):
                return _emit_reduced_result(
                    payload,
                    reduced_result=existing_reduced_result,
                    strategy="existing_reduced_result_fallback",
                    command=command,
                    attempts=0,
                    error="Reducer command did not match the allowed local reducer prefixes.",
                )
            return _emit_reduced_result(
                payload,
                reduced_result=None,
                strategy="rejected",
                command=command,
                attempts=0,
                error="Reducer command did not match the allowed local reducer prefixes.",
            )
        try:
            reduced_output, _stderr = _run_reduction_command(command, input_data)
            if reduced_output:
                return _emit_reduced_result(
                    payload,
                    reduced_result=reduced_output,
                    strategy="local_reduction_command",
                    command=command,
                    attempts=1,
                )
            if existing_reduced_result not in (None, "", [], {}):
                return _emit_reduced_result(
                    payload,
                    reduced_result=existing_reduced_result,
                    strategy="existing_reduced_result_fallback",
                    command=command,
                    attempts=1,
                    error="Reducer command produced empty output.",
                )
            return _emit_reduced_result(
                payload,
                reduced_result=None,
                strategy="local_reduction_command",
                command=command,
                attempts=1,
                error="Reducer command produced empty output.",
            )
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            if existing_reduced_result not in (None, "", [], {}):
                return _emit_reduced_result(
                    payload,
                    reduced_result=existing_reduced_result,
                    strategy="existing_reduced_result_fallback",
                    command=command,
                    attempts=1,
                    error=str(exc),
                )
            return _emit_reduced_result(
                payload,
                reduced_result=None,
                strategy="failed",
                command=command,
                attempts=1,
                error=str(exc),
            )

    if existing_reduced_result not in (None, "", [], {}):
        return _emit_reduced_result(
            payload,
            reduced_result=existing_reduced_result,
            strategy="existing_reduced_result",
            attempts=0,
        )

    return _emit_reduced_result(
        payload,
        reduced_result=None,
        strategy="noop",
        attempts=0,
        error="No reduction command or existing reduced result was provided.",
    )
