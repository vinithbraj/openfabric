import re

from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse, with_node_envelope
from agent_library.template import agent_api, agent_descriptor, emit_sequence, noop, task_result

app = FastAPI()

AGENT_DESCRIPTOR = agent_descriptor(
    name="ops_planner",
    role="router",
    description="Rule-based planner for file, shell, and notification operations.",
    capability_domains=["planning", "routing", "operations"],
    action_verbs=["plan", "route", "dispatch"],
    side_effect_policy="read_only",
    safety_enforced_by_agent=True,
    routing_notes=[
        "Use for simple rule-based routing of file reads, shell commands, and notifications.",
        "Falls back to task.result guidance when it cannot identify an actionable operation.",
    ],
    apis=[
        agent_api(
            name="plan_file_read",
            event="file.read",
            summary="Routes file read/open requests to the filesystem path.",
            when="Use for reading/opening files from user request.",
            deterministic=True,
            side_effect_level="read_only",
        ),
        agent_api(
            name="plan_cli_exec",
            event="shell.exec",
            summary="Routes explicit shell execution requests.",
            when="Use for shell command execution requests.",
            deterministic=True,
            side_effect_level="read_only",
        ),
        agent_api(
            name="plan_notification",
            event="notify.send",
            summary="Routes notify or alert requests to the notifier.",
            when="Use for notify/alert requests.",
            deterministic=True,
            side_effect_level="read_only",
        ),
        agent_api(
            name="planner_fallback",
            event="task.result",
            summary="Explains how to phrase an actionable operations request.",
            when="Use only when no actionable tool event applies.",
            deterministic=True,
            side_effect_level="read_only",
        ),
    ],
)
AGENT_METADATA = AGENT_DESCRIPTOR


def _extract_filepath(question: str):
    match = re.search(r"(?:read|open)\s+([./a-zA-Z0-9_-]+\.[a-zA-Z0-9]+)", question)
    if match:
        return match.group(1)
    return None


def _extract_command(question: str):
    match = re.search(r"(?:run|execute)\s+`([^`]+)`", question)
    if match:
        return match.group(1)
    return None


@app.post("/handle", response_model=EventResponse)
@with_node_envelope("ops_planner", "router")
def handle_event(req: EventRequest):
    if req.event != "user.ask":
        return noop()

    question = req.payload["question"]
    question_lc = question.lower()
    emits = []

    filepath = _extract_filepath(question)
    if filepath:
        emits.append({"event": "file.read", "payload": {"path": filepath}})

    command = _extract_command(question)
    if command:
        emits.append({"event": "shell.exec", "payload": {"command": command}})

    if "notify" in question_lc or "alert" in question_lc:
        emits.append(
            {
                "event": "notify.send",
                "payload": {
                    "channel": "console",
                    "message": f"Notification requested: {question}",
                },
            }
        )

    if not emits:
        return task_result(
            "No operations detected. Use phrases like 'read <file>', 'run `<command>`', or 'notify ...'."
        )

    return emit_sequence(emits)
