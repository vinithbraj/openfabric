from datetime import datetime, timezone
import re

from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse, task_plan_context, with_node_envelope
from agent_library.template import agent_api, agent_descriptor, emit, needs_decomposition, noop

app = FastAPI()

AGENT_DESCRIPTOR = agent_descriptor(
    name="notifier",
    role="notifier",
    description="Sends notification messages to configured channels.",
    capability_domains=["notification", "alerting"],
    action_verbs=["notify", "alert", "remind", "send"],
    side_effect_policy="allow_non_destructive_side_effects",
    safety_enforced_by_agent=True,
    routing_notes=["Use for explicit notify/alert/remind intents."],
    apis=[
        agent_api(
            name="send_notification",
            event="task.plan",
            summary="Sends notification payloads derived from task plans.",
            when="Sends notification payloads derived from task plans.",
            intent_tags=["notify", "alert", "reminder"],
            examples=["notify me when done", "send alert deployment finished"],
            deterministic=True,
            side_effect_level="non_destructive_write",
            input_schema={
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "message": {"type": "string"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "detail": {"type": "string"},
                },
            },
        )
    ],
)
AGENT_METADATA = AGENT_DESCRIPTOR


def _extract_message_from_task(task: str):
    match = re.search(
        r"(?:notify|alert|remind)(?:\s+me)?(?:\s+to)?\s+(.+)$",
        task,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return task.strip()


@app.post("/handle", response_model=EventResponse)
@with_node_envelope("notifier", "notifier")
def handle_event(req: EventRequest):
    if req.event == "notify.send":
        channel = req.payload["channel"]
        message = req.payload["message"]
    elif req.event == "task.plan":
        plan_context = task_plan_context(req.payload)
        instruction = req.payload.get("instruction")
        if isinstance(instruction, dict) and instruction.get("operation") == "send_notification":
            channel = instruction.get("channel", "console")
            message = instruction.get("message")
            if not isinstance(message, str) or not message.strip():
                if plan_context.targets("notifier"):
                    return needs_decomposition(
                        "Notifier agent needs an explicit message to send.",
                        suggested_capabilities=["notifier", "shell_runner"],
                    )
                return noop()
            message = message.strip()
            if not isinstance(channel, str) or not channel.strip():
                channel = "console"
        else:
            task = plan_context.classification_task
            if not task:
                return noop()
            task_lc = task.lower()
            if not any(token in task_lc for token in ("notify", "alert", "remind")):
                if plan_context.targets("notifier"):
                    return needs_decomposition(
                        "Notifier agent needs an explicit notify/alert/remind instruction.",
                        suggested_capabilities=["notifier", "shell_runner"],
                    )
                return noop()
            channel = "console"
            message = _extract_message_from_task(plan_context.execution_task)
    else:
        return noop()
    timestamp = datetime.now(timezone.utc).isoformat()
    detail = f"[{timestamp}] delivered to {channel}: {message}"
    return emit("notify.result", {"detail": detail})
