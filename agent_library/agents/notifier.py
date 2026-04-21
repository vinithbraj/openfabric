from datetime import datetime, timezone
import re

from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse, task_plan_context, with_node_envelope

app = FastAPI()

AGENT_METADATA = {
    "description": "Sends notification messages to configured channels.",
    "capability_domains": ["notification", "alerting"],
    "action_verbs": ["notify", "alert", "remind", "send"],
    "side_effect_policy": "allow_non_destructive_side_effects",
    "safety_enforced_by_agent": True,
    "routing_notes": [
        "Use for explicit notify/alert/remind intents.",
    ],
    "methods": [
        {
            "name": "send_notification",
            "event": "task.plan",
            "when": "Sends notification payloads derived from task plans.",
            "intent_tags": ["notify", "alert", "reminder"],
            "examples": ["notify me when done", "send alert deployment finished"],
        }
    ],
}


def _needs_decomposition(detail: str):
    return {
        "emits": [
            {
                "event": "task.result",
                "payload": {
                    "detail": detail,
                    "status": "needs_decomposition",
                    "error": detail,
                    "replan_hint": {
                        "reason": detail,
                        "failure_class": "needs_decomposition",
                        "suggested_capabilities": ["notifier", "shell_runner"],
                    },
                },
            }
        ]
    }


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
                    return _needs_decomposition("Notifier agent needs an explicit message to send.")
                return {"emits": []}
            message = message.strip()
            if not isinstance(channel, str) or not channel.strip():
                channel = "console"
        else:
            task = plan_context.classification_task
            if not task:
                return {"emits": []}
            task_lc = task.lower()
            if not any(token in task_lc for token in ("notify", "alert", "remind")):
                if plan_context.targets("notifier"):
                    return _needs_decomposition("Notifier agent needs an explicit notify/alert/remind instruction.")
                return {"emits": []}
            channel = "console"
            message = _extract_message_from_task(plan_context.execution_task)
    else:
        return {"emits": []}
    timestamp = datetime.now(timezone.utc).isoformat()
    detail = f"[{timestamp}] delivered to {channel}: {message}"
    return {"emits": [{"event": "notify.result", "payload": {"detail": detail}}]}
