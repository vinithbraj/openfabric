from datetime import datetime, timezone
import re

from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse

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
def handle_event(req: EventRequest):
    if req.event == "notify.send":
        channel = req.payload["channel"]
        message = req.payload["message"]
    elif req.event == "task.plan":
        task = req.payload.get("task", "")
        if not isinstance(task, str):
            return {"emits": []}
        task_lc = task.lower()
        if not any(token in task_lc for token in ("notify", "alert", "remind")):
            return {"emits": []}
        channel = "console"
        message = _extract_message_from_task(task)
    else:
        return {"emits": []}
    timestamp = datetime.now(timezone.utc).isoformat()
    detail = f"[{timestamp}] delivered to {channel}: {message}"
    return {"emits": [{"event": "notify.result", "payload": {"detail": detail}}]}
