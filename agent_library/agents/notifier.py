from datetime import datetime, timezone

from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse

app = FastAPI()

AGENT_METADATA = {
    "description": "Sends notification messages to configured channels.",
    "routing_notes": [
        "Use for explicit notify/alert/remind intents.",
    ],
    "methods": [
        {
            "name": "send_notification",
            "event": "notify.send",
            "when": "Sends notification payloads to configured notification channel.",
            "intent_tags": ["notify", "alert", "reminder"],
            "examples": ["notify me when done", "send alert deployment finished"],
        }
    ],
}


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event != "notify.send":
        return {"emits": []}

    channel = req.payload["channel"]
    message = req.payload["message"]
    timestamp = datetime.now(timezone.utc).isoformat()
    detail = f"[{timestamp}] delivered to {channel}: {message}"
    return {"emits": [{"event": "notify.result", "payload": {"detail": detail}}]}
