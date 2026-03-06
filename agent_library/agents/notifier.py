from datetime import datetime, timezone

from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse

app = FastAPI()


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event != "notify.send":
        return {"emits": []}

    channel = req.payload["channel"]
    message = req.payload["message"]
    timestamp = datetime.now(timezone.utc).isoformat()
    detail = f"[{timestamp}] delivered to {channel}: {message}"
    return {"emits": [{"event": "notify.result", "payload": {"detail": detail}}]}

