from pathlib import Path

from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse

app = FastAPI()


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event != "file.read":
        return {"emits": []}

    raw_path = req.payload["path"]
    base = Path(".").resolve()
    target = (base / raw_path).resolve()

    if not str(target).startswith(str(base)):
        return {
            "emits": [
                {
                    "event": "task.result",
                    "payload": {"detail": f"Blocked path outside workspace: {raw_path}"},
                }
            ]
        }

    if not target.exists():
        return {
            "emits": [
                {
                    "event": "task.result",
                    "payload": {"detail": f"File not found: {raw_path}"},
                }
            ]
        }

    if not target.is_file():
        return {
            "emits": [
                {
                    "event": "task.result",
                    "payload": {"detail": f"Not a file: {raw_path}"},
                }
            ]
        }

    content = target.read_text(encoding="utf-8", errors="replace")[:4000]
    return {
        "emits": [
            {"event": "file.content", "payload": {"path": raw_path, "content": content}}
        ]
    }

