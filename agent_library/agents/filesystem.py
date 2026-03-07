from pathlib import Path

from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse

app = FastAPI()

AGENT_METADATA = {
    "description": "Reads workspace files and returns file contents.",
    "routing_notes": [
        "Use only for opening or reading a specific file path.",
        "Do not use for search/discovery tasks like find, locate, or list files.",
    ],
    "methods": [
        {
            "name": "read_workspace_file",
            "event": "file.read",
            "when": "Reads a relative file path from workspace and emits file content.",
            "intent_tags": ["read_file", "open_file"],
            "examples": ["read agent_library/agents/calculator.py", "open README.md"],
            "anti_patterns": ["find files named vinith", "list all files containing foo"],
        }
    ],
}


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
