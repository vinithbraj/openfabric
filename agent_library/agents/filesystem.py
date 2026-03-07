from pathlib import Path
import re

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
            "event": "task.plan",
            "when": "Reads a relative file path from task plans and emits file content.",
            "intent_tags": ["read_file", "open_file"],
            "examples": ["read agent_library/agents/calculator.py", "open README.md"],
            "anti_patterns": ["find files named vinith", "list all files containing foo"],
        }
    ],
}


def _extract_filepath(task: str):
    match = re.search(r"(?:read|open|show|cat)\s+([./a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9]+)?)", task)
    if match:
        return match.group(1)

    direct_path = re.search(r"\b([./][a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+)\b", task)
    if direct_path:
        return direct_path.group(1)
    return None


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event == "file.read":
        raw_path = req.payload["path"]
    elif req.event == "task.plan":
        task = req.payload.get("task", "")
        if not isinstance(task, str):
            return {"emits": []}
        raw_path = _extract_filepath(task)
        if not raw_path:
            return {"emits": []}
    else:
        return {"emits": []}
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
