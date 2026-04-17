from pathlib import Path
import re

from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse, task_plan_context

app = FastAPI()

AGENT_METADATA = {
    "description": "Reads workspace files and returns file contents.",
    "capability_domains": ["filesystem", "file_reading", "workspace_inspection"],
    "action_verbs": ["read", "open", "show", "cat"],
    "side_effect_policy": "read_only",
    "safety_enforced_by_agent": True,
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
                        "suggested_capabilities": ["shell_runner", "filesystem"],
                    },
                },
            }
        ]
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
        plan_context = task_plan_context(req.payload)
        instruction = req.payload.get("instruction")
        if isinstance(instruction, dict) and instruction.get("operation") == "read_file":
            raw_path = instruction.get("path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                if plan_context.targets("filesystem"):
                    return _needs_decomposition("Filesystem agent needs a specific file path to read.")
                return {"emits": []}
            raw_path = raw_path.strip()
        else:
            task = plan_context.classification_task
            if not task:
                return {"emits": []}
            raw_path = _extract_filepath(task)
            if not raw_path:
                if plan_context.targets("filesystem"):
                    return _needs_decomposition("Filesystem agent needs a specific file path to read.")
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
