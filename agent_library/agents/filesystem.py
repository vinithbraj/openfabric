from pathlib import Path
import re

from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse, task_plan_context, with_node_envelope
from agent_library.template import agent_api, agent_descriptor, emit, needs_decomposition, noop, task_result

app = FastAPI()

AGENT_DESCRIPTOR = agent_descriptor(
    name="filesystem",
    role="filesystem",
    description="Reads workspace files and returns file contents.",
    capability_domains=["filesystem", "file_reading", "workspace_inspection"],
    action_verbs=["read", "open", "show", "cat"],
    side_effect_policy="read_only",
    safety_enforced_by_agent=True,
    routing_notes=[
        "Use only for opening or reading a specific file path.",
        "Do not use for search/discovery tasks like find, locate, or list files.",
    ],
    apis=[
        agent_api(
            name="read_workspace_file",
            trigger_event="task.plan",
            emits=["file.content", "task.result"],
            summary="Reads a relative file path from task plans and emits file content.",
            when="Reads a relative file path from task plans and emits file content.",
            intent_tags=["read_file", "open_file"],
            examples=["read agent_library/agents/calculator.py", "open README.md"],
            anti_patterns=["find files named vinith", "list all files containing foo"],
            deterministic=True,
            side_effect_level="read_only",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        )
    ],
)
AGENT_METADATA = AGENT_DESCRIPTOR


def _extract_filepath(task: str):
    match = re.search(r"(?:read|open|show|cat)\s+([./a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9]+)?)", task)
    if match:
        return match.group(1)

    direct_path = re.search(r"\b([./][a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+)\b", task)
    if direct_path:
        return direct_path.group(1)
    return None


@app.post("/handle", response_model=EventResponse)
@with_node_envelope("filesystem", "filesystem")
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
                    return needs_decomposition(
                        "Filesystem agent needs a specific file path to read.",
                        suggested_capabilities=["shell_runner", "filesystem"],
                    )
                return noop()
            raw_path = raw_path.strip()
        else:
            task = plan_context.classification_task
            if not task:
                return noop()
            raw_path = _extract_filepath(task)
            if not raw_path:
                if plan_context.targets("filesystem"):
                    return needs_decomposition(
                        "Filesystem agent needs a specific file path to read.",
                        suggested_capabilities=["shell_runner", "filesystem"],
                    )
                return noop()
    else:
        return noop()
    base = Path(".").resolve()
    target = (base / raw_path).resolve()

    if not str(target).startswith(str(base)):
        return task_result(f"Blocked path outside workspace: {raw_path}")

    if not target.exists():
        return task_result(f"File not found: {raw_path}")

    if not target.is_file():
        return task_result(f"Not a file: {raw_path}")

    content = target.read_text(encoding="utf-8", errors="replace")[:4000]
    return emit("file.content", {"path": raw_path, "content": content})
