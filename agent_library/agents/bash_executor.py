import re
import shlex
import subprocess

from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse

app = FastAPI()

AGENT_METADATA = {
    "description": "Executes bash commands and returns stdout/stderr/exit status.",
    "routing_notes": [
        "Use for command-line operations such as search, list, grep, and process inspection.",
        "Preferred for file discovery intents that mention find/search/list files.",
    ],
    "methods": [
        {
            "name": "execute_bash_command",
            "event": "task.plan",
            "when": "Runs bash command strings derived from task plans and emits shell execution result.",
            "intent_tags": ["cli_exec", "file_search", "workspace_inspection"],
            "examples": [
                "run `find . -iname \"*vinith*\"`",
                "execute `ls -la agent_library/agents`",
            ],
            "anti_patterns": ["read a specific file's contents"],
        }
    ],
}

BLOCKED_PATTERNS = [
    r"\brm\s+-rf\s+/\b",
    r"\bmkfs\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bdd\s+if=",
]


def _is_blocked(command: str) -> bool:
    command_lc = command.lower()
    return any(re.search(pattern, command_lc) for pattern in BLOCKED_PATTERNS)


def _extract_command_from_task(task: str):
    explicit = re.search(r"(?:run|execute)\s+`([^`]+)`", task, re.IGNORECASE)
    if explicit:
        return explicit.group(1)

    plain = re.search(r"(?:run|execute)\s+(.+)$", task, re.IGNORECASE)
    if plain:
        return plain.group(1).strip()

    task_lc = task.lower()
    if ("find" in task_lc or "search" in task_lc or "locate" in task_lc) and "file" in task_lc:
        name_match = re.search(r"(?:named|called)\s+['\"]?([a-zA-Z0-9._-]+)['\"]?", task, re.IGNORECASE)
        if name_match:
            token = name_match.group(1)
            return f"find . -iname '*{token}*'"
        return "find . -type f"

    if "list files" in task_lc or "show files" in task_lc:
        return "find . -type f"

    return None


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event == "shell.exec":
        command = req.payload["command"]
    elif req.event == "task.plan":
        task = req.payload.get("task", "")
        if not isinstance(task, str):
            return {"emits": []}
        command = _extract_command_from_task(task)
        if not command:
            return {"emits": []}
    else:
        return {"emits": []}
    if _is_blocked(command):
        return {
            "emits": [
                {
                    "event": "task.result",
                    "payload": {"detail": "Rejected potentially destructive command"},
                }
            ]
        }

    try:
        shlex.split(command)
    except ValueError as exc:
        return {
            "emits": [
                {"event": "task.result", "payload": {"detail": f"Invalid command: {exc}"}}
            ]
        }

    if not command.strip():
        return {"emits": [{"event": "task.result", "payload": {"detail": "Empty command rejected"}}]}

    try:
        completed = subprocess.run(
            ["/bin/bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return {
            "emits": [
                {
                    "event": "task.result",
                    "payload": {"detail": "Command timed out after 10 seconds"},
                }
            ]
        }

    return {
        "emits": [
            {
                "event": "shell.result",
                "payload": {
                    "command": command,
                    "stdout": completed.stdout.strip(),
                    "stderr": completed.stderr.strip(),
                    "returncode": completed.returncode,
                },
            }
        ]
    }
