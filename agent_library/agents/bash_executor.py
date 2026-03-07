import re
import shlex
import subprocess

from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse

app = FastAPI()

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


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event != "shell.exec":
        return {"emits": []}

    command = req.payload["command"]
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
