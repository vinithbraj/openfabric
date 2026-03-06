import shlex
import subprocess

from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse

app = FastAPI()

ALLOWED_COMMANDS = {"echo", "pwd", "date", "ls"}


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event != "shell.exec":
        return {"emits": []}

    command = req.payload["command"]
    try:
        args = shlex.split(command)
    except ValueError as exc:
        return {
            "emits": [
                {"event": "task.result", "payload": {"detail": f"Invalid command: {exc}"}}
            ]
        }

    if not args:
        return {
            "emits": [
                {"event": "task.result", "payload": {"detail": "Empty command rejected"}}
            ]
        }

    if args[0] not in ALLOWED_COMMANDS:
        return {
            "emits": [
                {
                    "event": "task.result",
                    "payload": {
                        "detail": f"Command '{args[0]}' not allowed. "
                        f"Allowed: {sorted(ALLOWED_COMMANDS)}"
                    },
                }
            ]
        }

    completed = subprocess.run(args, capture_output=True, text=True, timeout=5)
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

