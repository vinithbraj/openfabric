import re

from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse

app = FastAPI()


def _extract_filepath(question: str):
    match = re.search(r"(?:read|open)\s+([./a-zA-Z0-9_-]+\.[a-zA-Z0-9]+)", question)
    if match:
        return match.group(1)
    return None


def _extract_command(question: str):
    match = re.search(r"(?:run|execute)\s+`([^`]+)`", question)
    if match:
        return match.group(1)
    return None


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event != "user.ask":
        return {"emits": []}

    question = req.payload["question"]
    question_lc = question.lower()
    emits = []

    filepath = _extract_filepath(question)
    if filepath:
        emits.append({"event": "file.read", "payload": {"path": filepath}})

    command = _extract_command(question)
    if command:
        emits.append({"event": "shell.exec", "payload": {"command": command}})

    if "notify" in question_lc or "alert" in question_lc:
        emits.append(
            {
                "event": "notify.send",
                "payload": {
                    "channel": "console",
                    "message": f"Notification requested: {question}",
                },
            }
        )

    if not emits:
        emits.append(
            {
                "event": "task.result",
                "payload": {
                    "detail": "No operations detected. Use phrases like "
                    "'read <file>', 'run `<command>`', or 'notify ...'."
                },
            }
        )

    return {"emits": emits}

