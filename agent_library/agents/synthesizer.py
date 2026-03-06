from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse

app = FastAPI()


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event == "research.result":
        content = req.payload["content"]
        return {
            "emits": [
                {
                    "event": "answer.final",
                    "payload": {"answer": f"Research summary: {content}"},
                }
            ]
        }

    if req.event == "task.result":
        detail = req.payload["detail"]
        return {
            "emits": [
                {
                    "event": "answer.final",
                    "payload": {"answer": f"Task execution summary: {detail}"},
                }
            ]
        }

    if req.event == "file.content":
        path = req.payload["path"]
        content = req.payload["content"]
        return {
            "emits": [
                {
                    "event": "answer.final",
                    "payload": {"answer": f"File '{path}' content preview:\n{content}"},
                }
            ]
        }

    if req.event == "shell.result":
        command = req.payload["command"]
        stdout = req.payload["stdout"]
        stderr = req.payload["stderr"]
        returncode = req.payload["returncode"]
        answer = (
            f"Shell command '{command}' finished with code {returncode}. "
            f"stdout: {stdout or '<empty>'}; stderr: {stderr or '<empty>'}"
        )
        return {"emits": [{"event": "answer.final", "payload": {"answer": answer}}]}

    if req.event == "notify.result":
        detail = req.payload["detail"]
        return {
            "emits": [{"event": "answer.final", "payload": {"answer": f"Notification: {detail}"}}]
        }

    return {"emits": []}
