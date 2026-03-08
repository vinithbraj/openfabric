from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse

app = FastAPI()

AGENT_METADATA = {
    "description": "Builds final user-facing answers from tool results.",
    "capability_domains": ["response_synthesis", "final_answer"],
    "action_verbs": ["summarize", "format", "respond"],
    "side_effect_policy": "read_only",
    "safety_enforced_by_agent": True,
    "methods": [
        {
            "name": "synthesize_file_result",
            "event": "file.content",
            "when": "Converts file content into final answer.",
        },
        {
            "name": "synthesize_shell_result",
            "event": "shell.result",
            "when": "Converts shell execution result into final answer.",
        },
        {
            "name": "synthesize_notify_result",
            "event": "notify.result",
            "when": "Converts notify result into final answer.",
        },
        {
            "name": "synthesize_task_result",
            "event": "task.result",
            "when": "Converts generic task result into final answer.",
        },
    ],
}


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
