import re

from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse

app = FastAPI()


def _try_sum_expression(task: str):
    numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", task)]
    if len(numbers) < 2:
        return None
    if "sum" not in task.lower() and "+" not in task:
        return None
    value = sum(numbers)
    return int(value) if value.is_integer() else value


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event == "task.plan":
        task = req.payload["task"]
        total = _try_sum_expression(task)

        if total is not None:
            detail = f"Computed sum from task: {total}"
        else:
            detail = "No executable arithmetic tool action detected"

        return {"emits": [{"event": "task.result", "payload": {"detail": detail}}]}

    return {"emits": []}

