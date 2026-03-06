from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse

app = FastAPI()


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event == "user.ask":
        question = req.payload["question"]
        return {
            "emits": [
                {"event": "research.query", "payload": {"query": question}},
                {"event": "task.plan", "payload": {"task": question}},
            ]
        }

    return {"emits": []}

