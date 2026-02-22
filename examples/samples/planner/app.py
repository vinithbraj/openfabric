from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()


class EventRequest(BaseModel):
    event: str
    payload: Dict


class EmittedEvent(BaseModel):
    event: str
    payload: Dict


class EventResponse(BaseModel):
    emits: List[EmittedEvent]


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):

    if req.event == "user.ask":
        question = req.payload["question"]

        return {
            "emits": [
                {"event": "research.query", "payload": {"query": f"{question} overview"}},
                {"event": "research.query", "payload": {"query": f"{question} history"}},
                {"event": "research.query", "payload": {"query": f"{question} applications"}}
            ]
        }

    return {"emits": []}
