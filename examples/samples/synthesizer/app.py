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

    if req.event == "research.result":
        content = req.payload["content"]

        return {
            "emits": [
                {
                    "event": "answer.final",
                    "payload": {
                        "answer": f"Synthesized final answer from: {content}"
                    }
                }
            ]
        }

    return {"emits": []}
