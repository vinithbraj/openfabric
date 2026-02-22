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

    if req.event == "research.query":
        query = req.payload["query"]

        return {
            "emits": [
                {
                    "event": "research.result",
                    "payload": {
                        "content": f"Retrieved data for '{query}'"
                    }
                }
            ]
        }

    return {"emits": []}
