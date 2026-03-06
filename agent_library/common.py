from typing import Dict, List

from pydantic import BaseModel


class EventRequest(BaseModel):
    event: str
    payload: Dict


class EmittedEvent(BaseModel):
    event: str
    payload: Dict


class EventResponse(BaseModel):
    emits: List[EmittedEvent]

