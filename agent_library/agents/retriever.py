from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse

app = FastAPI()

AGENT_METADATA = {
    "description": "Fetches research content for research queries.",
    "methods": [
        {"name": "lookup_research", "event": "research.query"},
    ],
}


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event == "research.query":
        query = req.payload["query"]
        return {
            "emits": [
                {
                    "event": "research.result",
                    "payload": {"content": f"Knowledge lookup for: {query}"},
                }
            ]
        }

    return {"emits": []}
