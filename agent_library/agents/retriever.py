from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse, with_node_envelope
from agent_library.template import agent_api, agent_descriptor, emit, noop

app = FastAPI()

AGENT_DESCRIPTOR = agent_descriptor(
    name="retriever",
    role="executor",
    description="Fetches research content for research queries.",
    capability_domains=["research", "knowledge_lookup"],
    action_verbs=["search", "lookup", "retrieve"],
    side_effect_policy="read_only",
    safety_enforced_by_agent=True,
    apis=[
        agent_api(
            name="lookup_research",
            event="research.query",
            summary="Fetches research content for a research query.",
            when="Fetches research content for a research query.",
            deterministic=False,
            side_effect_level="read_only",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                },
            },
        )
    ],
)
AGENT_METADATA = AGENT_DESCRIPTOR


@app.post("/handle", response_model=EventResponse)
@with_node_envelope("retriever", "executor")
def handle_event(req: EventRequest):
    if req.event == "research.query":
        query = req.payload["query"]
        return emit("research.result", {"content": f"Knowledge lookup for: {query}"})

    return noop()
