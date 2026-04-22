from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse, with_node_envelope
from agent_library.template import agent_api, agent_descriptor, emit_sequence, noop

app = FastAPI()

AGENT_DESCRIPTOR = agent_descriptor(
    name="planner",
    role="router",
    description="Plans whether a request should trigger research, task execution, or both.",
    capability_domains=["planning", "routing"],
    action_verbs=["plan", "route", "dispatch"],
    side_effect_policy="read_only",
    safety_enforced_by_agent=True,
    routing_notes=[
        "Use for simple research-plus-execution fan-out from a user request.",
        "Emits both research.query and task.plan for the same incoming user question.",
    ],
    apis=[
        agent_api(
            name="plan_research",
            event="research.query",
            summary="Routes the user question to the research path.",
            when="When the request should also trigger research.",
            deterministic=True,
            side_effect_level="read_only",
        ),
        agent_api(
            name="plan_task",
            event="task.plan",
            summary="Routes the user question to the task execution path.",
            when="When the request should also trigger task execution.",
            deterministic=True,
            side_effect_level="read_only",
        ),
    ],
)
AGENT_METADATA = AGENT_DESCRIPTOR


@app.post("/handle", response_model=EventResponse)
@with_node_envelope("planner", "router")
def handle_event(req: EventRequest):
    if req.event == "user.ask":
        question = req.payload["question"]
        return emit_sequence(
            [
                {"event": "research.query", "payload": {"query": question}},
                {"event": "task.plan", "payload": {"task": question}},
            ]
        )

    return noop()
