# Agent Contract V1

This contract defines the normalized shape every new OpenFabric agent should follow.

## Goals

- one shared descriptor format for every agent
- one shared execution envelope for every agent
- one shared result envelope for every agent
- enough self-description that the planner can discover APIs without hardcoded agent rules

## Descriptor

New agents should export:

- `AGENT_DESCRIPTOR`
- optional `AGENT_METADATA = AGENT_DESCRIPTOR` for backwards compatibility

Use [agent_library/template.py](/home/vinith/Desktop/Workspace/openfabric/agent_library/template.py) helpers:

```python
from agent_library.template import agent_api, agent_descriptor

AGENT_DESCRIPTOR = agent_descriptor(
    name="example_agent",
    role="executor",
    description="Short description of the agent.",
    capability_domains=["example_domain"],
    action_verbs=["read", "execute"],
    side_effect_policy="read_only",
    safety_enforced_by_agent=True,
    apis=[
        agent_api(
            name="do_example_work",
            trigger_event="task.plan",
            emits=["task.result"],
            summary="Handles example task execution.",
            input_schema={
                "type": "object",
                "properties": {
                    "input_value": {"type": "string"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                },
            },
            deterministic=True,
            side_effect_level="read_only",
        )
    ],
    planning_hints={
        "keywords": ["example", "sample"],
        "preferred_task_shapes": ["lookup"],
        "instruction_operations": ["do_example_work"],
    },
)

AGENT_METADATA = AGENT_DESCRIPTOR
```

Use `trigger_event` to describe what input event activates the API.

Use `emits` to describe what output events the API may publish.

`event` still works as a legacy alias for `trigger_event`, but new agents should use the explicit form.

## Planning Hints

New agents should expose `planning_hints` in the descriptor, and optionally per API.

These hints are how the planner becomes less hardcoded over time.

Recommended descriptor-level fields:

- `keywords`
- `anti_keywords`
- `preferred_task_shapes`
- `instruction_operations`
- `structured_followup`
- `native_count_preferred`
- `routing_priority`

Recommended API-level fields:

- `instruction_operations`
- `keywords`
- `preferred_task_shapes`

These hints are advisory, not executable. They tell the planner when the agent is a good fit and which instruction operations it should prefer when building `task.plan` steps.

## Shared Request Envelope

Descriptor-level `request_schema` and `result_schema` are injected automatically.

The shared request contract includes:

- `node`
- `task`
- `original_task`
- `instruction.api`
- `instruction.input`
- `context`
- `policy`
- `artifacts`

## Shared Result Envelope

The shared result contract includes:

- `node`
- `status`
- `api`
- `detail`
- `raw_output`
- `structured_output`
- `artifacts`
- `reduction_request`
- `error`
- `metrics`

Existing agents may still emit legacy event-specific payloads while migrating, but new agents should converge on this structure inside their payloads.

## Template Helpers

[agent_library/template.py](/home/vinith/Desktop/Workspace/openfabric/agent_library/template.py) provides:

- `agent_api(...)`
- `agent_descriptor(...)`
- `emit(...)`
- `emit_many(...)`
- `noop()`
- `task_result(...)`
- `failure_result(...)`
- `needs_decomposition(...)`
- `final_answer(...)`

## Compatibility

- runtime continues to support legacy `AGENT_METADATA.methods`
- engine normalizes both legacy metadata and `AGENT_DESCRIPTOR`
- `system.capabilities` now publishes both legacy `methods` and normalized `apis`

## Definition Of Done For A New Agent

- exports a valid `AGENT_DESCRIPTOR`
- descriptor contains at least one API in `apis`
- `/handle` still returns `{"emits": [...]}` for runtime compatibility
- emitted payloads include the common `node` envelope through `with_node_envelope`
