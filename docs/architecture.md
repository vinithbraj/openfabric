# Agent Runtime Documentation

This page is the landing page for the current runtime documentation suite.

It is intentionally grounded in the code that exists today, not a future design.


## The Core Rule

At the highest level, the system follows one rule:

> The LLM decides meaning.  
> The runtime enforces structure and safety.  
> The gateway touches the real environment.


## What This Docs Suite Covers

This suite mostly explains the **typed agent runtime path** used by:

- Open WebUI
- `POST /v1/chat/completions`

The repository still contains older compatibility surfaces such as `/runs`,
`/sessions`, and a legacy echo-style CLI engine. Those are real and still
present, but they are not the main path this suite is trying to teach.


## Big Picture

```mermaid
flowchart TD
    UI[Open WebUI or API client] --> CHAT[/v1/chat/completions]
    CHAT --> RUNTIME[AgentRuntime]

    RUNTIME --> INPUT[Input Semantic Pipeline]
    INPUT --> DAG[Typed Action DAG]
    DAG --> SAFETY[Deterministic Safety Evaluation]
    SAFETY --> EXEC[Execution Engine]
    EXEC --> RESULTS[Result Bundle]
    RESULTS --> OUTPUT[Output Planning and Rendering]
    OUTPUT --> FINAL[Final Assistant Response]

    EXEC --> INTERNAL[Internal and local capabilities]
    EXEC --> GATEWAY[Gateway-backed capabilities]
    GATEWAY --> REMOTE[gateway_agent.remote_runner]
```

In plain language:

1. A user sends a natural-language request.
2. The runtime turns that request into typed planning decisions.
3. Those decisions become a validated DAG of actions.
4. Safety policy checks the DAG before execution.
5. Execution runs internal capabilities locally and environment-facing
   capabilities through the gateway.
6. Results are normalized into safe result shapes.
7. The runtime renders the final answer.


## Repository Map

The main pieces live here:

- API entrypoint: `src/aor_runtime/api/app.py`
- Runtime orchestrator: `src/agent_runtime/core/orchestrator.py`
- Core typed models: `src/agent_runtime/core/types.py`
- Semantic input stages: `src/agent_runtime/input_pipeline/`
- Capability registry and manifests: `src/agent_runtime/capabilities/`
- Execution engine and safety: `src/agent_runtime/execution/`
- Output planning and rendering: `src/agent_runtime/output_pipeline/`
- Observability and Open WebUI formatting: `src/agent_runtime/observability/`
- Gateway app and remote runner: `gateway_agent/`


## Documentation Guide

- [01_overview_for_humans.md](01_overview_for_humans.md)
  - A plain-language explanation of what the runtime is and why it exists.

- [02_request_lifecycle.md](02_request_lifecycle.md)
  - The full request flow from prompt intake to final rendering.

- [03_components_and_boundaries.md](03_components_and_boundaries.md)
  - The major subsystems, what each one owns, and where the boundaries are.

- [04_llm_stages_and_contracts.md](04_llm_stages_and_contracts.md)
  - Every stage where the LLM is used, plus the structured contracts and
    deterministic checks around it.

- [05_capabilities_safety_and_gateway.md](05_capabilities_safety_and_gateway.md)
  - How capabilities are modeled, selected, validated, gated, and executed.

- [06_worked_example_memory_report.md](06_worked_example_memory_report.md)
  - One full prompt walkthrough using a memory report plus file-save request.


## Good Starting Paths

If you are new to the system:

1. Start with [01_overview_for_humans.md](01_overview_for_humans.md)
2. Then read [06_worked_example_memory_report.md](06_worked_example_memory_report.md)
3. Then use [02_request_lifecycle.md](02_request_lifecycle.md) as the precise
   stage reference

If you are implementing or debugging:

1. Start with [02_request_lifecycle.md](02_request_lifecycle.md)
2. Then read [04_llm_stages_and_contracts.md](04_llm_stages_and_contracts.md)
3. Keep [05_capabilities_safety_and_gateway.md](05_capabilities_safety_and_gateway.md)
   nearby while changing capability behavior


## Current Truth Notes

- There is **no active `prompts/` directory** driving the runtime today.
  Prompt builders live in code, inside the stage modules that use them.
- The OpenAI-compatible chat path uses the typed runtime.
- Some compatibility endpoints and the legacy CLI still use a simpler engine.
- Output rendering can use LLM-selected display plans or a deterministic
  fallback.
- Confirmation is a first-class pause state, not just an execution error.
