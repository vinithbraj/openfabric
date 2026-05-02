# OpenFABRIC Agent Runtime

OpenFABRIC is a schema-driven agent runtime for tool use, safe execution, and
deterministic rendering.

It is built around one simple rule:

> The LLM decides meaning.  
> The runtime enforces structure and safety.  
> The gateway touches the real environment.

This repository contains one core runtime with multiple entrypoints:

- the **typed agent runtime** used by Open WebUI and the
  OpenAI-compatible `POST /v1/chat/completions` path;
- compatibility surfaces such as `/runs`, `/sessions`, and the `aor` CLI, which
  are still present and now route into that same typed runtime.

If you want to understand how the current planning, capability selection,
confirmation, execution, and rendering flow works, start here:

- [docs/architecture.md](docs/architecture.md)


## What The Current Runtime Does

The typed agent runtime can:

- classify prompts into tool and workflow intent;
- decompose prompts into typed tasks;
- assign semantic verbs and known object types;
- shortlist and validate capabilities;
- build and execute a typed action DAG;
- route environment-facing work through the gateway;
- pause for confirmation before confirmation-gated actions;
- render a final answer from safe result previews and deterministic shapes.

Examples of current built-in capability families include:

- runtime introspection
- structured data transforms
- filesystem read/search/write
- shell inspection
- system inspection
- structured SQL query planning


## High-Level Flow

```mermaid
flowchart TD
    A[User Prompt] --> B[/v1/chat/completions]
    B --> C[Agent Runtime]
    C --> D[Input Semantic Pipeline]
    D --> E[Typed Action DAG]
    E --> F[Execution Engine]
    F --> G[Gateway or Internal Capability]
    G --> H[Result Bundle]
    H --> I[Output Planning and Rendering]
    I --> J[Final Assistant Response]
```


## Quick Start

```bash
./install.sh
./startup.sh
```

You can still use `aor serve`, but `./startup.sh` is the simplest local
launcher and exposes the full runtime env surface in one place.

By default, `startup.sh` binds the API to `0.0.0.0:8011` so local Dockerized
Open WebUI setups can reach it. Set `AOR_HOST=127.0.0.1` if you want the API
to listen on loopback only.

`startup.sh` now prefers `config_ucla.yaml` when it exists, then falls back to
`config.yaml`. You can override that explicitly:

```bash
./startup.sh --config config.yaml
./startup.sh --config config_ucla.yaml --port 8311
```

If you want one command that installs and starts the agent:

```bash
./install.sh --launch
```

OpenAI-compatible endpoints:

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/chat/completions`

Compatibility endpoints still present:

- `POST /runs`
- `POST /runs/stream`
- `POST /sessions`
- `GET /sessions`
- `GET /sessions/{id}`
- `GET /sessions/{id}/events`


## Open WebUI

The main modern runtime path is the OpenAI-compatible chat endpoint:

- connect Open WebUI to this server
- send prompts through `POST /v1/chat/completions`
- the runtime will emit structured observability trace sections during planning,
  safety, execution, and rendering

Confirmation-gated actions, such as `filesystem.write_file`, pause with a
confirmation message instead of pretending execution failed.


## Gateway Agent

Environment-facing capabilities use the gateway agent for bounded filesystem,
shell, and system operations.

```bash
python -m pip install -e .
./src/gateway_agent/startup.sh
```

See [src/gateway_agent/README.md](/home/vraj/Desktop/workspace/openfabric/src/gateway_agent/README.md) for gateway usage details.


## Documentation Map

- [docs/architecture.md](docs/architecture.md) — landing page and system map
- [docs/01_overview_for_humans.md](docs/01_overview_for_humans.md) — simple,
  plain-language explanation
- [docs/02_request_lifecycle.md](docs/02_request_lifecycle.md) — full request
  flow, stage by stage
- [docs/03_components_and_boundaries.md](docs/03_components_and_boundaries.md)
  — subsystem responsibilities and boundaries
- [docs/04_llm_stages_and_contracts.md](docs/04_llm_stages_and_contracts.md) —
  exact LLM-involved stages and structured contracts
- [docs/05_capabilities_safety_and_gateway.md](docs/05_capabilities_safety_and_gateway.md)
  — capabilities, safety, gateway routing, and file writing
- [docs/06_worked_example_memory_report.md](docs/06_worked_example_memory_report.md)
  — one end-to-end walkthrough of a real prompt
