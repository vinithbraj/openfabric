# OpenFabric

OpenFabric is a framework for defining and running agentic systems using a declarative configuration model.

If Docker Compose describes how containerized services are wired together, OpenFabric does something similar for reasoning and execution systems: planners, tool runners, synthesizers, databases, LLM-backed agents, and other cooperating nodes.

## What It Is

OpenFabric gives you:

- A declarative language for describing an agentic system
- A specification model for nodes, contracts, events, and runtime bindings
- A runtime engine that orchestrates execution across a multi-node graph
- A reusable agent library for common reasoning, shell, SQL, Slurm, and synthesis tasks
- An OpenAI-compatible gateway so the system can be used from Open WebUI

OpenFabric is not the intelligence itself. The intelligence lives inside the agents: language models, databases, retrieval services, shell tools, schedulers, or any other computational component. OpenFabric is the control plane that composes those components into a coherent working system.

## Core Concept

An OpenFabric system is a directed graph of cooperating nodes:

- **Nodes** represent units of reasoning or execution
- **Edges** define how data moves between nodes
- **Contracts** define the expected data shape for events
- **Rules** constrain what can happen and when
- **Runtime bindings** specify how the engine reaches each node

This allows complex workflows to be described declaratively instead of hard-coding orchestration into one large application.

## Architecture

OpenFabric is easiest to think about in three layers.

### 1. Behavior Layer

This is where the actual work happens.

Examples:

- planners
- synthesizers
- shell runners
- SQL runners
- Slurm runners
- retrievers
- databases
- LLM-backed services

These components interpret instructions, execute actions, and produce results.

### 2. Orchestration Layer

This is the runtime engine in [`runtime/engine.py`](runtime/engine.py).

It is responsible for:

- parsing the spec
- instantiating runtime bindings
- routing events to subscribed agents
- enforcing contracts
- managing workflow execution
- handling validation, replanning, and recovery paths

This is the traffic controller of the system.

### 3. Blueprint Layer

This is the declarative system definition.

A spec describes:

- the nodes in the graph
- the events they emit and subscribe to
- the contracts those events must satisfy
- the runtime adapter and endpoint for each node
- autostart behavior for local services

In this repo, example specs live under [`examples/`](examples/) and agentic system specs live under [`agent_library/specs/`](agent_library/specs/).

## How It Works

1. Define a system in YAML.
2. Describe the participating nodes, events, contracts, and bindings.
3. Run the spec with the OpenFabric runtime.
4. The runtime instantiates or connects to the agents.
5. Events flow through the graph according to the declared topology.
6. The system produces a final answer, artifact, or downstream event.

This is conceptually similar to how Docker Compose:

- defines services in YAML
- wires networking and dependencies
- starts and coordinates the system

The difference is that OpenFabric is built around reasoning and execution graphs rather than container lifecycles.

## What This Repository Contains

This repository includes both the framework pieces and a concrete operational assistant stack.

### Runtime

- [`cli.py`](cli.py): CLI entrypoint for running specs
- [`runtime/engine.py`](runtime/engine.py): orchestration engine
- [`runtime/semantic_validator.py`](runtime/semantic_validator.py): spec/runtime validation helpers

### Reusable Agents

The main reusable agents live in [`agent_library/agents/`](agent_library/agents/).

Examples include:

- `planner.py`
- `retriever.py`
- `synthesizer.py`
- `operations_planner.py`
- `llm_operations_planner.py`
- `shell_runner.py`
- `sql_runner.py`
- `slurm_runner.py`
- `filesystem.py`
- `validator.py`

These agents follow a simple HTTP event contract style using a `/handle` endpoint with an input event and emitted output events.

### Agent Specs

Useful specs in this repo:

- [`examples/hello.yml`](examples/hello.yml): minimal example
- [`examples/hello_http.yml`](examples/hello_http.yml): HTTP-wired example
- [`agent_library/specs/research_task_assistant.yml`](agent_library/specs/research_task_assistant.yml): research-style assistant
- [`agent_library/specs/ops_assistant.yml`](agent_library/specs/ops_assistant.yml): rule-based operational assistant
- [`agent_library/specs/ops_assistant_llm.yml`](agent_library/specs/ops_assistant_llm.yml): LLM-guided multi-agent operations stack

### Open WebUI Gateway

The repo also exposes the planner stack through an OpenAI-compatible gateway:

- [`openwebui_gateway.py`](openwebui_gateway.py)
- [`OPENWEBUI.md`](OPENWEBUI.md)

This lets Open WebUI act as the chat frontend while OpenFabric handles planning, execution, validation, and synthesis behind the scenes.

## Key Characteristics

- **Declarative**: define what the system is, not every imperative orchestration detail
- **Composable**: build larger systems from smaller reusable nodes
- **Distributed**: nodes can run across separate services or hosts
- **Agentic-first**: designed around reasoning workflows and tool coordination
- **Event-driven**: communication happens through explicit event contracts
- **Extensible**: add new agent types without rewriting the runtime

## Quickstart

### Prerequisites

- Python 3
- `pip`
- optional: a local OpenAI-compatible LLM endpoint if using the LLM-driven planner

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the minimal example

```bash
python cli.py examples/hello.yml
```

### 3. Run the HTTP example

The runtime auto-starts the planner and synthesizer services described in the spec:

```bash
python cli.py examples/hello_http.yml
```

### 4. Run the operational assistant

```bash
python cli.py agent_library/specs/ops_assistant_llm.yml
```

Useful variants:

```bash
python cli.py agent_library/specs/ops_assistant_llm.yml --list-agents
python cli.py agent_library/specs/ops_assistant_llm.yml --agent shell_runner
python cli.py agent_library/specs/ops_assistant_llm.yml --agent sql_runner --agent synthesizer
```

## Persisted Runs And Graph Inspection

Workflow runs are checkpointed under `artifacts/runtime_runs` by default. You can override that location with `OPENFABRIC_RUN_STORE_DIR`.

Useful inspection commands:

```bash
python cli.py --list-runs
python cli.py --list-runs --run-status completed
python cli.py --show-run 20260421T010203Z_abcd1234ef56
python cli.py --show-run-graph 20260421T010203Z_abcd1234ef56
python cli.py --show-run-graph 20260421T010203Z_abcd1234ef56 --graph-format json
python cli.py --serve-runs-ui
python cli.py --serve-runs-ui --ui-port 8787
```

Each persisted run directory now includes:

- `state.json`: the latest durable runtime state
- `timeline.jsonl`: checkpoint-by-checkpoint stage history
- `summary.json`: compact inspection summary
- `graph.json`: persisted workflow execution graph
- `graph.mmd`: Mermaid flowchart rendering of the workflow graph

The browser UI serves a run list, summary cards, timeline checkpoints, Mermaid text, and an interactive SVG graph renderer over the persisted workflow graph.

## LLM Configuration

The LLM-driven planner stack uses shared environment variables across agents.

Common settings:

```bash
export LLM_OPS_API_KEY=dummy
export LLM_OPS_BASE_URL=http://127.0.0.1:8000/v1
export LLM_OPS_MODEL=your-local-model
export LLM_OPS_TIMEOUT_SECONDS=300
```

Additional optional override:

```bash
export LLM_OPS_SYNTH_MODEL=your-synthesis-model
```

If the LLM call fails or configuration is unavailable, parts of the system can fall back to deterministic planning or formatting paths where appropriate.

## Open WebUI Integration

If you already have a local OpenAI-compatible model server running, you can start the gateway with:

```bash
bash scripts/start_openwebui_gateway.sh
```

Defaults:

- Gateway host: `0.0.0.0`
- Gateway port: `8310`
- LLM base URL: `http://127.0.0.1:8000/v1`

The Open WebUI connection typically looks like:

```text
URL: http://127.0.0.1:8310/v1
API Key: dummy
Model: openfabric-planner
```

See [`OPENWEBUI.md`](OPENWEBUI.md) for setup details, context handling, SQL configuration, and gateway behavior.

## SQL Support

The operational assistant includes read-only SQL agents.

These agents can:

- inspect schemas
- list tables and columns
- generate read-only SQL from user requests
- execute queries
- return structured rows for downstream synthesis

Typical environment setup:

```bash
export SQL_DATABASE_URL=postgresql://user:password@host:5432/database
export SQL_AGENT_ROW_LIMIT=100
```

Multiple SQL databases can be declared in the spec, each with its own DSN and metadata. The planner can route database requests to the correct SQL agent based on database name and routing hints.

## Safety Model

OpenFabric is designed so orchestration and execution policy can be enforced centrally rather than left entirely to free-form prompting.

Examples in this repo include:

- capability discovery via `system.capabilities`
- validator-based workflow checking
- read-only SQL enforcement
- shell safety policies
- recovery and replanning paths
- runtime execution policy propagation

The current operational assistant is optimized for non-destructive workflows, grounded results, and deterministic fallbacks where a free-form LLM answer would be risky.

## When To Use OpenFabric

OpenFabric is a good fit when you want:

- multi-agent AI systems
- tool-augmented LLM workflows
- database-backed assistants
- research and experimentation with agent orchestration
- structured operational assistants
- distributed reasoning pipelines

It is especially useful when one model or tool should not own the entire workflow, and you want explicit reasoning and execution nodes instead.

## Repository Guide

- [`README.md`](README.md): top-level overview
- [`OPENWEBUI.md`](OPENWEBUI.md): Open WebUI gateway usage
- [`agent_library/README.md`](agent_library/README.md): reusable agent library overview
- [`SESSION_CHECKPOINT.md`](SESSION_CHECKPOINT.md): recent architecture notes and checkpoints

## Philosophy

OpenFabric enables a shift from:

- single-agent scripts to multi-agent systems
- linear pipelines to graph-based execution
- ad hoc orchestration to declarative composition
- opaque behavior to explicit event-driven coordination

It provides a structured way to design, operate, and evolve systems where reasoning and execution are distributed across cooperating nodes.

## License

See [`LICENSE`](LICENSE).
