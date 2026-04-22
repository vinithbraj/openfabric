# Agent Orchestration Runtime

Agent Orchestration Runtime is a local-first platform for defining, compiling, and running multi-agent execution graphs.

Core properties:

- LangGraph-backed graph execution
- YAML agent DSL
- FastAPI control plane
- Local vLLM-compatible LLM integration
- SQLite-backed run state and event logging
- Tool-aware agents with normalized I/O contracts

## Architectural Shape

The implementation follows a hybrid control-plane / execution-plane split:

- Control plane: YAML DSL, graph compiler, API, CLI, run persistence
- Execution plane: LangGraph runtime, router, agents, tools
- Infra layer: local OpenAI-compatible LLM endpoint, SQLite run store, shell/filesystem tools

The LLM is used for routing and agent decisions. The runtime remains deterministic for graph execution, persistence, node retries, and conditional transitions.

## Quick Start

1. Create and activate the venv:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e .
   ```

2. Configure the LLM endpoint:

   ```bash
   export AOR_LLM_BASE_URL=http://127.0.0.1:8000/v1
   export AOR_LLM_API_KEY=local
   export AOR_DEFAULT_MODEL=qwen-coder
   ```

   The runtime can auto-resolve against `/v1/models` if the requested model name is a shorthand or alias.

3. Start the API:

   ```bash
   uvicorn aor_runtime.api.app:create_app --factory --reload
   ```

4. Execute the example workflow:

   ```bash
   aor run examples/general_purpose_assistant.yaml --input '{"task":"Inspect the repository root"}'
   ```

## Project Layout

- `src/aor_runtime/` core package
- `examples/` example YAML graphs
- `prompts/` prompt templates used by agents
- `scripts/` convenience scripts

## Main Components

- `src/aor_runtime/dsl/` normalized DSL models and loader
- `src/aor_runtime/runtime/compiler.py` DSL -> compiled graph spec
- `src/aor_runtime/runtime/engine.py` LangGraph orchestration engine
- `src/aor_runtime/runtime/store.py` SQLite-backed runs, events, and snapshots
- `src/aor_runtime/agents/base.py` iterative agent execution model
- `src/aor_runtime/tools/` built-in local tools
- `src/aor_runtime/api/app.py` FastAPI control plane
