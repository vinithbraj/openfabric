# Agent Orchestration Runtime

Agent Orchestration Runtime is a local-first platform for defining, compiling, and running deterministic agent workflows.

Core properties:

- LangGraph-backed graph execution
- YAML agent DSL
- FastAPI control plane
- Local vLLM-compatible LLM integration
- SQLite-backed run state and event logging
- Deterministic planner-executor-validator runtime

## Architectural Shape

The implementation follows a control-plane / execution-plane split:

- Control plane: YAML DSL, graph compiler, API, CLI, run persistence
- Execution plane: LangGraph runtime, planner, deterministic executor, deterministic validator
- Infra layer: local OpenAI-compatible LLM endpoint, SQLite run store, filesystem/shell/python tools

The LLM is used only to create the execution plan and optional retry plans. Execution, validation, logging, and final output are deterministic.

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

5. Start an interactive session:

   ```bash
   aor chat examples/general_purpose_assistant.yaml
   ```

## Gateway Agent

For gateway-backed shell execution, a separate node-local agent scaffold lives in `gateway_agent/`.

Use matching config on both sides:

```bash
export GATEWAY_NODE_NAME=localhost
```

The example runtime spec already includes:

```yaml
nodes:
  default: localhost
  endpoints:
    - name: localhost
      url: http://127.0.0.1:8787/exec
```

So once the agent is running, `examples/general_purpose_assistant.yaml` can use shell commands without extra `AOR_GATEWAY_URL` exports. See `gateway_agent/README.md` for install and run instructions.

## Project Layout

- `src/aor_runtime/` core package
- `examples/` example runtime specs
- `prompts/` prompt templates used by the planner
- `scripts/` convenience scripts

## Main Components

- `src/aor_runtime/dsl/` normalized DSL models and loader
- `src/aor_runtime/runtime/compiler.py` DSL -> compiled graph spec
- `src/aor_runtime/runtime/engine.py` LangGraph planner -> executor -> validator runtime
- `src/aor_runtime/runtime/planner.py` single-call LLM planner
- `src/aor_runtime/runtime/executor.py` deterministic tool executor
- `src/aor_runtime/runtime/validator.py` deterministic validation layer
- `src/aor_runtime/runtime/store.py` SQLite-backed runs, events, and snapshots
- `src/aor_runtime/tools/` built-in local tools
- `src/aor_runtime/api/app.py` FastAPI control plane
