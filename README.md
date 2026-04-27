# Agent Orchestration Runtime

Agent Orchestration Runtime (AOR) is a deterministic-first local agent runtime for turning natural-language tasks into validated execution plans. The current implementation routes supported prompts through a capability registry, compiles typed intents into explicit plans, executes tools deterministically, validates results, and shapes final output through `runtime.return` and `OutputContract`.

Major runtime properties:

- deterministic capability planning through `TaskPlanner` and `CapabilityRegistry`
- shared typed-intent infrastructure for filesystem, SQL, shell, fetch, and compound workflows
- native content search via `fs.search_content`
- gateway-routed shell and SLURM execution
- SQLite-backed sessions, events, and snapshots
- capability eval gates plus a global exhaustive regression suite
- optional SLURM-only typed LLM-intent fallback, disabled by default

## Architecture Snapshot

At runtime, the CLI or API hands a request to `ExecutionEngine`. The engine invokes `TaskPlanner`, which first asks `CapabilityRegistry` whether any capability pack can classify the prompt into a typed intent. If a pack matches, that pack compiles the intent into an explicit `ExecutionPlan`; only when capability matching misses does the runtime fall back to the raw planner path. The executor resolves step-to-step dataflow, runs tools, and streams progress/events when requested. The validator re-checks deterministic expectations against tool outputs. Final user-facing output is shaped through `runtime.return`, which applies an `OutputContract` to normalize and render results consistently.

## Quick Start

1. Create and activate the venv:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e .
   ```

   This install includes the PostgreSQL driver used by `sql.query` for
   `postgresql+psycopg://...` database URLs.

2. Create the local app config:

   ```bash
   edit config.yaml
   ```

   `config.yaml` is the tracked app config. Edit it with your LLM and optional SQL settings. The runtime auto-loads `config.yaml` from the current working directory, or you can pass `--config /path/to/config.yaml`.

3. Start the API:

   ```bash
   aor serve
   ```

4. Execute the example workflow:

   ```bash
   aor run examples/general_purpose_assistant.yaml --input '{"task":"Inspect the repository root"}'
   ```

5. Start an interactive session:

   ```bash
   aor chat examples/general_purpose_assistant.yaml
   ```

6. See live progress while running:

   ```bash
   aor chat examples/general_purpose_assistant.yaml --progress
   ```

## App Config

Application-level settings now live in `config.yaml`, not `.env.example`.

- `config.yaml` is the tracked app config
- use `--config` on CLI commands if you want to point at a different file

The runtime spec continues to own agent behavior, tools, and node routing. The app config owns process settings such as LLM, SQL, retry, and server defaults.

By default, planner and decomposer model selection also comes from `config.yaml -> llm.default_model`. Set `planner.model` in a runtime spec only when you intentionally want that spec to override the app default.

Important runtime flags and envs include:

- `AOR_ENABLE_LLM_INTENT_EXTRACTION`
  - enables the optional typed LLM intent extractor used for fuzzy SLURM prompts
- `AOR_SLURM_FIXTURE_DIR`
  - fixture-backed SLURM mode for tests and evals
- `AOR_LLM_INTENT_FIXTURE_PATH`
  - fixture-backed typed intent responses for evals

## Gateway Agent

For gateway-backed shell and SLURM execution, a separate node-local agent scaffold lives in `gateway_agent/`.

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

## Major Features

- Deterministic capability planning:
  - supported prompts route through capability packs before raw planner fallback
- Native filesystem and content-search tooling:
  - including `fs.search_content`
- Output contracts:
  - explicit shaping through `runtime.return` and `OutputContract`
- Evaluation gates:
  - exhaustive NLP regression plus capability-pack evals
- Read-only SLURM support:
  - queue, accounting, nodes, partitions, metrics, and slurmdbd health
- Optional typed LLM-intent fallback:
  - implemented for fuzzy SLURM inspection prompts only, and disabled by default

## Usage Entrypoints

- CLI:
  - `aor run ...`
  - `aor chat ...`
  - `aor serve`
- API:
  - `POST /runs`
  - `POST /runs/stream`
  - `POST /sessions`
  - `GET /sessions/{id}/events`
- Gateway:
  - `/exec`
  - `/exec/stream`
- Example runtime spec:
  - `examples/general_purpose_assistant.yaml`

## Safety Principles

- Deterministic-first:
  - capability packs are tried before raw planner fallback
- No raw LLM tool plans where avoidable:
  - supported behavior should compile from typed intents
- No arbitrary shell planning for domain capabilities:
  - domain packs such as SLURM should use native tools, not raw shell prompts
- Final output is explicit:
  - `runtime.return` and `OutputContract` define the user-visible shape

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Capability Packs](docs/CAPABILITY_PACKS.md)
- [Intent and Compiler Model](docs/INTENT_AND_COMPILER_MODEL.md)
- [Tools and Runtime](docs/TOOLS_AND_RUNTIME.md)
- [Evaluation and Regression](docs/EVALUATION_AND_REGRESSION.md)
- [Adding a Capability](docs/ADDING_A_CAPABILITY.md)
- [SLURM Capability Design](docs/SLURM_CAPABILITY_DESIGN.md)
- [LLM Intent Extraction Design](docs/LLM_INTENT_EXTRACTION_DESIGN.md)
- [Debugging Guide](docs/DEBUGGING_GUIDE.md)
- [Mermaid Diagrams](docs/MERMAID_DIAGRAMS.md)

## Project Layout

- `src/aor_runtime/` core package
- `docs/` technical documentation
- `examples/` example runtime specs
- `prompts/` prompt templates used by the planner
- `scripts/` convenience scripts

## Main Components

- `src/aor_runtime/dsl/` normalized DSL models and loader
- `src/aor_runtime/runtime/compiler.py` DSL -> compiled runtime spec
- `src/aor_runtime/runtime/engine.py` execution orchestration, persistence, and events
- `src/aor_runtime/runtime/planner.py` deterministic-first planner with fallback modes
- `src/aor_runtime/runtime/capabilities/` capability-pack routing and pack implementations
- `src/aor_runtime/runtime/executor.py` deterministic tool executor
- `src/aor_runtime/runtime/validator.py` deterministic validation layer
- `src/aor_runtime/runtime/output_contract.py` normalization and final output shaping rules
- `src/aor_runtime/runtime/store.py` SQLite-backed runs, events, and snapshots
- `src/aor_runtime/tools/` built-in local tools
- `src/aor_runtime/api/app.py` FastAPI control plane
