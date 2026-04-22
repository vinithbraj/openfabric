# Graph Architecture

This document defines the target graph-oriented execution model for the Version 4 runtime.

## Goal

Turn the assistant into an explicit graph:

- each agent is a node with a clear role
- nodes are connected through typed edges
- data flows through the graph as a common envelope
- execution, reduction, validation, and retry are all visible in the same structure

The runtime already behaves like an implicit graph. The current implementation now makes execution, reduction, validation, routing, replanning, and clarification machine-readable inside the workflow graph.

## Target Node Roles

- `router`: chooses the next node or subgraph
- `executor`: runs a concrete capability such as SQL, Slurm, shell, filesystem, or notification
- `reducer`: transforms raw outputs into a smaller structured result
- `validator`: checks whether a node or whole workflow satisfied the requested intent
- `synthesizer`: formats the final answer for the user

## Common Node Template

Every agent should converge on the same logical template:

1. Receive an instruction envelope.
2. Inspect its declared capabilities and choose a concrete API.
3. Execute through a command, gateway API, SQL statement, or code snippet.
4. Emit raw execution data.
5. Send raw data to a reducer node.
6. Send reduced data to a validator node.
7. Route the validated result to the next node or back to the router.

The current codebase already has most of these behaviors, but some are still embedded inside executor agents instead of exposed as first-class graph nodes.

## Common Data Schema

Phase 1 introduces a shared graph schema in [runtime/graph.py](/home/vinith/Desktop/Workspace/openfabric/runtime/graph.py).

The important structures are:

- capability graph:
  - static topology of agents and event nodes
  - shows which events each agent consumes and emits
- workflow execution graph:
  - one root workflow node
  - one node per attempt
  - one node per execution step
  - optional reducer nodes
  - validator nodes for both steps and attempts
  - router nodes for both step-level and workflow-level decisions
  - replan nodes when the runtime asks the planner to replace a step or derive a fallback workflow
  - clarification nodes when the run needs user input to continue

Each execution step should eventually expose:

- `task`
- `target_agent`
- `instruction`
- `depends_on`
- `when`
- `event`
- `raw_output`
- `result`
- `validation`
- `duration_ms`

## Phase Plan

### Phase 1

Implemented in this change:

- shared runtime graph schema
- agent capability graph projection
- workflow result graph projection
- reducer and validator nodes modeled in the graph when data is available

### Phase 2

Implemented in this change:

- introduce a first-class `data_reducer` agent and `data.reduce` -> `data.reduced` event path
- route completed step outputs through the reducer before validation when a local reduction command or reduced result is available
- preserve backward compatibility by allowing the reducer to fall back to executor-provided reduced output when replay is unavailable or fails

### Phase 3

Implemented in this change:

- add a shared reduction contract in [agent_library/reduction.py](/home/vinith/Desktop/Workspace/openfabric/agent_library/reduction.py)
- make executor agents emit raw output plus `reduction_request` metadata instead of planning local reducer commands inline
- let `data_reducer` own reduction planning and execution for shell, SQL, and Slurm reduction paths
- surface planned reducer requests in the workflow graph

### Phase 4

Implemented in this change:

- make workflow-level and step-level router decisions explicit graph records
- represent retry, fallback, replan, and clarification transitions as graph nodes instead of hidden engine branches
- preserve backward-compatible `workflow.result` payloads while adding `routing`, `replan`, and `clarification` execution records
- project router, replan, and clarification nodes into [runtime/graph.py](/home/vinith/Desktop/Workspace/openfabric/runtime/graph.py)

### Phase 5

Implemented in this change:

- add durable run-state persistence in [runtime/run_store.py](/home/vinith/Desktop/Workspace/openfabric/runtime/run_store.py)
- checkpoint in-flight workflow execution so runs can resume after crashes
- persist terminal workflow payloads and expose `run_id` plus persistence metadata in `workflow.result` and `clarification.required`
- add `Engine.replay_run(run_id)` and `Engine.resume_run(run_id)` for replay and recovery flows
- attach `run_id` to the workflow graph so stored runs can be correlated with visualized executions

### Phase 6

Implemented in this change:

- add a shared `node` envelope helper in [agent_library/common.py](/home/vinith/Desktop/Workspace/openfabric/agent_library/common.py)
- standardize core execution-path node metadata across executor, reducer, and validator agents
- attach `run_id`, `attempt`, `step_id`, `task`, `target_agent`, `operation`, and role metadata to runtime-sent `task.plan`, `validation.request`, and `data.reduce` payloads
- preserve node metadata inside compacted step payloads so workflow records retain the agent-level execution envelope

### Phase 7

Implemented in this change:

- add persisted run-inspection helpers in [runtime/run_inspector.py](/home/vinith/Desktop/Workspace/openfabric/runtime/run_inspector.py)
- derive and store `summary.json`, `graph.json`, and `graph.mmd` alongside persisted run state in [runtime/run_store.py](/home/vinith/Desktop/Workspace/openfabric/runtime/run_store.py)
- expose `Engine.list_runs(...)`, `Engine.inspect_run(...)`, and `Engine.render_run_graph(...)` for programmatic debugging and replay-adjacent tooling
- add CLI inspection commands for listing runs, dumping a run inspection payload, and rendering stored workflow graphs as Mermaid or JSON

### Phase 8

Implemented in this change:

- extend shared node-envelope coverage to the remaining non-core agents in [agent_library/agents/](/home/vinith/Desktop/Workspace/openfabric/agent_library/agents)
- propagate node metadata through planner, operations planner, filesystem, notifier, retriever, calculator, and synthesizer outputs
- make the final `answer.final` path emit a first-class synthesizer node envelope
- expand spec contracts so planner-facing requests and user-facing final answers accept the common `node` schema

### Phase 9

Implemented in this change:

- add a dedicated persisted-run visualization app in [runtime/run_visualizer.py](/home/vinith/Desktop/Workspace/openfabric/runtime/run_visualizer.py)
- expose `/api/runs`, `/api/runs/{run_id}`, and graph-format endpoints backed by [runtime/run_store.py](/home/vinith/Desktop/Workspace/openfabric/runtime/run_store.py)
- render a browser UI with run filtering, summary cards, timeline checkpoints, Mermaid export, and an interactive SVG graph view
- expose the dashboard from the CLI with `python cli.py --serve-runs-ui`

### Phase 10

Implemented in this change:

- add persisted run observability derivation in [runtime/run_observability.py](/home/vinith/Desktop/Workspace/openfabric/runtime/run_observability.py)
- derive and store `observability.json` alongside each persisted run in [runtime/run_store.py](/home/vinith/Desktop/Workspace/openfabric/runtime/run_store.py)
- enrich run summaries and inspections with timing, validation, routing, failure, and per-agent metrics in [runtime/run_inspector.py](/home/vinith/Desktop/Workspace/openfabric/runtime/run_inspector.py)
- expose audit-style run filtering and observability inspection through the CLI and engine helpers in [cli.py](/home/vinith/Desktop/Workspace/openfabric/cli.py) and [runtime/engine.py](/home/vinith/Desktop/Workspace/openfabric/runtime/engine.py)
- surface observability in the persisted-run browser UI and add observability-aware API endpoints in [runtime/run_visualizer.py](/home/vinith/Desktop/Workspace/openfabric/runtime/run_visualizer.py)

### Phase 11

Implemented in this change:

- add strict `Agent Contract V1` foundation models in [agent_library/contracts.py](/home/vinith/Desktop/Workspace/openfabric/agent_library/contracts.py)
- add reusable agent descriptor and response helpers in [agent_library/template.py](/home/vinith/Desktop/Workspace/openfabric/agent_library/template.py)
- normalize legacy `AGENT_METADATA.methods` into contract-level `apis` during runtime capability discovery in [runtime/engine.py](/home/vinith/Desktop/Workspace/openfabric/runtime/engine.py)
- project normalized `apis` plus shared request/result schemas into the capability graph in [runtime/graph.py](/home/vinith/Desktop/Workspace/openfabric/runtime/graph.py)
- document the new builder contract in [AGENT_CONTRACT_V1.md](/home/vinith/Desktop/Workspace/openfabric/AGENT_CONTRACT_V1.md)

## Current Mapping

Today’s runtime already maps naturally to the target graph:

- `llm_operations_planner` is the primary router
- `shell_runner`, `sql_runner`, `slurm_runner`, `filesystem`, and `notifier` are executors
- `agent_library/reduction.py` is the shared reduction planning contract
- `data_reducer` is the first-class reduction node
- `validator` is already a validation node
- `synthesizer` is already the final output node

## Current Graph-Native Flow

The runtime now exposes the following execution chains directly in the graph:

- step success path:
  - `executor -> reducer? -> validator -> step_router`
- step recovery path:
  - `executor -> step_router -> replan -> replacement subgraph`
- workflow validation path:
  - `attempt -> validator -> workflow_router`
- workflow fallback path:
  - `attempt -> validator -> workflow_router -> replan -> derived attempt`
- clarification path:
  - `validator/router -> clarification`

## Remaining Gaps

The main remaining architectural gaps before production are:

- harden the visualization layer with authentication, multi-user deployment posture, and larger-run pagination/virtualization
- add exportable metrics backends and long-term audit retention patterns beyond per-run JSON artifacts
- introduce durable alerting and SLO-oriented thresholds on top of the new observability layer
