# Debugging Guide

## Overview

The runtime is designed to leave a trail:

- persisted session state
- step history
- planner metadata
- event logs
- eval reports

Use those first. Most issues can be localized without guessing.

## First Tools to Reach For

### CLI progress mode

Use `--progress` to see:

- planner start/completion
- step start/completion
- command previews
- streamed stdout/stderr for streaming tools
- validator and finalize milestones

This is the fastest way to understand which layer failed.

### Session inspection

The runtime persists sessions, history, and final output in SQLite-backed session records.

Useful places to inspect:

- session state
- `history`
- `final_output`
- `planning_metadata`
- `metrics`

### Event inspection

Stored events are available through the CLI/API session flow and the native API event endpoints.

Important event types:

- `planner.started`
- `planner.completed`
- `executor.step.started`
- `executor.step.output`
- `executor.step.completed`
- `validator.started`
- `validator.completed`
- `finalize.completed`

## Common Failure Types

### Action planner failure

Symptoms:

- planner metadata shows `planning_mode` as `validator_enforced_action_planner`
- action-plan metadata includes validation or canonicalization errors

Look at:

- `raw_action_plan`, `normalized_action_plan`, and `canonicalized_action_plan`
- action validation errors
- tool/domain/result-shape contract failures

### Dataflow reference error

Symptoms:

- executor fails with missing path or unknown reference errors

Look at:

- `$ref` and `path` wiring in the compiled plan
- default ref path behavior in `runtime/dataflow.py`
- whether the producer tool returned the expected shape

### Output-shaping mismatch

Symptoms:

- plan succeeds but strict or semantic output checks fail

Look at:

- final `runtime.return` step
- `OutputContract`
- `normalize_output(...)`
- `render_output(...)`
- any `json_shape` or `path_style` choice

### Validator mismatch

Symptoms:

- step executes but validation fails

Look at:

- `runtime/validator.py`
- actual tool result vs validator’s recomputed expected result
- fixture mode or environment assumptions

### SLURM fixture or gateway issue

Symptoms:

- SLURM tool failures
- inconsistent queue/node/accounting outputs
- compound SLURM prompts return a safe coverage failure

Look at:

- `AOR_SLURM_FIXTURE_DIR`
- gateway node resolution
- command preview and gateway trace logs
- `slurm_semantic_frame`
- `slurm_requests_extracted`
- `slurm_requests_covered`
- `slurm_requests_missing`
- `slurm_constraints_covered`
- `slurm_constraints_missing`
- `slurm_coverage_passed`

Compound SLURM prompts are coverage-gated. If the user asks for running jobs, pending jobs, and problematic nodes, all three requests must be represented by typed read-only SLURM intents before any `slurm.*` tool runs. Mutation/admin prompts such as `sbatch`, `scancel`, `drain`, `resume`, `requeue`, or service restarts compile only to a refusal.

### SQL schema or quoting issue

Symptoms:

- `relation does not exist`
- `column does not exist`
- only `public` tables appear
- mixed-case PostgreSQL names fail unless quoted

Start with:

- `List all tables in <database>.`
- `Describe table <schema>."<MixedCaseTable>" in <database>.`

The SQL catalog includes non-system PostgreSQL schemas outside `public` and preserves exact table/column case. The correct PostgreSQL form for a mixed-case table in a schema is `schema."Table"`, not `"schema.Table"`.

Broad SQL generation is enabled only when `AOR_ENABLE_SQL_LLM_GENERATION=true`. With it disabled, deterministic schema-aware SQL still works, and broad joins/grouping questions fail safely with SQL-specific suggestions.

If a constrained prompt returns a SQL-safe failure with `sql_constraint_unresolved` or `sql_constraint_uncovered`, inspect the planning metadata fields `sql_constraints_extracted`, `sql_constraints_covered`, and `sql_constraints_missing`. A constrained prompt must not execute as a plain unfiltered `COUNT(*)`.

Mutation requests such as `delete`, `drop`, `alter`, and `update` are rejected before execution.
- SLURM input validation failures

### Prompt suggestion issue

Symptoms:

- failed output contains weak or wrong suggestions

Look at:

- failure classification
- prompt suggestion generation
- whether the failure happened during action planning, validation, execution, or final presentation

## Planner Metadata

`planning_metadata` is one of the most useful debugging artifacts.

Key fields:

- `planning_mode`
- `capability`
- `llm_intent_type`
- `llm_intent_confidence`
- `llm_intent_reason`
- `llm_intent_calls`
- `raw_planner_llm_calls`

Use this to answer:

- did the action planner produce usable structured actions?
- did canonicalization repair dataflow or temporal arguments?
- did validators refuse a tool, schema reference, path, command, or output shape?

## Gateway Tracing

For gateway-routed execution:

- start the gateway with command tracing enabled
- use runtime progress mode to see the previewed command at the client side

This is especially helpful for:

- shell execution
- SLURM tool routing

## Eval Debugging

### Exhaustive regression

Use the exhaustive report when a broad planner or output regression is suspected.

### Capability packs

Use the capability eval report when a feature-specific regression is suspected.

For each case, check:

- `content`
- strict/semantic flags
- `planning_mode`
- `llm_calls`
- `llm_intent_calls`
- `raw_planner_llm_calls`

This is the fastest way to tell whether a prompt fell off the deterministic path.

## Practical Debugging Order

1. Reproduce with `--progress`
2. Inspect final session state and planner metadata
3. Inspect step history
4. Inspect stored events
5. Compare against validator expectations
6. Compare against relevant deterministic tests or eval cases

## Where to Look by Problem Area

- Planner routing:
  - `runtime/planner.py`
  - `runtime/capabilities/registry.py`
  - relevant capability pack
- Shared deterministic parsing:
  - `runtime/intent_classifier.py`
  - `runtime/intent_compiler.py`
- Output issues:
  - `runtime/output_contract.py`
  - `tools/runtime_return.py`
  - `runtime/executor.py`
- Dataflow issues:
  - `runtime/dataflow.py`
- SLURM:
  - `runtime/capabilities/slurm.py`
  - `tools/slurm.py`
- Eval issues:
  - `scripts/evaluate_capability_packs.py`
  - `runtime/eval_fixtures.py`

See also:

- [EVALUATION_AND_REGRESSION.md](./EVALUATION_AND_REGRESSION.md)
- [TOOLS_AND_RUNTIME.md](./TOOLS_AND_RUNTIME.md)
