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

### Deterministic miss

Symptoms:

- planner metadata shows `planning_mode` as `direct` or `hierarchical`
- `raw_planner_llm_calls` is non-zero

Look at:

- capability-pack classification rules
- pack ordering in `CapabilityRegistry`
- prompt wording compared to existing deterministic tests and evals

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

Look at:

- `AOR_SLURM_FIXTURE_DIR`
- gateway node resolution
- command preview and gateway trace logs
- SLURM input validation failures

### Prompt suggestion issue

Symptoms:

- failed output contains weak or wrong suggestions

Look at:

- failure classification
- prompt suggestion generation
- whether the failure was deterministic, raw planner, or typed LLM intent related

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

- was the request handled deterministically?
- did it use typed LLM intent extraction?
- did it fall all the way back to the raw planner?

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
