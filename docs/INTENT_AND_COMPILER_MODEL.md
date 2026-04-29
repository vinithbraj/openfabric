# Intent and Compiler Model

## Overview

Typed intents are retained as helper and compatibility models. The active runtime first asks the LLM action planner for structured tool actions, then canonicalizes and validates those actions before compiling an `ExecutionPlan`.

This model separates:

- natural-language understanding
- deterministic plan construction
- tool execution

That separation still matters for tests and helper compilers, but natural-language runtime requests now use the action-planning LLM.

## Shared Intents

Shared typed intents live in `src/aor_runtime/runtime/intents.py`.

Current shared intent families include:

- filesystem
  - `ReadFileLineIntent`
  - `CountFilesIntent`
  - `ListFilesIntent`
  - `SearchFileContentsIntent`
  - `WriteTextIntent`
  - `WriteResultIntent`
- SQL
  - `SqlCountIntent`
  - `SqlSelectIntent`
- shell
  - `ShellCommandIntent`
- fetch
  - `FetchExtractIntent`
- transforms
  - `TransformIntent`
  - `TransformChainIntent`
- compound
  - `CompoundIntent`

These intents are still used by compatibility helpers, capability-pack unit tests, and some evaluator fixtures.

## Shared Classifier and Compiler Are Helper Infrastructure

`src/aor_runtime/runtime/intent_classifier.py` and `src/aor_runtime/runtime/intent_compiler.py` are not dead code.

They remain shared deterministic infrastructure used by:

- filesystem
- SQL
- shell
- fetch
- compound

The architecture changed at the top level, not by removing those modules.

Today the real stack is:

- `TaskPlanner` always calls the validator-enforced action planner
- compatibility tools and tests may call capability packs directly
- many packs call the shared classifier/compiler underneath

So the correct way to think about the shared modules is:

- not the top-level routing layer
- still core deterministic infrastructure for several capability packs

## Shared Deterministic Compilation

`runtime/intent_compiler.py` turns supported intents into `ExecutionPlan` objects.

Typical patterns include:

- filesystem discovery -> `runtime.return`
- SQL query -> `runtime.return`
- shell command -> `runtime.return`
- compound workflows with intermediate aliases and final return shaping

The compiler uses `build_output_contract(...)` to make the output shape explicit at plan-construction time.

## Pack-Local Compilation

`SlurmCapabilityPack` is the first strong domain-specific example.

It does not rely on the shared classifier/compiler modules for its core behavior. Instead it:

- defines SLURM-specific intent models
- parses SLURM prompts directly
- compiles to `slurm.*` tools plus `runtime.return`

This is the model to follow when a new domain needs:

- custom safety policy
- domain-specific output handling
- pack-owned classifier/compiler behavior

## What Compilation Produces

Every intent compiler path eventually produces an `ExecutionPlan` made of ordered steps. Each step contains:

- `id`
- `action`
- `args`
- optional `input`
- optional `output`

Plans can pass data between steps using named outputs and `$ref` values. Final user output is usually produced by a final `runtime.return` step.

## `IntentResult` and Metadata

Capability classification returns `IntentResult`, which contains:

- `matched`
- `intent`
- `reason`
- `metadata`

The `metadata` field is important in the current architecture because it carries planner-facing context such as:

- `planning_mode`
- `capability`
- `llm_calls`
- `llm_intent_calls`
- `raw_planner_llm_calls`
- `llm_intent_type`
- `llm_intent_confidence`
- `llm_intent_reason`

## Planner Modes and Counters

`TaskPlanner` tracks the path used to produce a plan.

Important fields:

- `planning_mode`
  - normally `validator_enforced_action_planner`
- `llm_calls`
  - total planner-related LLM calls for the request
- `llm_intent_calls`
  - typed LLM intent extraction calls
- `raw_planner_llm_calls`
  - deprecated compatibility counter; normally `0`
- `capability`
  - matched capability pack name when applicable

These metrics are preserved in runtime state and eval reporting.

## Why `runtime.return` Matters

Compilation is not finished when the tool step is chosen. The plan must also define how the result becomes the final user response.

That is why most deterministic compilers end with `runtime.return`:

- tool steps fetch or produce structured data
- `runtime.return` applies the `OutputContract`
- the final user output becomes stable and testable

This is a major difference from LLM-planned agent systems that let the model invent the final textual answer from tool results.

## Safety Rule

The runtime’s intent model is designed so that:

- the shared deterministic compiler emits fixed plans
- pack-local compilers emit fixed plans
- any optional LLM participation must stop at validated intent JSON

The LLM must never be allowed to emit:

- raw tool calls
- shell commands
- gateway commands
- `python.exec`
- raw `ExecutionPlan` payloads

See also:

- [CAPABILITY_PACKS.md](./CAPABILITY_PACKS.md)
- [TOOLS_AND_RUNTIME.md](./TOOLS_AND_RUNTIME.md)
- [LLM_INTENT_EXTRACTION_DESIGN.md](./LLM_INTENT_EXTRACTION_DESIGN.md)
