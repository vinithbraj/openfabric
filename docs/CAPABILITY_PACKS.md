# Capability Packs

## Overview

Capability packs are the runtime’s top-level deterministic routing mechanism. `TaskPlanner` delegates supported natural-language requests to `CapabilityRegistry`, and the registry asks each registered `CapabilityPack` whether it can classify the request into a typed intent.

If a pack matches:

- it returns an `IntentResult`
- the registry sends the intent back to the matching pack for compilation
- the pack returns a `CompiledIntentPlan`

This keeps intent selection and plan generation close to the capability that owns the behavior.

## Core Types

The pack interface lives in `src/aor_runtime/runtime/capabilities/base.py`.

### `ClassificationContext`

Carries inputs needed during classification:

- `schema_payload`
- `allowed_tools`
- `settings`
- `input_payload`

This lets a pack make safe decisions using the current runtime spec, database schema context, and app settings.

### `CompileContext`

Carries:

- `allowed_tools`
- `settings`

This is the pack’s safety boundary for compilation. A pack should only compile to tools available in the current runtime spec.

### `CompiledIntentPlan`

Wraps the compiled `ExecutionPlan` plus planning metadata:

- `plan`
- `planning_mode`
- `metadata`

This is how packs communicate whether a plan came from a standard deterministic path or a more specialized path like typed LLM intent extraction.

### `CapabilityPack`

Each pack implements:

- `classify(goal, context) -> IntentResult`
- `compile(intent, context) -> CompiledIntentPlan | None`

Optional second-pass hooks also exist:

- `supports_llm_intent_extraction`
- `is_llm_intent_domain(...)`
- `try_llm_extract(...)`

Today only SLURM opts into those hooks.

## Registry Order

The default registry is built in `src/aor_runtime/runtime/capabilities/registry.py`.

Current order:

1. `compound`
2. `filesystem`
3. `text_transform`
4. `sql`
5. `slurm`
6. `shell`
7. `fetch`

This order matters. More specific or higher-priority capabilities should appear before broader ones. For example, SLURM is placed before shell so SLURM inspection prompts do not accidentally become shell prompts.

## Classification Lifecycle

The registry does three things:

1. Try deterministic pack classification in order
2. If enabled, try pack-scoped typed LLM intent extraction for packs that opt in
3. If nothing matches, return an unmatched result so `TaskPlanner` can fall back to the raw planner path

In other words:

- capability packs are the first deterministic routing layer
- typed LLM intent extraction is a capability-owned second chance, not a raw planner shortcut
- the generic planner is the last resort

## Two Pack Patterns in the Current Repo

### Pattern 1: Thin wrappers over shared deterministic infrastructure

Several packs use the shared classifier/compiler modules:

- `FilesystemCapabilityPack`
- `SqlCapabilityPack`
- `ShellCapabilityPack`
- `FetchCapabilityPack`
- `CompoundCapabilityPack`

These packs:

- call `runtime/intent_classifier.py`
- validate that the returned intent type belongs to the pack
- call `runtime/intent_compiler.py`

This pattern is good when:

- the intent shapes are already represented in `runtime/intents.py`
- the compilation logic is structurally similar to existing deterministic capabilities

### Pattern 2: Pack-local logic

`SlurmCapabilityPack` is the main example.

It defines:

- domain-specific intent types
- domain-specific classification logic
- pack-local compiler logic
- read-only safety enforcement
- optional typed LLM intent extraction

This pattern is appropriate when:

- the domain has specialized safety rules
- the output contracts need domain-specific shaping
- the intent space does not fit the shared classifier/compiler cleanly

## Current Packs

### `CompoundCapabilityPack`

Handles supported multi-step deterministic workflows such as:

- produce a result
- transform it
- save it
- return it

Uses shared `intent_classifier.py` and `intent_compiler.py`.

### `FilesystemCapabilityPack`

Handles:

- read line
- count files
- list files
- native content search
- write-and-return patterns

Uses shared deterministic infrastructure.

### `TextTransformCapabilityPack`

This pack is intentionally top-level disabled right now.

It exists to represent transform intent types, but `classify()` always returns no match for top-level user prompts. Transform behavior is currently exercised through supported compound flows instead.

### `SqlCapabilityPack`

Handles deterministic SQL count/select prompts using shared intent logic, then compiles to `sql.query` and `runtime.return`.

### `SlurmCapabilityPack`

Handles read-only SLURM inspection and metrics. This is the strongest current example of a true domain pack.

It owns:

- its own intent models
- its own classifier/compiler
- SLURM safety policy
- optional typed LLM intent extraction

### `ShellCapabilityPack`

Handles explicit shell prompts only. It should not be the default path for domain behavior that already has a safer dedicated capability.

### `FetchCapabilityPack`

Handles deterministic `fetch/curl and extract` prompts using shared infrastructure.

## Planning Metadata

Packs can contribute metadata through `IntentResult.metadata` and `CompiledIntentPlan.metadata`.

Current fields used by the runtime include:

- `planning_mode`
- `capability`
- `intent_type`
- `llm_calls`
- `llm_intent_calls`
- `raw_planner_llm_calls`
- `llm_intent_type`
- `llm_intent_confidence`
- `llm_intent_reason`

That metadata flows into planner tracking, session state, events, and eval reporting.

## Safety Rules for Pack Authors

When adding or updating a pack:

- classification should be conservative
- compilation must only use allowed tools
- domain packs should prefer native tools over raw shell planning
- any optional LLM path must produce typed intent only, never tools or plans
- unsafe mutation/admin operations should fail closed

See also:

- [INTENT_AND_COMPILER_MODEL.md](./INTENT_AND_COMPILER_MODEL.md)
- [ADDING_A_CAPABILITY.md](./ADDING_A_CAPABILITY.md)
- [SLURM_CAPABILITY_DESIGN.md](./SLURM_CAPABILITY_DESIGN.md)
