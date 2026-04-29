# Adding a Capability

## Overview

The runtime is designed so new behavior is usually added as a tool plus deterministic validators/formatters. Capability packs may still be useful for compatibility helpers and tests, but top-level natural-language planning is handled by the validator-enforced LLM action planner.

The first design question is:

- can this capability fit the existing shared intent model?
- or does it need pack-local logic like SLURM?

## Decision Rule

Use shared `intent_classifier.py` and `intent_compiler.py` when:

- the new feature fits an existing intent family
- its classification rules are close to existing filesystem/sql/shell/fetch/compound behavior
- its plan structure follows an existing deterministic pattern

Use pack-local logic when:

- the domain needs specialized intent types
- safety rules are domain-specific
- compilation is not naturally expressed through the shared compiler
- output shaping or gating is domain-specific

SLURM is the reference example of the second case.

## Step-by-Step Process

### 1. Define the intent model

Choose one of:

- reuse an existing shared intent from `runtime/intents.py`
- add a new shared intent there
- define pack-local intent models in the capability module

Use pack-local intent models when the domain does not naturally belong in the shared intent set.

### 2. Implement the capability pack

Add or update a pack under `src/aor_runtime/runtime/capabilities/`.

The pack must implement:

- `classify(...)`
- `compile(...)`

Return:

- `IntentResult` from classification
- `CompiledIntentPlan` from compilation

If the capability needs typed LLM intent extraction, it must opt into that explicitly and safely. Do not add raw tool planning.

### 3. Register the pack

Register the pack in `build_default_capability_registry()` in `runtime/capabilities/registry.py`.

Be deliberate about ordering:

- specific or safety-critical packs should come before broader ones
- shell should remain behind domain capabilities that own a safer native path

### 4. Add or reuse tools

If the capability needs new tool support:

- implement tools under `src/aor_runtime/tools/`
- register them in `src/aor_runtime/tools/factory.py`
- add them to the runtime spec if they should be available in the default assistant

Tools should:

- validate inputs strictly
- return structured results
- avoid arbitrary command execution unless the tool is explicitly a shell tool

### 5. Compile to `runtime.return`

Prefer deterministic plans that end with `runtime.return`.

That ensures:

- output shape is explicit
- tests can validate both normalized value and rendered output
- user-facing responses stay stable

Use `build_output_contract(...)` when you need specific shaping behavior.

### 6. Update dataflow if needed

If the new tool returns a structured result that other steps should reference naturally, add a default ref path in `runtime/dataflow.py`.

Examples already in code:

- `sql.query` -> `rows`
- `fs.search_content` -> `matches`
- `slurm.queue` -> `jobs`

### 7. Update validator support if needed

If the tool has deterministic semantics that should be re-checked, extend `runtime/validator.py`.

This is especially important for:

- fixture-backed tools
- domain-specific tools
- safety-critical outputs

### 8. Add tests

At minimum, add tests for:

- classification
- compilation
- end-to-end deterministic behavior
- validator or tool behavior as needed

Common runtime test areas:

- `tests/runtime/test_deterministic_intents.py`
- `tests/runtime/test_compound_deterministic_intents.py`
- tool-specific tests
- capability-specific tests

### 9. Add a capability eval pack

Create a checked-in pack under `evals/capabilities/`.

Include:

- representative prompts
- expected outputs
- thresholds
- fixture-backed setup assumptions when needed

The capability should be able to pass its own pack without relying on unstable external systems.

## Safety Checklist

Before considering a capability complete:

- natural-language prompts can be planned by the action planner using the new tool manifest
- tool set is minimal and explicit
- `runtime.return` shapes the final output
- no arbitrary shell planning is introduced when a native tool path exists
- any LLM participation stops at validated structured action JSON
- validator and eval coverage exist

## Good Patterns in This Repo

- Filesystem: shared intent infrastructure + deterministic output shaping
- Compound: supported multi-step deterministic flows using shared infrastructure
- SLURM: pack-local domain capability with native tools, safety rules, and dedicated evals

## Avoid These Patterns

- letting the LLM emit raw tool calls
- letting the LLM emit `ExecutionPlan`
- adding domain behavior by routing everything through `shell.exec`
- skipping `runtime.return` when visible output shape matters
- adding a capability without a pack-specific eval gate

See also:

- [CAPABILITY_PACKS.md](./CAPABILITY_PACKS.md)
- [TOOLS_AND_RUNTIME.md](./TOOLS_AND_RUNTIME.md)
- [EVALUATION_AND_REGRESSION.md](./EVALUATION_AND_REGRESSION.md)
