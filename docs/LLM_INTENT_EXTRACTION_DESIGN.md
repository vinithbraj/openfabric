# LLM Intent Extraction Design

## Implemented Today

The runtime currently includes an optional typed LLM intent extraction path behind:

- `AOR_ENABLE_LLM_INTENT_EXTRACTION`

It is:

- implemented
- disabled by default
- currently scoped to SLURM only

Its purpose is to safely handle fuzzy SLURM inspection prompts that do not match the deterministic classifier cleanly.

## Design Goal

Allow limited LLM help without allowing the LLM to design execution.

That means the LLM may help answer:

- which SLURM intent type best matches a fuzzy request?
- what safe arguments belong on that intent?

But it may not answer:

- which raw tools to call?
- which shell or gateway commands to execute?
- what the execution graph should be?

## Core Components

### `LLMIntentExtractor`

Lives in `src/aor_runtime/runtime/llm_intent_extractor.py`.

It:

- derives a minimal schema description from allowed Pydantic models
- asks the LLM for JSON only
- validates the JSON strictly
- rejects malformed or unsafe payloads
- returns a typed intent object only on success

### Pack integration

Capability packs can opt into second-pass intent extraction via:

- `supports_llm_intent_extraction`
- `is_llm_intent_domain(...)`
- `try_llm_extract(...)`

Today only `SlurmCapabilityPack` uses this path.

## Current Flow

1. Deterministic capability classification runs first.
2. If no deterministic SLURM match is found and the feature flag is enabled, the SLURM pack may attempt typed LLM intent extraction.
3. The extractor returns either:
   - a validated typed SLURM intent
   - or no match
4. The SLURM pack applies additional safety validation.
5. If valid, the same SLURM compiler path builds a deterministic `ExecutionPlan`.
6. If no valid intent is produced, normal raw planner fallback behavior remains available.

## Hard Safety Rule

The LLM must never emit:

- raw tool calls
- shell commands
- SLURM command strings
- gateway commands
- `python.exec`
- `ExecutionPlan`

The extractor explicitly rejects payloads that look like those shapes.

## JSON-Only Contract

The expected extractor output is JSON with fields such as:

- `matched`
- `intent_type`
- `confidence`
- `arguments`
- `reason`

The extractor rejects:

- malformed JSON
- unknown intent types
- invalid arguments
- low confidence
- payloads that look like commands, tools, or plans

## Current SLURM Scope

Allowed typed outputs are limited to safe read-only SLURM intents, including:

- queue inspection
- job detail
- accounting inspection
- node and partition inspection
- metrics
- slurmdbd health

Mutating/admin requests are rejected instead of translated into a tool path.

## Safety Validation After Extraction

The SLURM pack applies domain-specific post-LLM validation:

- Pydantic validation
- safe value validation for job IDs, nodes, users, partitions, states, and times
- suspicious metacharacter rejection
- read-only scope enforcement

This second validation pass is required because schema-valid JSON is not automatically domain-safe JSON.

## Planner Metadata

Successful typed intent extraction is tracked distinctly from raw planner fallback.

Current metadata fields include:

- `planning_mode`
- `capability`
- `llm_calls`
- `llm_intent_calls`
- `raw_planner_llm_calls`
- `llm_intent_type`
- `llm_intent_confidence`
- `llm_intent_reason`

This is why evals can distinguish:

- deterministic capability matches
- typed LLM intent extraction
- raw planner usage

## Why This Design Exists

This design gives the runtime a narrow, testable use of LLM reasoning without surrendering execution authority.

It preserves the most important guarantees:

- plans are still compiler-owned
- tools are still deterministic and validated
- domain safety remains code-enforced
- evals can detect drift between deterministic behavior and LLM-assisted intent extraction

## What Must Never Be Done

- do not let the LLM emit tool calls directly
- do not let the LLM emit shell or gateway command strings
- do not let the LLM emit `ExecutionPlan`
- do not skip post-LLM safety validation
- do not use typed LLM intent extraction as a shortcut around domain capability design

See also:

- [SLURM_CAPABILITY_DESIGN.md](./SLURM_CAPABILITY_DESIGN.md)
- [CAPABILITY_PACKS.md](./CAPABILITY_PACKS.md)
