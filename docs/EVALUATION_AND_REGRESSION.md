# Evaluation and Regression

## Overview

The runtime currently uses two complementary quality gates:

- a global exhaustive NLP regression
- per-capability eval packs with thresholds

Together they protect both:

- broad end-to-end prompt behavior
- narrower capability-specific correctness

## Global Exhaustive NLP Regression

The global regression runner is `scripts/evaluate_exhaustive_nlp_regression.py`.

Purpose:

- verify that a broad set of representative prompts still produce the expected results
- catch regressions in deterministic routing, output shaping, and fallback behavior

Current evaluation dimensions:

- strict pass
- semantic pass
- fallback count

The exhaustive suite is the high-level end-to-end regression gate for the runtime as a whole.

## Capability-Pack Eval Runner

The pack runner is `scripts/evaluate_capability_packs.py`.

Purpose:

- make each capability carry its own checked-in prompt pack
- enforce per-capability thresholds
- keep promotion gates close to the capability being extended

This runner is additive. It does not replace the exhaustive regression.

## Eval Pack Models

The schemas live in `src/aor_runtime/runtime/capabilities/eval.py`.

### `CapabilityEvalCase`

Each case describes:

- `id`
- `prompt`
- one or more expectations
  - `expected`
  - `expected_contains`
  - `expected_regex`
- `expect_llm_calls`
- optional `category`
- optional `setup`
- optional `tags`

### `CapabilityEvalPack`

Each pack defines:

- `capability`
- `cases`
- `strict_threshold`
- `semantic_threshold`
- `max_llm_fallbacks`

### `CapabilityEvalResult`

Each pack result tracks:

- `capability`
- `total`
- `strict_pass`
- `semantic_pass`
- `llm_fallbacks`
- `llm_intent_calls`
- `raw_planner_llm_calls`
- `deterministic_calls`
- `failures`

## Current Eval Metrics

The pack runner tracks:

- strict
  - exact visible-output and call-count expectations
- semantic
  - content-equivalence checks
- `llm_fallbacks`
  - total LLM-involved calls as currently counted by the runner
- `llm_intent_calls`
  - typed LLM intent extraction calls
- `raw_planner_llm_calls`
  - deprecated compatibility counter; expected to remain `0`
- `deterministic_calls`
  - cases that completed without LLM help

This split is especially important now that the runtime can distinguish:

- validator-enforced action-planner execution
- optional helper-level typed LLM intent extraction in compatibility tests

## Eval Fixtures

`src/aor_runtime/runtime/eval_fixtures.py` rebuilds deterministic fixture workspaces for eval runs.

Current fixture groups include:

- notes files
- nested txt/md trees
- content-search samples
- write-output directories
- SQL sample databases
- local fetch fixtures
- SLURM command fixtures
- SLURM typed-intent fixture responses

This is why capability evals do not require a live SLURM cluster, live network, or live gateway.

## Checked-In Eval Packs

Capability packs live under `evals/capabilities/`.

Current packs include:

- `compound.json`
- `fetch.json`
- `filesystem.json`
- `prompt_suggestions.json`
- `shell.json`
- `slurm.json`
- `slurm_llm_intent.json`
- `sql.json`
- `text_transform.json`

Case IDs must be unique across all packs.

## Reports

The runners write reports under `artifacts/reports/`.

Important report outputs:

- exhaustive regression report
- capability eval report

The capability eval report includes:

- totals
- per-pack summaries
- per-case results
- threshold failures

## How Strict vs Semantic Checks Work

In broad terms:

- strict checks require the exact expected visible output and expected call counts
- semantic checks allow structured equivalence checks such as fragment presence or JSON equivalence

The exact matching logic lives in `scripts/evaluate_capability_packs.py`.

## When to Add a New Eval Pack

Add a new capability pack when:

- a new capability is introduced
- a domain-specific pack adds meaningful behavior
- a risky extension needs its own promotion gate

A new pack should:

- use deterministic fixtures whenever possible
- define thresholds intentionally
- include representative prompts that reflect real supported behavior

## When to Expand the Exhaustive Regression

Expand the global regression when:

- a change affects cross-cutting planner behavior
- an issue was found that could affect multiple capabilities
- a new failure mode deserves broad end-to-end protection

## Practical Guidance

- If a capability-specific change fails only its own pack, look there first.
- If both the pack runner and exhaustive regression fail, suspect a broader planner, output-shaping, or execution-path regression.
- If `raw_planner_llm_calls` rises unexpectedly, a prompt may have fallen off the deterministic path.

See also:

- [ADDING_A_CAPABILITY.md](./ADDING_A_CAPABILITY.md)
- [DEBUGGING_GUIDE.md](./DEBUGGING_GUIDE.md)
