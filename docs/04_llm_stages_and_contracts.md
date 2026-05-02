# LLM Stages And Contracts

This document explains exactly where the LLM is involved today, what each stage
is allowed to decide, and what deterministic checks follow.

The most important design idea is simple:

> The LLM proposes.  
> The runtime validates.  
> Execution happens only after deterministic trust checks.


## Where Prompt Builders Live

There is **no active `prompts/` directory** driving the current runtime.

Prompt builders live in code, inside the stage modules themselves. Structured
calls are routed through:

- `src/agent_runtime/llm/client.py`
- `src/agent_runtime/llm/structured_call.py`


## Shared Structured-Call Contract

Almost every LLM-assisted stage follows the same pattern:

1. build a stage-specific prompt
2. ask for JSON only
3. validate the result against a Pydantic schema
4. apply deterministic normalization and validation
5. either accept, reject, or fall back

`structured_call(...)` also records typed diagnostics when the LLM fails
structurally, for example:

- `transport_error`
- `invalid_json`
- `schema_validation_error`
- `empty_response`
- `parsing_error`


## Stage Inventory

| Stage | File | Why LLM Is Used | Runtime Guardrail |
| --- | --- | --- | --- |
| Prompt classification | `input_pipeline/decomposition.py` | classify broad user intent | schema validation + normalization |
| Task decomposition | `input_pipeline/decomposition.py` | break one prompt into tasks | deterministic task validation |
| Decomposition critique | `llm/critique.py` | advisory review of decomposition | advisory only, not authoritative |
| Verb/object assignment | `input_pipeline/verb_classification.py` | assign semantic verb and known object type | constrained object vocabulary |
| Capability selection | `input_pipeline/domain_selection.py` | judge bounded shortlist candidates | shortlist must come from registry |
| Capability fit | `input_pipeline/capability_fit.py` | binary semantic fit judgment | canonical compatibility + hard conflicts |
| Output overlap review | `input_pipeline/capability_fit.py` | boolean review of whether a downstream task is already satisfied by upstream declared outputs | declared output contract must still match deterministically |
| Dataflow planning | `input_pipeline/dataflow_planning.py` | propose producer-consumer refs | strict dataflow validation |
| Argument extraction | `input_pipeline/argument_extraction.py` | fill task-specific argument values | schema/path/type validation |
| DAG review | `input_pipeline/planning_review.py` | advisory review of sanitized DAG metadata | advisory only, references validated |
| Display selection | `output_pipeline/display_selection.py` | choose presentation from safe previews | display refs validated |
| Failure repair | `execution/failure_repair.py` | propose one safe repair after a failure | deterministic fit + safety re-check |


## Prompt Classification

**Why the LLM is used**

Classification is fuzzy. A human prompt can look like a question, a tool task,
or a multi-step workflow.

**What the LLM returns**

- prompt type
- likely domains
- risk level
- clarification hints

**What the runtime checks**

- schema shape
- allowed enum-like values
- deterministic normalization for obvious well-known prompt families


## Task Decomposition

**Why the LLM is used**

Turning one sentence into multiple meaningful tasks is a semantic problem.

**What the LLM returns**

- task list
- dependencies
- global constraints

**What the runtime checks**

- task ids and dependency validity
- normalized task frames
- critique-driven repair if the first decomposition is weak


## Decomposition Critique

**Why the LLM is used**

This stage looks at a proposed decomposition and asks, “Did we miss intent or
invent nonsense?”

**What the LLM returns**

- missing user intents
- hallucinated tasks
- dependency warnings
- optional repair recommendation

**What the runtime checks**

- critique is advisory
- only one bounded repair attempt is considered
- the repaired proposal still has to pass normal decomposition validation


## Verb And Object Assignment

**Why the LLM is used**

Tasks still need semantic typing such as `read system.memory` or
`create filesystem.file`.

**What the LLM returns**

- semantic verb
- object type
- risk hint

**What the runtime checks**

- object type must come from a runtime-known vocabulary
- free-form inventions are rejected or normalized
- the runtime can match loose text to a known family, but only within trusted
  semantic boundaries

This is where the system deliberately avoids letting the LLM invent arbitrary
object categories.


## Capability Selection

**Why the LLM is used**

Multiple real registered capabilities can be semantically adjacent.

**What the LLM returns**

- per-candidate fit judgments over a runtime-provided shortlist

**What the runtime checks**

- the runtime owns candidate enumeration
- the LLM cannot invent new capability ids
- selected capability ids and operation ids must match the manifest exactly

This is one of the biggest anti-drift choices in the current design.


## Capability Fit

**Why the LLM is used**

A shortlisted capability can still be semantically wrong in a subtle way.

**What the LLM returns**

- `fits: true|false`
- confidence
- optional primary failure mode
- structured reasons

**What the runtime checks**

- canonical domain compatibility
- canonical object-type compatibility
- semantic verb compatibility
- missing argument concerns
- hard conflict rejection

**Important current design choice**

The LLM fit signal is **binary and advisory**. Deterministic hard conflicts
still win.


## Output Overlap Review

**Why the LLM is used**

Sometimes the deterministic overlap matcher is too literal. A downstream task
like “tell me where the saved report lives” may be semantically satisfied by an
upstream `filesystem.write_file`, even if the task wording does not mention
`absolute_path` or `full path` directly.

**What the LLM returns**

- `satisfied_from_output: true|false`
- confidence
- task ids that must match the reviewed pair

No free-form semantic object is accepted here. The contract is deliberately
boolean.

**What the runtime checks**

- the downstream task must actually depend on the upstream task
- the downstream task must be non-mutating
- the upstream capability must already be fit-approved
- the upstream capability must already declare relevant output contract
  metadata such as output object types, output fields, or output affordances
- the runtime rejects any overlap judgment that depends on undeclared outputs

This is the current sweet spot: the model can help with fuzzy semantics, but it
cannot invent new producer outputs or bypass safety.


## Dataflow Planning

**Why the LLM is used**

When multiple tasks exist, semantic data dependency is not always obvious from
surface syntax alone.

**What the LLM returns**

- producer-consumer refs
- optional derived internal tasks

**What the runtime checks**

- no self-references
- consumer and producer types must make sense
- only dataflow-eligible arguments can be wired
- single-step or non-dataflow prompts can skip this stage


## Argument Extraction

**Why the LLM is used**

Argument filling still requires natural-language understanding.

**What the LLM returns**

- proposed argument values

**What the runtime checks**

- manifest schema validation
- path normalization
- format inference constraints
- `input_ref` handling
- safe defaults such as `overwrite=False`


## DAG Review

**Why the LLM is used**

This stage asks for a “sanity check” over a trusted DAG, using sanitized
metadata only.

**What the LLM returns**

- missing user intents
- suspicious nodes
- dependency warnings
- output expectation warnings

**What the runtime checks**

- referenced node ids must exist
- review suggestions cannot directly mutate the DAG

This stage is explicitly advisory.


## Display Selection

**Why the LLM is used**

The system can have multiple legitimate ways to present results.

**What the LLM returns**

- display type
- optional multi-section plan
- source references

**What the runtime checks**

- source refs must exist
- display types must be allowed
- the LLM may only use safe previews and validated refs

If this stage fails, the system falls back to deterministic rendering.


## Failure Repair

**Why the LLM is used**

After some safe failures, the runtime can ask for one narrow repair proposal.

**What the LLM returns**

- corrected arguments
- alternate capability suggestion
- ask-user suggestion
- skip-with-explanation suggestion

**What the runtime checks**

- repaired node must match the failed node
- no arbitrary shell text
- alternate capabilities must pass deterministic fit
- repaired DAG must pass safety again
- confirmation cannot be bypassed


## Prompt Hardening And Diagnostics

The current runtime takes prompt reliability seriously:

- prompts ask for JSON only
- schema injection is centralized at the structured-call boundary
- stage prompts are compact and deduped where possible
- malformed structured output produces typed diagnostics
- capability-fit diagnostics are carried into observability


## The Most Important Anti-Drift Choices

These are the current design decisions that most directly reduce LLM drift:

1. **Known object-type vocabulary**
2. **Runtime-owned capability shortlist**
3. **Binary capability fit**
4. **Binary output-overlap review**
5. **Schema-validated structured calls**
6. **Declared capability output contracts**
7. **Deterministic canonical compatibility**
8. **Deterministic hard safety conflicts**

That is the current core contract between the LLM and the runtime.
