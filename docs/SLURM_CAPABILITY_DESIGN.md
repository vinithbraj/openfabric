# SLURM Capability Design

This document describes SLURM domain behavior and helper capability logic. Normal user prompts reach SLURM through the validator-enforced LLM action planner selecting native `slurm.*` tools, followed by deterministic SLURM safety, semantic, temporal, and presentation checks. See [System Design](./SYSTEM_DESIGN.md).

## Implemented Today

The runtime currently includes a read-only SLURM domain capability implemented by `SlurmCapabilityPack` and native `slurm.*` tools.

This is the first strong example of a domain-specific capability pack in the repo.

Supported scope is inspection and metrics only.

## Safe Scope

The implemented read-only SLURM surface includes:

- queue inspection
- job detail
- accounting history
- node status
- node detail
- partition status
- cluster and queue metrics
- GPU availability summaries
- slurmdbd/accounting health

Current tools:

- `slurm.queue`
- `slurm.job_detail`
- `slurm.nodes`
- `slurm.node_detail`
- `slurm.partitions`
- `slurm.accounting`
- `slurm.metrics`
- `slurm.slurmdbd_health`

## Routing and Execution

SLURM tool execution is gateway-routed.

The SLURM tools:

- build fixed read-only command templates internally
- validate user-facing fields before command construction
- route execution through the same gateway transport used by `shell.exec`
- support streamed stdout/stderr in progress mode

This is different from shell planning:

- the planner and capability pack never expose raw `squeue`, `sinfo`, `sacct`, or `scontrol` commands to the user-facing plan surface
- the pack compiles directly to structured `slurm.*` tools
- the active action planner may choose `slurm.*` tool actions, but deterministic SLURM validators still own command construction and safety

## Fixture Mode

Tests and evals do not require a live SLURM cluster.

The tools support fixture mode via:

- `AOR_SLURM_FIXTURE_DIR`

In fixture mode, the tools read prepared outputs instead of calling a live gateway/cluster. This is how SLURM capability evals remain deterministic.

## Capability-Pack Helper Behavior

`SlurmCapabilityPack` is pack-local rather than a thin wrapper over the shared intent classifier/compiler.

It owns:

- SLURM-specific intent types
- SLURM semantic frame extraction
- request and constraint coverage validation
- SLURM prompt classification
- compilation to `slurm.*` plus `runtime.return`
- optional typed LLM intent extraction for fuzzy SLURM prompts
- SLURM-specific safety rules

When compatibility code calls the pack directly, the flow is:

1. Extract a `SlurmSemanticFrame` from the prompt.
2. Resolve every `SlurmRequest` into a read-only typed SLURM intent.
3. Validate safety and coverage before execution.
4. Compile only to `slurm.*` tools and `runtime.return`.

Compound prompts use `SlurmCompoundIntent`. A prompt such as “How many jobs are running and pending, and are there any problematic nodes?” becomes separate covered child intents for running job count, pending job count, and problematic nodes. If any requested fact is missing, the runtime returns a SLURM-specific safe failure instead of a partial answer.

## Safety Rules

### Allowed

Read-only inspection and metrics only.

### Blocked

These operations must not be compiled to a tool:

- `sbatch`
- `scancel`
- `scontrol update`
- drain
- resume
- requeue
- kill job
- submit job
- change partition
- arbitrary admin mutation

### Why

The SLURM capability is intended to provide safe operational visibility, not cluster mutation. Domain capabilities should narrow risk, not expand it.

## Output Strategy

SLURM plans compile to `runtime.return` with explicit output contracts.

Current output patterns include:

- queue/accounting -> `jobs`
- nodes -> `nodes`
- partitions -> `partitions`
- detail views -> raw text or structured fields
- metrics -> structured metrics payload
- compound prompts -> JSON with `results` and `coverage`

This keeps outputs deterministic and compatible with validation and eval gates.

Coverage metadata includes:

- `slurm_semantic_frame`
- `slurm_requests_extracted`
- `slurm_requests_covered`
- `slurm_requests_missing`
- `slurm_constraints_extracted`
- `slurm_constraints_covered`
- `slurm_constraints_missing`
- `slurm_coverage_passed`
- `slurm_tools_used`

## Typed LLM Intent Extraction

The SLURM pack is also the current owner of the optional typed LLM intent extractor path.

That path:

- is disabled by default
- only runs after deterministic classification misses
- only accepts validated typed intent JSON, including bounded compound intents
- rejects raw commands, argv, shell fields, tool calls, gateway commands, and execution plans
- runs the same safety and coverage validators as deterministic extraction
- still compiles through the same SLURM compiler path

The LLM never emits commands or plans.

## Guidance for Future SLURM Extensions

Future SLURM work should keep these rules:

- prefer new native `slurm.*` tools over shell planning
- keep the scope read-only unless the architecture and safety model are explicitly redesigned
- validate all domain inputs strictly
- keep fixture-backed tests and evals
- add capability-pack eval coverage for every meaningful new prompt family

## What Must Not Be Done

- do not route supported SLURM inspection prompts through `shell.exec`
- do not let the LLM emit raw SLURM commands
- do not let the planner produce arbitrary gateway commands
- do not add mutation/admin operations casually under the same capability surface

See also:

- [LLM_INTENT_EXTRACTION_DESIGN.md](./LLM_INTENT_EXTRACTION_DESIGN.md)
- [TOOLS_AND_RUNTIME.md](./TOOLS_AND_RUNTIME.md)
