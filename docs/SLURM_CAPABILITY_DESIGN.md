# SLURM Capability Design

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

## Fixture Mode

Tests and evals do not require a live SLURM cluster.

The tools support fixture mode via:

- `AOR_SLURM_FIXTURE_DIR`

In fixture mode, the tools read prepared outputs instead of calling a live gateway/cluster. This is how SLURM capability evals remain deterministic.

## Capability-Pack Behavior

`SlurmCapabilityPack` is pack-local rather than a thin wrapper over the shared intent classifier/compiler.

It owns:

- SLURM-specific intent types
- SLURM prompt classification
- compilation to `slurm.*` plus `runtime.return`
- optional typed LLM intent extraction for fuzzy SLURM prompts
- SLURM-specific safety rules

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

This keeps outputs deterministic and compatible with validation and eval gates.

## Typed LLM Intent Extraction

The SLURM pack is also the current owner of the optional typed LLM intent extractor path.

That path:

- is disabled by default
- only runs after deterministic classification misses
- only accepts validated typed intent JSON
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
