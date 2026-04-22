# Planner Normalization

This document explains the current planner architecture after the normalization pass.

## Goal

Move planner knowledge out of bespoke family-specific branches and into agent metadata wherever possible, while keeping the deterministic rules that are genuinely required for safety and workflow quality.

## Bucket 1: Necessary Deterministic Rules

These rules are intentionally still hardcoded because they protect execution quality, safety, or workflow structure:

- unsafe machine request guards
- deterministic shell inventory/count workflows
- compound request splitting and recovery for shared-verb requests
- workflow-shape repairs when the LLM collapses required multi-step plans
- exact SQL instance selection when multiple configured databases exist
- explicit file-read routing where a direct path read should prefer `filesystem`

These are guardrails, not general routing preferences.

## Bucket 2: Rules Removed From Bespoke Planner Branches

These used to be planner-family heuristics and are now descriptor-driven:

- shell/sql/slurm/notifier/filesystem/calculator routing keyword preferences
- native count preference
- structured follow-up preference
- preferred task shapes per agent
- advertised instruction operations for planner prompting
- routing priority biases

These now live in `planning_hints` on `AGENT_DESCRIPTOR` and, where useful, on individual APIs.

## Bucket 3: Rules That Now Belong In Agent Contracts

Every new agent should describe planner-facing behavior through `planning_hints`, for example:

- `keywords`
- `anti_keywords`
- `preferred_task_shapes`
- `instruction_operations`
- `structured_followup`
- `native_count_preferred`
- `routing_priority`

This makes new agents more plug-and-play and reduces planner edits.

## Runtime Effects

The planner now:

- formats discovered planning hints into the runtime capability guide
- scores candidate agents from descriptor hints instead of mostly family-specific boosts
- exposes planning hints through capability summaries
- uses descriptor hints in prompts so the LLM sees agent-local routing guidance

## Remaining Intentional Hardcoding

Some planner logic is still intentionally specialized:

- deterministic shell command derivation for known-safe operational workflows
- SQL-vs-shell ambiguity handling for repository file scans
- Slurm summary shortcuts for node inventory and elapsed-time summary tasks
- compound request decomposition repairs

These are still useful because they protect correctness in real-world live scenarios where the planner LLM can collapse or drift.

## What “Ideal State” Means Here

Ideal does not mean zero deterministic rules.

Ideal means:

- routing preferences come from descriptors
- instance-specific hints come from config metadata
- hardcoded rules exist only for safety, exact instance selection, and known workflow-shape repairs
- adding a new agent normally requires metadata, not planner surgery
