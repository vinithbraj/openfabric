# Insight Layer

The insight layer runs after tool execution and presentation normalization. It interprets sanitized facts into short user-facing summaries and findings without changing planning, tool choice, SQL generation, SLURM commands, shell commands, or filesystem operations.

## Flow

```text
tool result
  -> presenter / normalizer
  -> sanitized facts
  -> deterministic insights
  -> optional LLM analyst
  -> final Markdown response
```

The deterministic result remains the source of truth. Insights are additive and appear above the normal result and execution details.

## Safety Rules

The insight layer does not receive raw command output, raw SQL rows, raw SLURM rows, raw filesystem matches, semantic frames, coverage metadata, planner telemetry, credentials, environment variables, or secrets.

LLM insights are disabled by default. When enabled, the LLM receives only sanitized compact facts and deterministic insight messages. It cannot choose tools, alter commands, generate SQL, or change factual values.

## Configuration

- `AOR_ENABLE_INSIGHT_LAYER=true|false`, default `true`
- `AOR_ENABLE_LLM_INSIGHTS=true|false`, default `false`
- `AOR_INSIGHT_MAX_FACTS=50`
- `AOR_INSIGHT_MAX_INPUT_CHARS=4000`
- `AOR_INSIGHT_MAX_OUTPUT_CHARS=1500`

## Domains

- SLURM insights cover queue pressure, problematic or drained nodes, GPU availability, accounting health, and recent failures.
- SQL insights cover zero matches, applied constraints or projections, and truncation.
- Filesystem insights cover no matches, large aggregate size, recursive scope, and truncation.

## Adding a Domain

Add a compact facts builder in `runtime/facts.py`, add deterministic rules in `runtime/insights.py`, and add renderer tests proving raw payloads are excluded and user-mode Markdown remains readable.
