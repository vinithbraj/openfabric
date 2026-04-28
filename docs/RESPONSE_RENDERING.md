# Response Rendering

The runtime separates execution telemetry from user-facing answers. Tool execution, planner events, validation checks, coverage metadata, and raw payloads remain available through session events and debug/raw modes, but OpenWebUI receives polished Markdown by default.

## Modes

- `user`: default. Shows the final result plus safe execution details such as the SQL query, SLURM command template, shell command, or filesystem operation. Hides lifecycle noise and internal metadata.
- `debug`: keeps the polished answer and may include compact debug metadata when configured.
- `raw`: preserves legacy/raw output for development and evals.

Environment flags:

- `AOR_RESPONSE_RENDER_MODE=user|debug|raw`
- `AOR_SHOW_EXECUTED_COMMANDS=true|false`
- `AOR_SHOW_VALIDATION_EVENTS=true|false`
- `AOR_SHOW_PLANNER_EVENTS=true|false`
- `AOR_SHOW_TOOL_EVENTS=true|false`
- `AOR_SHOW_DEBUG_METADATA=true|false`

`AOR_PRESENTATION_MODE` remains a backward-compatible alias for the render mode when `AOR_RESPONSE_RENDER_MODE` is not set.

## What User Mode Hides

User mode does not include lifecycle text such as `Thinking...`, `Plan ready`, `Executing runtime.return`, `Validating...`, or `Validation passed`. It also strips raw planner metadata, semantic frames, coverage blocks, gateway internals, stdout/stderr payloads, and raw command output unless that output is the requested result.

OpenAI-compatible streaming sends only the final rendered Markdown in user mode. Native event streams such as `/runs/stream` and `/sessions/{id}/events/stream` still expose events for debugging and progress tooling.

## Execution Details

The renderer appends safe execution details after the deterministic result:

- SQL: `Query Used` with a fenced SQL block and execution table.
- SLURM: `Commands Used` with fixed read-only command templates.
- Filesystem: `Operation` table with path, pattern, recursion, and aggregate settings when available.
- Shell: `Command Used` for already-approved shell execution paths.

`runtime.return` is never shown as a user-visible tool in user mode.

## Optional LLM-Assisted Presentation

LLM summaries are disabled by default:

- `AOR_ENABLE_PRESENTATION_LLM_SUMMARY=false`
- `AOR_PRESENTATION_LLM_MAX_FACTS=50`
- `AOR_PRESENTATION_LLM_MAX_INPUT_CHARS=4000`
- `AOR_PRESENTATION_LLM_MAX_OUTPUT_CHARS=1500`
- `AOR_PRESENTATION_LLM_INCLUDE_ROW_SAMPLES=false`
- `AOR_PRESENTATION_LLM_INCLUDE_PATHS=false`

When enabled, the LLM receives only deterministic sanitized facts, never raw tool output. Allowed facts are compact aggregates such as counts, status booleans, database names, safe tool names, and capped summaries. Raw SQL rows, SLURM payloads, filesystem contents, stdout/stderr, semantic frames, coverage metadata, telemetry, credentials, environment variables, and gateway internals are rejected or stripped before any LLM call.

The deterministic result and execution details remain authoritative. If LLM summarization fails or the sanitized facts fail validation, the renderer falls back to deterministic Markdown.

## Adding A Presenter

New capabilities should first add deterministic presentation in `presentation.py`, returning compact Markdown and sanitized summary facts. The response renderer can then add an executed-action formatter for the capability’s safe command/query/operation details. Capability correctness, planner behavior, and tool execution should remain independent of response rendering.
