# Tools and Runtime Behavior

## Tool Registry

The runtime tool registry is built in `src/aor_runtime/tools/factory.py`.

Current tool families:

- filesystem
  - `fs.exists`
  - `fs.not_exists`
  - `fs.copy`
  - `fs.read`
  - `fs.write`
  - `fs.mkdir`
  - `fs.list`
  - `fs.glob`
  - `fs.find`
  - `fs.search_content`
  - `fs.size`
- SLURM
  - `slurm.queue`
  - `slurm.job_detail`
  - `slurm.nodes`
  - `slurm.node_detail`
  - `slurm.partitions`
  - `slurm.accounting`
  - `slurm.metrics`
  - `slurm.slurmdbd_health`
- shell
  - `shell.exec`
- SQL
  - `sql.query`
- internal
  - `python.exec`
  - `runtime.return`

The example runtime spec in `examples/general_purpose_assistant.yaml` controls which of those tools are enabled for a given assistant.

## Filesystem Tools

Filesystem tools live in `src/aor_runtime/tools/filesystem.py`.

They provide deterministic local file and directory operations such as:

- reading file contents
- writing content
- listing directories
- globbing files
- recursive searching by filename pattern
- checking size and existence

All filesystem paths are resolved through the configured workspace rules.

## Native Content Search

`fs.search_content` lives in `src/aor_runtime/tools/search_content.py` and is the preferred deterministic path for file content search.

It supports:

- root path
- needle text
- filename pattern
- recursive or top-level traversal
- `name`, `relative`, or `absolute` path style
- optional case-insensitive search
- `max_matches`

Return shape:

- `matches`
  - the shaped path strings used by most deterministic plans
- `entries`
  - structured records with:
    - `name`
    - `path`
    - `relative_path`
    - `matched_lines`

Important behavior:

- skips missing or invalid directories with explicit tool errors
- ignores binary-like files containing NUL bytes
- ignores unreadable or undecodable files instead of crashing
- sorts results by relative path

## File Query Normalization

`src/aor_runtime/runtime/file_query.py` normalizes common natural-language file-discovery phrasing into `FileQuerySpec`.

It infers:

- `root_path`
- `pattern`
- `recursive`
- `path_style`

Examples of normalization behavior:

- “top-level” and “only in this folder” map to non-recursive search
- “under” and “recursively” map to recursive search
- “txt files” and similar phrases normalize to `*.txt`
- “filenames only” maps to `path_style="name"`

This lets the shared classifier/compiler consistently compile file queries to `fs.glob` or `fs.find`.

## Shell and Gateway Routing

`shell.exec` lives in `src/aor_runtime/tools/shell.py` and executes through the gateway transport in `src/aor_runtime/tools/gateway.py`.

Key rules:

- the runtime routes shell execution to a logical node
- gateway requests go to `/exec` or `/exec/stream`
- progress mode and streamed API endpoints use the streaming path
- high-risk destructive commands are blocked when destructive shell is disabled

The gateway layer is transport only. Shell capability planning remains separate from domain-specific capabilities like SLURM.

## SQL

`sql.query` is the read-only database execution tool used by deterministic SQL intents.

The deterministic compiler emits:

- `SqlCountIntent` -> `sql.query` -> `runtime.return`
- `SqlSelectIntent` -> `sql.query` -> `runtime.return`

The validator re-checks SQL outputs against deterministic expectations where applicable.

## SLURM Tools

SLURM tools live in `src/aor_runtime/tools/slurm.py`.

They implement read-only inspection and metrics:

- queue
- job detail
- nodes
- node detail
- partitions
- accounting
- metrics
- slurmdbd health

Important runtime properties:

- gateway-routed execution
- strict argument validation
- no arbitrary command input surface
- fixture mode for tests and evals via `AOR_SLURM_FIXTURE_DIR`

See [SLURM_CAPABILITY_DESIGN.md](./SLURM_CAPABILITY_DESIGN.md) for the domain-level design.

## `runtime.return`

`runtime.return` lives in `src/aor_runtime/tools/runtime_return.py` and is the final deterministic shaping tool.

It takes:

- `value`
- `mode`
  - `text`
  - `csv`
  - `json`
  - `count`
- optional `output_contract`

It returns:

- `value`
  - normalized structured value
- `output`
  - final rendered string

This tool is what turns structured intermediate results into stable user-facing output.

## Output Contracts

`src/aor_runtime/runtime/output_contract.py` defines `OutputContract`.

Fields:

- `mode`
- `path_style`
- `json_shape`
- `include_extra_text`

### Modes

- `text`
- `csv`
- `json`
- `count`

### JSON shapes

The runtime currently uses:

- `matches`
- `rows`
- `count`
- `value`

Examples:

- file listings in JSON often use `{"matches": [...]}`.
- SQL or row-oriented outputs may use `{"rows": [...]}`.
- count responses may be scalar or `{"count": N}` depending on contract.

### Path styles

For sequence-like outputs, the contract can apply:

- `name`
- `relative`
- `absolute`

### Normalization vs rendering

`normalize_output(...)`:

- converts the raw tool result into the intended structured shape

`render_output(...)`:

- converts the normalized value into the final string returned to the user

This split is why deterministic output can be validated both semantically and textually.

### Row rendering

When `json_shape="rows"`:

- `csv` mode renders rows as CSV
- `text` mode renders rows as a text table

### Count rendering

`count` mode accepts several source shapes and normalizes them into a single integer. It can also wrap that value as `{"count": N}` when requested.

## Dataflow

`src/aor_runtime/runtime/dataflow.py` resolves values between steps using `$ref`.

Core concepts:

- `{"$ref": "alias"}`
- optional `path`
  - `{"$ref": "alias", "path": "rows.0.name"}`

This is how later steps use outputs from earlier steps without relying on the LLM to wire the plan manually.

### Default ref paths

The runtime assigns default output subpaths for common tools, for example:

- `fs.read` -> `content`
- `fs.glob` -> `matches`
- `fs.search_content` -> `matches`
- `sql.query` -> `rows`
- `shell.exec` -> `stdout`
- `runtime.return` -> `value`
- `slurm.queue` / `slurm.accounting` -> `jobs`
- `slurm.nodes` -> `nodes`
- `slurm.partitions` -> `partitions`
- `slurm.job_detail` / `slurm.node_detail` -> `fields`

### `fs.write` coercion

When `fs.write` receives non-string content through dataflow, `dataflow.py` coerces it into a string safely. Special handling exists for:

- `runtime.return` results
- `python.exec` results
- shell results
- generic dict/list values via JSON serialization

## Executor Behavior

`src/aor_runtime/runtime/executor.py` is responsible for deterministic step execution.

Key behaviors:

- resolves dataflow references before invoking a tool
- supports streamed tools when progress/event streaming is enabled
- exposes preview information such as node and command text
- records per-step success/error results in `StepLog`

### Streaming tools

If a tool exposes `stream(...)`, the executor:

- consumes stdout/stderr chunks
- emits `executor.step.output` events
- accumulates final stdout/stderr
- asks the tool to build the final structured result

This is how shell and gateway-routed SLURM tools integrate with progress mode and SSE.

### Final-output summarization

If the last step is not already `runtime.return`, `summarize_final_output(...)` still tries to shape the final result into a concise output based on the goal and last tool result.

That said, deterministic capabilities should prefer explicit `runtime.return` so output stays stable and testable.

## Validator

`src/aor_runtime/runtime/validator.py` re-checks runtime behavior after execution.

It validates tool results against deterministic truth, including:

- filesystem results
- search-content results
- SLURM fixture-backed results
- `runtime.return` behavior

This validator is a major safety layer because it prevents silently accepting planner or tool mismatches.

See also:

- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [INTENT_AND_COMPILER_MODEL.md](./INTENT_AND_COMPILER_MODEL.md)
- [DEBUGGING_GUIDE.md](./DEBUGGING_GUIDE.md)
