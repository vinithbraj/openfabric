# V10 Echo Reset

The V10 branch is a deliberate architecture reset.

## Preserved

- `gateway_agent/`
- app config loading
- CLI command names
- FastAPI server startup
- OpenAI-compatible chat API
- OpenWebUI-compatible streaming shape

## Removed

- semantic frame internals
- action planner internals
- domain tools
- SQL/SLURM/filesystem/shell runtime execution
- old eval packs
- old runtime tests
- old architecture documents

## Current Data Flow

```mermaid
flowchart LR
    A[User Prompt] --> B[FastAPI or CLI]
    B --> C[Prompt Extractor]
    C --> D[Echo Engine]
    D --> E[Final Output]
    E --> F[User Response]
```

## Contract

The reset runtime does one thing:

```text
assistant_response = latest_actionable_user_prompt
```

No raw operational data is read, generated, or routed because no tools are
invoked.
