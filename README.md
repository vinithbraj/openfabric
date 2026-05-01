# OpenFABRIC V10 Reset Runtime

This branch intentionally removes the previous internal planner, semantic-frame,
tool, SQL, SLURM, filesystem, shell, eval, and rendering machinery.

The preserved system is a clean interface shell:

- Gateway agent remains in `gateway_agent/`.
- Config loading remains in `config.yaml`, `config_ucla.yaml`, and
  `src/aor_runtime/config.py`.
- The FastAPI/OpenAI-compatible/OpenWebUI surface remains available.
- Every prompt is currently passed through and echoed as the assistant response.

This is a reset point for designing the next runtime without old internal
behavior accidentally participating.

## Behavior

Given:

```text
count of patients in dicom
```

The runtime returns:

```text
count of patients in dicom
```

There are no LLM calls, tool invocations, SQL queries, shell commands, SLURM
queries, action DAGs, or output-reconstruction steps in this reset runtime.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
aor serve
```

OpenAI-compatible endpoints:

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/chat/completions`

Compatibility endpoints retained for clients:

- `POST /runs`
- `POST /runs/stream`
- `POST /sessions`
- `GET /sessions`
- `GET /sessions/{id}`
- `GET /sessions/{id}/events`

## CLI

```bash
aor run examples/general_purpose_assistant.yaml --prompt "hello"
aor chat examples/general_purpose_assistant.yaml
aor serve --host 0.0.0.0 --port 8310
```

The `spec_path` argument is retained for compatibility but is not executed.

## Gateway Agent

The gateway agent is preserved unchanged as the future node-local interface:

```bash
cd gateway_agent
python -m pip install -e .
gateway-agent
```

See `gateway_agent/README.md` for its own usage.

## Reset Notes

The old architecture documentation and eval packs were removed because they
described systems that are no longer active on this branch. The new design can
now be rebuilt against this much smaller surface.
