# Codex-Style Standalone UI

This repo also includes a standalone browser UI that talks to the exact same
OpenAI-compatible API surface used by Open WebUI:

- `GET /v1/models`
- `POST /v1/chat/completions`

That means you can use either:

- Open WebUI against the gateway
- the built-in Codex-style standalone UI

Both hit the same OpenFabric planner backend.

## Start

Keep your LLM endpoint running first, then launch the standalone UI:

```bash
bash scripts/start_codex_ui.sh
```

Default bind:

```text
http://127.0.0.1:8314
```

Runtime configuration UI:

```text
http://127.0.0.1:8314/config
```

You can also run the Python entrypoint directly:

```bash
python codex_ui_gateway.py --host 127.0.0.1 --port 8314
```

## What It Does

- serves a Codex-style chat UI at `/`
- serves a runtime config panel at `/config`
- also exposes the same OpenAI-compatible endpoints at `/v1/*`
- stores local chat sessions in browser `localStorage`
- streams assistant responses through `chat.completions`

## Configuration

Same LLM environment variables as the Open WebUI gateway apply:

```bash
LLM_OPS_BASE_URL=http://127.0.0.1:8000/v1
LLM_OPS_MODEL=your-model
bash scripts/start_codex_ui.sh
```

Optional:

```bash
OPENFABRIC_CODEX_UI_HOST=0.0.0.0
OPENFABRIC_CODEX_UI_PORT=8315
ENABLE_CONTEXT=1
```

## Notes

- The API remains Open WebUI-compatible, so you can switch frontends without
  changing the backend contract.
- The UI is a close Codex-style match built locally in this repo, not a bundled
  upstream Codex client.
