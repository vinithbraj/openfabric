#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export LLM_OPS_API_KEY="${LLM_OPS_API_KEY:-dummy}"
export LLM_OPS_BASE_URL="${LLM_OPS_BASE_URL:-http://127.0.0.1:8000/v1}"
if [[ -n "${LLM_OPS_MODEL:-}" ]]; then
  export LLM_OPS_MODEL
fi
if [[ -n "${LLM_OPS_SYNTH_MODEL:-}" ]]; then
  export LLM_OPS_SYNTH_MODEL
fi

HOST="${OPENFABRIC_CODEX_UI_HOST:-0.0.0.0}"
PORT="${OPENFABRIC_CODEX_UI_PORT:-8314}"
TIMEOUT="${OPENFABRIC_GATEWAY_TIMEOUT:-300}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

exec "$PYTHON_BIN" codex_ui_gateway.py \
  --host "$HOST" \
  --port "$PORT" \
  --timeout "$TIMEOUT" \
  "$@"
