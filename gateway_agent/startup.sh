#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export GATEWAY_NODE_NAME="${GATEWAY_NODE_NAME:-localhost}"
export GATEWAY_BIND_HOST="${GATEWAY_BIND_HOST:-127.0.0.1}"
export GATEWAY_BIND_PORT="${GATEWAY_BIND_PORT:-8787}"
export GATEWAY_EXEC_TIMEOUT_SECONDS="${GATEWAY_EXEC_TIMEOUT_SECONDS:-30}"

cd "${REPO_ROOT}"

exec python -m uvicorn gateway_agent.app:app \
  --host "${GATEWAY_BIND_HOST}" \
  --port "${GATEWAY_BIND_PORT}"
