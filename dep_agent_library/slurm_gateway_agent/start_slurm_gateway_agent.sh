#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
HOST="${SLURM_GATEWAY_BIND_HOST:-0.0.0.0}"
PORT="${SLURM_GATEWAY_BIND_PORT:-8312}"

if [[ ! -x "${VENV_DIR}/bin/uvicorn" ]]; then
  echo "Missing virtualenv at ${VENV_DIR}. Run install_slurm_gateway_agent.sh first." >&2
  exit 1
fi

source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${SCRIPT_DIR}/../..:${PYTHONPATH:-}"

exec uvicorn dep_agent_library.slurm_gateway_agent.app:app --host "${HOST}" --port "${PORT}"
