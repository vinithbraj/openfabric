#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LAUNCH_AFTER_INSTALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --launch)
      LAUNCH_AFTER_INSTALL=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--launch]" >&2
      exit 1
      ;;
  esac
done

cd "${REPO_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  "${PYTHON_BIN}" -m venv .venv
fi

source .venv/bin/activate

python -m pip install -U pip setuptools wheel
pip install -e .

chmod +x startup.sh src/gateway_agent/startup.sh

echo "Install complete."
echo "Virtual environment: ${REPO_ROOT}/.venv"
echo "Agent launcher: ${REPO_ROOT}/startup.sh"
echo "Gateway launcher: ${REPO_ROOT}/src/gateway_agent/startup.sh"

if [[ "${LAUNCH_AFTER_INSTALL}" == "1" ]]; then
  exec ./startup.sh
fi
