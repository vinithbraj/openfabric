#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
uvicorn aor_runtime.api.app:create_app --factory --host "${AOR_HOST:-127.0.0.1}" --port "${AOR_PORT:-8011}" --reload
