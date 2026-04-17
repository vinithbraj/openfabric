#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LLM_OPS_MODEL="${LLM_OPS_MODEL:-deepseek-ai/deepseek-coder-6.7b-instruct}"
export LLM_OPS_SYNTH_MODEL="${LLM_OPS_SYNTH_MODEL:-$LLM_OPS_MODEL}"

exec "$SCRIPT_DIR/start_openwebui_gateway.sh"
