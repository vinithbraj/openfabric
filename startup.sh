#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

CONFIG_OVERRIDE=""
HOST_OVERRIDE=""
PORT_OVERRIDE=""
RELOAD_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --config" >&2
        echo "Usage: $0 [--config PATH] [--host HOST] [--port PORT] [--reload]" >&2
        exit 1
      fi
      CONFIG_OVERRIDE="$2"
      shift 2
      ;;
    --host)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --host" >&2
        echo "Usage: $0 [--config PATH] [--host HOST] [--port PORT] [--reload]" >&2
        exit 1
      fi
      HOST_OVERRIDE="$2"
      shift 2
      ;;
    --port)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --port" >&2
        echo "Usage: $0 [--config PATH] [--host HOST] [--port PORT] [--reload]" >&2
        exit 1
      fi
      PORT_OVERRIDE="$2"
      shift 2
      ;;
    --reload)
      RELOAD_OVERRIDE="1"
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--config PATH] [--host HOST] [--port PORT] [--reload]" >&2
      exit 1
      ;;
  esac
done

cd "${REPO_ROOT}"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv in ${REPO_ROOT}." >&2
  echo "Run: ./install.sh" >&2
  exit 1
fi

source .venv/bin/activate

# Launcher-level settings.
# Default to 0.0.0.0 so local Dockerized clients such as Open WebUI can reach
# the server without extra overrides. Set AOR_HOST=127.0.0.1 if you want the
# API bound to loopback only.
export AOR_HOST="${HOST_OVERRIDE:-${AOR_HOST:-0.0.0.0}}"
export AOR_PORT="${PORT_OVERRIDE:-${AOR_PORT:-8011}}"
export AOR_RELOAD="${RELOAD_OVERRIDE:-${AOR_RELOAD:-0}}"

# Config loading. Prefer config_ucla.yaml when present, then config.yaml, unless
# the caller already supplied AOR_APP_CONFIG_PATH or --config explicitly.
if [[ -n "${CONFIG_OVERRIDE}" ]]; then
  export AOR_APP_CONFIG_PATH="${CONFIG_OVERRIDE}"
elif [[ -n "${AOR_APP_CONFIG_PATH:-}" ]]; then
  export AOR_APP_CONFIG_PATH="${AOR_APP_CONFIG_PATH}"
elif [[ -f "${REPO_ROOT}/config_ucla.yaml" ]]; then
  export AOR_APP_CONFIG_PATH="${REPO_ROOT}/config_ucla.yaml"
elif [[ -f "${REPO_ROOT}/config.yaml" ]]; then
  export AOR_APP_CONFIG_PATH="${REPO_ROOT}/config.yaml"
else
  export AOR_APP_CONFIG_PATH=""
fi

# Gateway and node routing.
export AOR_AVAILABLE_NODES="${AOR_AVAILABLE_NODES:-localhost}"
export AOR_DEFAULT_NODE="${AOR_DEFAULT_NODE:-localhost}"
export AOR_GATEWAY_URL="${AOR_GATEWAY_URL:-http://127.0.0.1:8787}"
export AOR_GATEWAY_TIMEOUT_SECONDS="${AOR_GATEWAY_TIMEOUT_SECONDS:-30}"

# Shell and execution controls.
export AOR_SHELL_MODE="${AOR_SHELL_MODE:-read_only}"
export AOR_SHELL_ALLOW_MUTATION_WITH_APPROVAL="${AOR_SHELL_ALLOW_MUTATION_WITH_APPROVAL:-0}"
export AOR_SHELL_ALLOWED_ROOTS="${AOR_SHELL_ALLOWED_ROOTS:-}"
export AOR_SHELL_DEFAULT_CWD="${AOR_SHELL_DEFAULT_CWD:-}"
export AOR_SHELL_MAX_OUTPUT_CHARS="${AOR_SHELL_MAX_OUTPUT_CHARS:-20000}"
export AOR_SHELL_COMMAND_TIMEOUT_SECONDS="${AOR_SHELL_COMMAND_TIMEOUT_SECONDS:-30}"
export AOR_SHUTDOWN_GRACE_SECONDS="${AOR_SHUTDOWN_GRACE_SECONDS:-5}"
export AOR_WORKER_JOIN_TIMEOUT_SECONDS="${AOR_WORKER_JOIN_TIMEOUT_SECONDS:-2}"
export AOR_TOOL_PROCESS_KILL_GRACE_SECONDS="${AOR_TOOL_PROCESS_KILL_GRACE_SECONDS:-1}"
export AOR_RUNTIME_TIMEZONE="${AOR_RUNTIME_TIMEZONE:-}"

# Planning and LLM-assisted behavior.
export AOR_ENABLE_LLM_INTENT_EXTRACTION="${AOR_ENABLE_LLM_INTENT_EXTRACTION:-0}"
export AOR_ENABLE_SQL_LLM_GENERATION="${AOR_ENABLE_SQL_LLM_GENERATION:-0}"
export AOR_ACTION_PLANNER_ENABLED="${AOR_ACTION_PLANNER_ENABLED:-1}"
export AOR_LEGACY_EXECUTION_PLANNER_ENABLED="${AOR_LEGACY_EXECUTION_PLANNER_ENABLED:-0}"

# Presentation and trace settings.
export AOR_PRESENTATION_MODE="${AOR_PRESENTATION_MODE:-user}"
export AOR_RESPONSE_RENDER_MODE="${AOR_RESPONSE_RENDER_MODE:-user}"
export AOR_ENABLE_LLM_SUMMARY="${AOR_ENABLE_LLM_SUMMARY:-0}"
export AOR_LLM_SUMMARY_MAX_FACTS="${AOR_LLM_SUMMARY_MAX_FACTS:-50}"
export AOR_INCLUDE_INTERNAL_TELEMETRY="${AOR_INCLUDE_INTERNAL_TELEMETRY:-0}"
export AOR_SHOW_EXECUTED_COMMANDS="${AOR_SHOW_EXECUTED_COMMANDS:-1}"
export AOR_SHOW_VALIDATION_EVENTS="${AOR_SHOW_VALIDATION_EVENTS:-0}"
export AOR_SHOW_PLANNER_EVENTS="${AOR_SHOW_PLANNER_EVENTS:-0}"
export AOR_SHOW_TOOL_EVENTS="${AOR_SHOW_TOOL_EVENTS:-0}"
export AOR_OPENWEBUI_TRACE_MODE="${AOR_OPENWEBUI_TRACE_MODE:-}"
export AOR_SHOW_RESPONSE_STATS="${AOR_SHOW_RESPONSE_STATS:-1}"
export AOR_SHOW_PROMPT_SUGGESTIONS="${AOR_SHOW_PROMPT_SUGGESTIONS:-0}"
export AOR_SHOW_DEBUG_METADATA="${AOR_SHOW_DEBUG_METADATA:-0}"

# Presentation LLM summary settings.
export AOR_ENABLE_PRESENTATION_LLM_SUMMARY="${AOR_ENABLE_PRESENTATION_LLM_SUMMARY:-0}"
export AOR_PRESENTATION_LLM_MAX_FACTS="${AOR_PRESENTATION_LLM_MAX_FACTS:-50}"
export AOR_PRESENTATION_LLM_MAX_INPUT_CHARS="${AOR_PRESENTATION_LLM_MAX_INPUT_CHARS:-4000}"
export AOR_PRESENTATION_LLM_MAX_OUTPUT_CHARS="${AOR_PRESENTATION_LLM_MAX_OUTPUT_CHARS:-1500}"
export AOR_PRESENTATION_LLM_INCLUDE_ROW_SAMPLES="${AOR_PRESENTATION_LLM_INCLUDE_ROW_SAMPLES:-0}"
export AOR_PRESENTATION_LLM_INCLUDE_PATHS="${AOR_PRESENTATION_LLM_INCLUDE_PATHS:-0}"

# Intelligent output and semantic frame settings.
export AOR_INTELLIGENT_OUTPUT_MODE="${AOR_INTELLIGENT_OUTPUT_MODE:-off}"
export AOR_INTELLIGENT_OUTPUT_MAX_FIELDS="${AOR_INTELLIGENT_OUTPUT_MAX_FIELDS:-8}"
export AOR_SEMANTIC_FRAME_MODE="${AOR_SEMANTIC_FRAME_MODE:-enforce}"
export AOR_SEMANTIC_FRAME_MAX_DEPTH="${AOR_SEMANTIC_FRAME_MAX_DEPTH:-10}"
export AOR_SEMANTIC_FRAME_MAX_CHILDREN="${AOR_SEMANTIC_FRAME_MAX_CHILDREN:-8}"
export AOR_LLM_STAGE_MAX_DEPTH="${AOR_LLM_STAGE_MAX_DEPTH:-10}"
export AOR_PRESENTATION_INTENT_MAX_DEPTH="${AOR_PRESENTATION_INTENT_MAX_DEPTH:-10}"

# Insight layer settings.
export AOR_ENABLE_INSIGHT_LAYER="${AOR_ENABLE_INSIGHT_LAYER:-1}"
export AOR_ENABLE_LLM_INSIGHTS="${AOR_ENABLE_LLM_INSIGHTS:-0}"
export AOR_INSIGHT_MAX_FACTS="${AOR_INSIGHT_MAX_FACTS:-50}"
export AOR_INSIGHT_MAX_INPUT_CHARS="${AOR_INSIGHT_MAX_INPUT_CHARS:-4000}"
export AOR_INSIGHT_MAX_OUTPUT_CHARS="${AOR_INSIGHT_MAX_OUTPUT_CHARS:-1500}"

# Automatic artifact settings.
export AOR_AUTO_ARTIFACTS_ENABLED="${AOR_AUTO_ARTIFACTS_ENABLED:-1}"
export AOR_AUTO_ARTIFACT_ROW_THRESHOLD="${AOR_AUTO_ARTIFACT_ROW_THRESHOLD:-50}"
export AOR_AUTO_ARTIFACT_DIR="${AOR_AUTO_ARTIFACT_DIR:-outputs}"
export AOR_AUTO_ARTIFACT_FORMAT="${AOR_AUTO_ARTIFACT_FORMAT:-csv}"

UVICORN_ARGS=(
  --app-dir src
  agent_runtime.api.app:create_app
  --factory
  --host "${AOR_HOST}"
  --port "${AOR_PORT}"
)

case "${AOR_RELOAD,,}" in
  1|true|yes|on)
    UVICORN_ARGS+=(--reload)
    ;;
esac

exec python -m uvicorn "${UVICORN_ARGS[@]}"
