#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate

echo "Starting planner on :8001"
uvicorn examples.samples.planner.app:app --host 127.0.0.1 --port 8001 &
PLANNER_PID=$!

echo "Starting retriever on :8002"
uvicorn examples.samples.retriever.app:app --host 127.0.0.1 --port 8002 &
RETRIEVER_PID=$!

echo "Starting synthesizer on :8003"
uvicorn examples.samples.synthesizer.app:app --host 127.0.0.1 --port 8003 &
SYNTHESIZER_PID=$!

cleanup() {
  echo
  echo "Stopping sample HTTP agents..."
  kill "$PLANNER_PID" "$RETRIEVER_PID" "$SYNTHESIZER_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

echo "All sample HTTP agents are running. Press Ctrl+C to stop."
wait
