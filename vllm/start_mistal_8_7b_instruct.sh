#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/models/data}"

python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 2096 \
  --port 8000
