#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/models/data}"

# Experimental 4-bit community quantization of Qwen3-Coder-30B-A3B.
# If this OOMs or fails to load on a single 24 GB GPU, reduce
# --max-model-len first. Some Qwen3-Coder AWQ builds may also require
# a newer vLLM than the Qwen2.5 scripts.
python -m vllm.entrypoints.openai.api_server \
  --model stelterlab/Qwen3-Coder-30B-A3B-Instruct-AWQ \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --port 8000
