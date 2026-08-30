#!/usr/bin/env bash
set -euo pipefail

# Run inside richard-base-dev-sysnice after the target environment is ready.
# This is the vanilla vLLM 0.26.0 DSpark baseline. SuperInfer flags are omitted.

cd /workspace/re-SuperInfer/vllm-superinfer-v4
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export VLLM_TARGET_DEVICE=cuda

exec .venv/bin/python -m vllm.entrypoints.cli.main serve \
  /workspace/models/DeepSeek-V4-Flash-0731 \
  --host 0.0.0.0 \
  --port "${VLLM_PORT:-8202}" \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash-0731 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --enable-prefix-caching \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}" \
  --max-model-len 1048576 \
  --max-num-seqs "${MAX_NUM_SEQS:-32}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}" \
  --block-size "${BLOCK_SIZE:-256}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}" \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
