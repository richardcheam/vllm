#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="richard-base-dev"
PROFILE="superinfer"
PORT="18743"
GPU_MEMORY_UTILIZATION="0.94"
MAX_MODEL_LEN="2048"
MAX_NUM_SEQS="128"
MAX_NUM_BATCHED_TOKENS="16384"
NUM_SPECULATIVE_TOKENS="3"
SWAP_CPU_MEMORY_GB="4"
PROACTIVE_SWAP_BUDGET="4"
VLT_BETA_BANDWIDTH="1"
SUPERINFER_HIGH_RISK_MODE="1"
GH200_TOPOLOGY_TUNED="0"
LOCAL_CPU_POOL_FRACTION="0.75"
LOCAL_SWAP_BANDWIDTH_BYTES_PER_S="966367641600"
REMOTE_SWAP_BANDWIDTH_BYTES_PER_S="300647710720"
LOG_NAME="direct_superinfer_perf_18743.log"
BLOCK_SIZE="16"
DISTRIBUTED_EXECUTOR_BACKEND=""
GENERATION_CONFIG=""
MODEL_PATH=""
SERVED_MODEL_NAME="deepseek-ai/DeepSeek-V4-Flash-0731"
ENFORCE_EAGER="0"

usage() {
  cat <<'EOF'
Usage: launch_direct_deepseek_superinfer.sh [options]

Profiles:
  baseline    No SuperInfer offload flags.
  swap-only   CPU offload enabled, proactive/VLT disabled.
  superinfer  CPU offload + proactive/VLT enabled.

Options:
  --container NAME
  --profile baseline|swap-only|superinfer
  --port PORT
  --gpu-memory-utilization VALUE
  --max-model-len VALUE
  --max-num-seqs VALUE
  --max-num-batched-tokens VALUE
  --num-speculative-tokens VALUE
  --proactive-swap-budget VALUE
  --vlt-beta-bandwidth VALUE
  --superinfer-high-risk-mode 0|1
  --gh200-topology-tuned 0|1
  --local-cpu-pool-fraction VALUE
  --local-swap-bandwidth-bytes-per-s VALUE
  --remote-swap-bandwidth-bytes-per-s VALUE
  --block-size VALUE
  --distributed-executor-backend VALUE
  --generation-config VALUE
  --model-path PATH
  --served-model-name NAME
  --enforce-eager 0|1
  --log-name NAME
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --container)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --gpu-memory-utilization)
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --max-num-seqs)
      MAX_NUM_SEQS="$2"
      shift 2
      ;;
    --max-num-batched-tokens)
      MAX_NUM_BATCHED_TOKENS="$2"
      shift 2
      ;;
    --num-speculative-tokens)
      NUM_SPECULATIVE_TOKENS="$2"
      shift 2
      ;;
    --proactive-swap-budget)
      PROACTIVE_SWAP_BUDGET="$2"
      shift 2
      ;;
    --vlt-beta-bandwidth)
      VLT_BETA_BANDWIDTH="$2"
      shift 2
      ;;
    --superinfer-high-risk-mode)
      SUPERINFER_HIGH_RISK_MODE="$2"
      shift 2
      ;;
    --gh200-topology-tuned)
      GH200_TOPOLOGY_TUNED="$2"
      shift 2
      ;;
    --local-cpu-pool-fraction)
      LOCAL_CPU_POOL_FRACTION="$2"
      shift 2
      ;;
    --local-swap-bandwidth-bytes-per-s)
      LOCAL_SWAP_BANDWIDTH_BYTES_PER_S="$2"
      shift 2
      ;;
    --remote-swap-bandwidth-bytes-per-s)
      REMOTE_SWAP_BANDWIDTH_BYTES_PER_S="$2"
      shift 2
      ;;
    --block-size)
      BLOCK_SIZE="$2"
      shift 2
      ;;
    --distributed-executor-backend)
      DISTRIBUTED_EXECUTOR_BACKEND="$2"
      shift 2
      ;;
    --generation-config)
      GENERATION_CONFIG="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --served-model-name)
      SERVED_MODEL_NAME="$2"
      shift 2
      ;;
    --enforce-eager)
      ENFORCE_EAGER="$2"
      shift 2
      ;;
    --log-name)
      LOG_NAME="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "${PROFILE}" in
  baseline|swap-only|superinfer)
    ;;
  *)
    echo "Invalid profile: ${PROFILE}" >&2
    usage
    exit 1
    ;;
esac

WORKDIR="/workspace/re-SuperInfer/vllm-modern"

docker exec -d \
  -e PROFILE="${PROFILE}" \
  -e PORT="${PORT}" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
  -e NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS}" \
  -e BLOCK_SIZE="${BLOCK_SIZE}" \
  -e DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND}" \
  -e GENERATION_CONFIG="${GENERATION_CONFIG}" \
  -e MODEL_PATH="${MODEL_PATH}" \
  -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
  -e ENFORCE_EAGER="${ENFORCE_EAGER}" \
  -e SWAP_CPU_MEMORY_GB="${SWAP_CPU_MEMORY_GB}" \
  -e PROACTIVE_SWAP_BUDGET="${PROACTIVE_SWAP_BUDGET}" \
  -e VLT_BETA_BANDWIDTH="${VLT_BETA_BANDWIDTH}" \
  -e SUPERINFER_HIGH_RISK_MODE="${SUPERINFER_HIGH_RISK_MODE}" \
  -e GH200_TOPOLOGY_TUNED="${GH200_TOPOLOGY_TUNED}" \
  -e LOCAL_CPU_POOL_FRACTION="${LOCAL_CPU_POOL_FRACTION}" \
  -e LOCAL_SWAP_BANDWIDTH_BYTES_PER_S="${LOCAL_SWAP_BANDWIDTH_BYTES_PER_S}" \
  -e REMOTE_SWAP_BANDWIDTH_BYTES_PER_S="${REMOTE_SWAP_BANDWIDTH_BYTES_PER_S}" \
  -e LOG_NAME="${LOG_NAME}" \
  -w "${WORKDIR}" \
  "${CONTAINER_NAME}" \
  bash scripts/run_direct_deepseek_server_inside_container.sh

echo "Launched ${PROFILE} on port ${PORT}; log=${WORKDIR}/${LOG_NAME}"
