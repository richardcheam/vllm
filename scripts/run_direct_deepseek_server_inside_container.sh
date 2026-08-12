#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PROFILE:-superinfer}"
PORT="${PORT:-18743}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-}"
GENERATION_CONFIG="${GENERATION_CONFIG:-}"
SWAP_CPU_MEMORY_GB="${SWAP_CPU_MEMORY_GB:-4}"
PROACTIVE_SWAP_BUDGET="${PROACTIVE_SWAP_BUDGET:-4}"
VLT_BETA_BANDWIDTH="${VLT_BETA_BANDWIDTH:-1}"
SUPERINFER_HIGH_RISK_MODE="${SUPERINFER_HIGH_RISK_MODE:-0}"
GH200_TOPOLOGY_TUNED="${GH200_TOPOLOGY_TUNED:-0}"
LOCAL_CPU_POOL_FRACTION="${LOCAL_CPU_POOL_FRACTION:-0.75}"
LOCAL_SWAP_BANDWIDTH_BYTES_PER_S="${LOCAL_SWAP_BANDWIDTH_BYTES_PER_S:-966367641600}"
REMOTE_SWAP_BANDWIDTH_BYTES_PER_S="${REMOTE_SWAP_BANDWIDTH_BYTES_PER_S:-300647710720}"
LOG_NAME="${LOG_NAME:-direct_superinfer_perf_18743.log}"
#MODEL_PATH="${MODEL_PATH:-/workspace/re-SuperInfer/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136}"
MODEL_PATH="${MODEL_PATH:-/workspace/re-SuperInfer/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-ai/DeepSeek-V4-Flash}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

map_profile_label() {
  case "$1" in
    baseline)
      printf '%s' "vanilla_baseline"
      ;;
    swap-only)
      if [[ "${GH200_TOPOLOGY_TUNED}" == "1" || "${GH200_TOPOLOGY_TUNED}" == "true" || "${GH200_TOPOLOGY_TUNED}" == "True" ]]; then
        printf '%s' "gh200_topology_tuned"
      else
        printf '%s' "superinfer_swap_enabled"
      fi
      ;;
    superinfer)
      if [[ "${GH200_TOPOLOGY_TUNED}" == "1" || "${GH200_TOPOLOGY_TUNED}" == "true" || "${GH200_TOPOLOGY_TUNED}" == "True" ]]; then
        printf '%s' "superinfer_swap_topology_aware"
      else
        printf '%s' "superinfer_swap_enabled"
      fi
      ;;
    *)
      printf '%s' "unknown"
      ;;
  esac
}

case "${PROFILE}" in
  baseline|swap-only|superinfer)
    ;;
  *)
    echo "Invalid PROFILE=${PROFILE}" >&2
    exit 1
    ;;
esac

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/workspace/re-SuperInfer
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_LOG_STATS_INTERVAL=1
export VLLM_RPC_TIMEOUT=600000
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

args=(
  "${MODEL_PATH}"
  --trust-remote-code
  --host 0.0.0.0
  --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size 2
  --enable-expert-parallel
  --disable-custom-all-reduce
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --block-size "${BLOCK_SIZE}"
  --tokenizer-mode deepseek_v4
  --reasoning-parser deepseek_v4
  --tool-call-parser deepseek_v4
  --enable-auto-tool-choice
  --kv-cache-dtype fp8
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
  --attention-config '{"use_fp4_indexer_cache":false}'
  --max-cudagraph-capture-size 128
  --enable-prefix-caching
  --download-dir /workspace/re-SuperInfer
)

if [[ "${NUM_SPECULATIVE_TOKENS}" =~ ^[0-9]+$ ]] && (( NUM_SPECULATIVE_TOKENS > 0 )); then
  args+=(
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS}}"
  )
fi

if [[ "${ENFORCE_EAGER}" == "1" || "${ENFORCE_EAGER}" == "true" || "${ENFORCE_EAGER}" == "True" ]]; then
  args+=(
    --enforce-eager
  )
fi

if [[ -n "${DISTRIBUTED_EXECUTOR_BACKEND}" ]]; then
  args+=(
    --distributed-executor-backend "${DISTRIBUTED_EXECUTOR_BACKEND}"
  )
fi

if [[ -n "${GENERATION_CONFIG}" ]]; then
  args+=(
    --generation-config "${GENERATION_CONFIG}"
  )
fi

if [[ "${PROFILE}" == "swap-only" || "${PROFILE}" == "superinfer" ]]; then
  args+=(
    --swap-cpu-memory-gb "${SWAP_CPU_MEMORY_GB}"
    --pin-memory-fix
    --swapper-block-first
  )
fi

if [[ "${PROFILE}" == "superinfer" ]]; then
  args+=(
    --proactive-swap-budget "${PROACTIVE_SWAP_BUDGET}"
    --vlt-beta-bandwidth "${VLT_BETA_BANDWIDTH}"
  )
  if [[ "${SUPERINFER_HIGH_RISK_MODE}" == "1" || "${SUPERINFER_HIGH_RISK_MODE}" == "true" || "${SUPERINFER_HIGH_RISK_MODE}" == "True" ]]; then
    args+=(
      --superinfer-high-risk-mode
    )
  fi
fi

if [[ ("${PROFILE}" == "swap-only" || "${PROFILE}" == "superinfer") && ("${GH200_TOPOLOGY_TUNED}" == "1" || "${GH200_TOPOLOGY_TUNED}" == "true" || "${GH200_TOPOLOGY_TUNED}" == "True") ]]; then
  args+=(
    --gh200-topology-tuned
    --local-cpu-pool-fraction "${LOCAL_CPU_POOL_FRACTION}"
    --local-swap-bandwidth-bytes-per-s "${LOCAL_SWAP_BANDWIDTH_BYTES_PER_S}"
    --remote-swap-bandwidth-bytes-per-s "${REMOTE_SWAP_BANDWIDTH_BYTES_PER_S}"
  )
fi

PROFILE_LABEL="$(map_profile_label "${PROFILE}")"
{
  echo "[profile] ${PROFILE_LABEL}"
  echo "[effective] NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-<unset>}"
  echo "[effective] disable_custom_all_reduce=1"
  echo "[effective] gh200_topology_tuned=${GH200_TOPOLOGY_TUNED}"
  echo "[effective] swap_cpu_memory_gb=${SWAP_CPU_MEMORY_GB}"
  echo "[effective] proactive_swap_budget=${PROACTIVE_SWAP_BUDGET}"
  echo "[effective] superinfer_high_risk_mode=${SUPERINFER_HIGH_RISK_MODE}"
  echo "[effective] local_cpu_pool_fraction=${LOCAL_CPU_POOL_FRACTION}"
  echo "[effective] local_swap_bandwidth_bytes_per_s=${LOCAL_SWAP_BANDWIDTH_BYTES_PER_S}"
  echo "[effective] remote_swap_bandwidth_bytes_per_s=${REMOTE_SWAP_BANDWIDTH_BYTES_PER_S}"
  echo "[effective] model_path=${MODEL_PATH}"
  echo "[effective] served_model_name=${SERVED_MODEL_NAME}"
  echo "[effective] enforce_eager=${ENFORCE_EAGER}"
} > "${LOG_NAME}.profile"

printf '%q ' .venv/bin/vllm serve "${args[@]}" > "${LOG_NAME}.cmd"
printf '\n' >> "${LOG_NAME}.cmd"
printf '%s\n' "# profile=${PROFILE_LABEL}" >> "${LOG_NAME}.cmd"

exec nohup .venv/bin/vllm serve "${args[@]}" > "${LOG_NAME}" 2>&1 &
