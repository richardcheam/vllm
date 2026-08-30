#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env}"
DRY_RUN=0
NSYS_OVERRIDE=""
NSYS_OUTPUT_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --nsys)
      NSYS_OVERRIDE=1
      shift
      ;;
    --no-nsys)
      NSYS_OVERRIDE=0
      shift
      ;;
    --nsys-output-dir)
      NSYS_OUTPUT_OVERRIDE="$2"
      shift 2
      ;;
    -h|--help)
      printf 'Usage: %s [ENV_FILE] [--env-file PATH] [--dry-run] [--nsys]\n' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    --*)
      printf 'Unknown argument: %s\n' "$1" >&2
      exit 1
      ;;
    *)
      if [[ "${ENV_FILE}" != "${ROOT_DIR}/.env" ]]; then
        printf 'Multiple environment files specified.\n' >&2
        exit 1
      fi
      ENV_FILE="$1"
      shift
      ;;
  esac
done

if [[ ! -f "${ENV_FILE}" ]]; then
  printf 'Missing %s. Copy .env.example to .env first.\n' "${ENV_FILE}" >&2
  exit 1
fi
ENV_FILE="$(realpath "${ENV_FILE}")"

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

if [[ -n "${NSYS_OVERRIDE}" ]]; then
  NSYS_PROFILE="${NSYS_OVERRIDE}"
fi
if [[ -n "${NSYS_OUTPUT_OVERRIDE}" ]]; then
  NSYS_OUTPUT_DIR="${NSYS_OUTPUT_OVERRIDE}"
fi

MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:?SERVED_MODEL_NAME is required}"
VLLM_PORT="${VLLM_PORT:-8202}"
PROFILE="${PROFILE:-superinfer}"
STATE_DIR="${STATE_DIR:-${ROOT_DIR}/.run}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-3600}"
NSYS_PROFILE="${NSYS_PROFILE:-0}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt}"
NSYS_OUTPUT_DIR="${NSYS_OUTPUT_DIR:-${ROOT_DIR}/benchmark_artifacts/nsys}"

case "${PROFILE}" in
  vanilla|native-offload|superinfer|superinfer-high-risk) ;;
  *) printf 'Unsupported PROFILE=%s\n' "${PROFILE}" >&2; exit 1 ;;
esac

mkdir -p "${STATE_DIR}" "${LOG_DIR}"
PID_FILE="${STATE_DIR}/server.pid"
PGID_FILE="${STATE_DIR}/server.pgid"
PROFILE_FILE="${STATE_DIR}/server.profile"
LOG_FILE="${LOG_DIR}/server_${PROFILE}_${VLLM_PORT}.log"

if [[ -s "${PID_FILE}" ]]; then
  old_pid="$(<"${PID_FILE}")"
  if kill -0 "${old_pid}" 2>/dev/null; then
    if [[ ! -f "${PROFILE_FILE}" || "$(<"${PROFILE_FILE}")" != "${ENV_FILE}" ]]; then
      printf 'A server is already running with a different or unknown profile. '
      printf 'Stop it before launching %s.\n' "${ENV_FILE}" >&2
      exit 1
    fi
    printf 'Server already running with profile=%s: pid=%s log=%s\n' \
      "${PROFILE}" "${old_pid}" "${LOG_FILE}"
    exit 0
  fi
  rm -f "${PID_FILE}" "${PGID_FILE}" "${PROFILE_FILE}"
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  printf 'Model config not found: %s/config.json\n' "${MODEL_PATH}" >&2
  exit 1
fi

PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  printf 'Python executable not found: %s\n' "${PYTHON}" >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

cmd=(
  "${PYTHON}" -m vllm.entrypoints.cli.main serve "${MODEL_PATH}"
  --host "${VLLM_HOST:-0.0.0.0}"
  --port "${VLLM_PORT}" 
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --tensor-parallel-size "${TP_SIZE:-2}"
  --pipeline-parallel-size "${PP_SIZE:-1}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}"
  --max-model-len "${MAX_MODEL_LEN:-1048576}"
  --max-num-seqs "${MAX_NUM_SEQS:-32}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}"
  --block-size "${BLOCK_SIZE:-256}"
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}"
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS:-5}}"
)

if [[ "${NUMA_BIND:-0}" == "1" ]]; then
  cmd+=(--numa-bind)
  if [[ -n "${NUMA_BIND_NODES:-}" ]]; then
    IFS=',' read -r -a numa_nodes <<< "${NUMA_BIND_NODES}"
    cmd+=(--numa-bind-nodes "${numa_nodes[@]}")
  fi
  if [[ -n "${NUMA_BIND_CPUS:-}" ]]; then
    IFS=';' read -r -a numa_cpus <<< "${NUMA_BIND_CPUS}"
    cmd+=(--numa-bind-cpus "${numa_cpus[@]}")
  fi
fi

if [[ "${ENABLE_AUTO_TOOL_CHOICE:-0}" == "1" ]]; then
  if [[ -z "${TOOL_CALL_PARSER:-}" ]]; then
    printf 'ENABLE_AUTO_TOOL_CHOICE=1 requires TOOL_CALL_PARSER.\n' >&2
    exit 1
  fi
  cmd+=(
    --enable-auto-tool-choice
    --tool-call-parser "${TOOL_CALL_PARSER}"
  )
fi

if [[ -n "${REASONING_PARSER:-}" ]]; then
  cmd+=(--reasoning-parser "${REASONING_PARSER}")
fi

if [[ "${ENABLE_PREFIX_CACHING:-1}" == "1" ]]; then
  cmd+=(--enable-prefix-caching)
else
  cmd+=(--no-enable-prefix-caching)
fi

if [[ "${PROFILE}" == "superinfer" || "${PROFILE}" == "superinfer-high-risk" ]]; then
  cmd+=(
    --swap-cpu-memory-gb "${SWAP_CPU_MEMORY_GB:-64}"
    --proactive-swap-budget "${PROACTIVE_SWAP_BUDGET:-2400}"
    --vlt-alpha "${VLT_ALPHA:-3}"
    --vlt-beta-bandwidth "${VLT_BETA_BANDWIDTH:-1}"
    --vlt-beta-future "${VLT_BETA_FUTURE:-1}"
    --slo-ttft "${SLO_TTFT:-5}"
    --slo-tbt "${SLO_TBT:-0.1}"
    --pin-memory-fix
    --gh200-topology-tuned
    --local-cpu-pool-fraction "${LOCAL_CPU_POOL_FRACTION:-0.75}"
    --local-swap-bandwidth-bytes-per-s "${LOCAL_SWAP_BANDWIDTH_BYTES_PER_S:-966367641600}"
    --remote-swap-bandwidth-bytes-per-s "${REMOTE_SWAP_BANDWIDTH_BYTES_PER_S:-300647710720}"
    --cpu-kv-allocation-mode "${CPU_KV_ALLOCATION_MODE:-zero}"
  )
  [[ "${NATIVE_COPY_BACKEND:-0}" == "1" ]] && cmd+=(--native-copy-backend)
  [[ "${SWAPPER_BLOCK_FIRST:-0}" == "1" ]] && cmd+=(--swapper-block-first)
  [[ "${SUPERINFER_HIGH_RISK_MODE:-0}" == "1" ]] && cmd+=(--superinfer-high-risk-mode)
elif [[ "${PROFILE}" == "native-offload" ]]; then
  cmd+=(
    --kv-offloading-size "${KV_OFFLOADING_SIZE:-8}"
    --kv-offloading-backend native
  )
fi

if [[ "${NSYS_PROFILE}" == "1" ]]; then
  if ! command -v nsys >/dev/null 2>&1; then
    printf 'nsys is required for --nsys but was not found.\n' >&2
    exit 1
  fi
  mkdir -p "${NSYS_OUTPUT_DIR}"
  cmd=(
    nsys profile
    --force-overwrite=true
    --trace="${NSYS_TRACE}"
    --sample=none
    --wait=primary
    --export=sqlite
    --output="${NSYS_OUTPUT_DIR}/server"
    "${cmd[@]}"
  )
fi

printf 'Starting profile=%s port=%s\n' "${PROFILE}" "${VLLM_PORT}"
printf 'Model: %s\nLog: %s\n' "${MODEL_PATH}" "${LOG_FILE}"
printf 'Command:'; printf ' %q' "${cmd[@]}"; printf '\n'

if (( DRY_RUN )); then
  printf 'Dry run only; server was not started.\n'
  exit 0
fi

setsid "${cmd[@]}" >"${LOG_FILE}" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" >"${PID_FILE}"
printf '%s\n' "${pid}" >"${PGID_FILE}"
printf '%s\n' "${ENV_FILE}" >"${PROFILE_FILE}"

cleanup_start_failure() {
  "${ROOT_DIR}/scripts/stop_superinfer.sh" --env-file "${ENV_FILE}" >/dev/null 2>&1 || true
}
trap cleanup_start_failure ERR

deadline=$((SECONDS + STARTUP_TIMEOUT_S))
while (( SECONDS < deadline )); do
  if curl --noproxy '*' -fsSL "http://127.0.0.1:${VLLM_PORT}/health" \
    >/dev/null 2>&1; then
    trap - ERR
    printf 'READY pid=%s url=http://127.0.0.1:%s log=%s\n' "${pid}" "${VLLM_PORT}" "${LOG_FILE}"
    if [[ "${SKIP_LAUNCH_WARMUP:-0}" != "1" ]]; then
      "${ROOT_DIR}/scripts/warmup_superinfer.sh" --env-file "${ENV_FILE}" || true
    else
      printf 'Production warmup skipped by caller.\n'
    fi
    exit 0
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    printf 'Server exited during startup. See %s\n' "${LOG_FILE}" >&2
    exit 1
  fi
  sleep 2
done

printf 'Startup timeout. Stopping process group. See %s\n' "${LOG_FILE}" >&2
exit 1
