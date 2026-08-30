#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file) ENV_FILE="$2"; shift 2 ;;
    -h|--help)
      printf 'Usage: %s [ENV_FILE] [--env-file PATH]\n' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    --*) printf 'Unknown argument: %s\n' "$1" >&2; exit 1 ;;
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
  printf 'Missing %s; skipping production warmup.\n' "${ENV_FILE}" >&2
  exit 0
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/..:${PYTHONPATH:-}"

if [[ "${LAUNCH_WARMUP_ENABLED:-0}" != "1" ]]; then
  printf 'Production warmup disabled.\n'
  exit 0
fi

PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
BENCH_URL="${BENCH_URL:-http://127.0.0.1:${VLLM_PORT:-8202}}"
MODEL="${SERVED_MODEL_NAME:?SERVED_MODEL_NAME is required}"
TIMEOUT_S="${LAUNCH_WARMUP_TIMEOUT_S:-300}"
OUTPUT_ROOT="${LAUNCH_WARMUP_OUTPUT_DIR:-${ROOT_DIR}/benchmark_artifacts/launch_warmup}"
RUN_DIR="${OUTPUT_ROOT}/$(date +%Y%m%d_%H%M%S)_${PROFILE:-superinfer}"
SERVER_LOG="${LOG_DIR:-${ROOT_DIR}/logs}/server_${PROFILE:-superinfer}_${VLLM_PORT:-8202}.log"
mkdir -p "${RUN_DIR}"

capture_snapshot() {
  local base="$1"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader >"${base}.gpu.csv" 2>/dev/null || true
  curl --noproxy '*' -fsSL "${BENCH_URL}/metrics" \
    >"${base}.metrics.txt" 2>/dev/null || : >"${base}.metrics.txt"
}

run_stage() {
  local name="$1"
  shift
  local output="${RUN_DIR}/${name}.json"
  local status=0
  capture_snapshot "${RUN_DIR}/${name}.before"
  printf 'Production warmup stage=%s\n' "${name}"
  timeout --signal=TERM --kill-after=10s "${TIMEOUT_S}s" \
    "${PYTHON}" "$@" --output-json "${output}" || status=$?
  capture_snapshot "${RUN_DIR}/${name}.after"
  if [[ "${status}" -ne 0 ]]; then
    printf 'Warmup stage %s failed (status=%s); continuing.\n' "${name}" "${status}" >&2
    printf '{"stage":"%s","status":%s}\n' "${name}" "${status}" \
      >"${RUN_DIR}/${name}.status.json"
  fi
}

if ! curl --noproxy '*' -fsSL "${BENCH_URL}/health" >/dev/null 2>&1; then
  printf 'Server health check failed; skipping production warmup.\n' >&2
  exit 0
fi

ROBUST_CLIENT="${ROOT_DIR}/../test_concurrent_robust.py"
BASIC_CLIENT="${ROOT_DIR}/../test_concurrent.py"
if [[ ! -f "${ROBUST_CLIENT}" || ! -f "${BASIC_CLIENT}" ]]; then
  printf 'Warmup clients unavailable; skipping production warmup.\n' >&2
  exit 0
fi

run_stage robust_canary "${ROBUST_CLIENT}" \
  --url "${BENCH_URL}" --users "${LAUNCH_WARMUP_ROBUST_USERS:-1}" \
  --requests "${LAUNCH_WARMUP_ROBUST_REQUESTS:-2}" \
  --prompt-len "${LAUNCH_WARMUP_ROBUST_PROMPT_LEN:-4096}" \
  --max-tokens "${LAUNCH_WARMUP_ROBUST_MAX_TOKENS:-64}" \
  --model "${MODEL}" --seed "${LAUNCH_WARMUP_SEED:-20260812}" \
  --run-label launch-robust-canary

run_stage basic_throughput "${BASIC_CLIENT}" \
  --url "${BENCH_URL}" --users "${LAUNCH_WARMUP_BASIC_USERS:-4}" \
  --requests "${LAUNCH_WARMUP_BASIC_REQUESTS:-8}" \
  --prompt-len "${LAUNCH_WARMUP_BASIC_PROMPT_LEN:-4096}" \
  --max-tokens "${LAUNCH_WARMUP_BASIC_MAX_TOKENS:-64}" \
  --model "${MODEL}"

[[ -f "${SERVER_LOG}" ]] && cp "${SERVER_LOG}" "${RUN_DIR}/server.log" 2>/dev/null || true
printf 'Production warmup complete: %s\n' "${RUN_DIR}"
exit 0
