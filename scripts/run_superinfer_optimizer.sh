#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env.superinfer-reload"
NATIVE_COPY_BACKEND="${NATIVE_COPY_BACKEND:-0}"
MAX_ATTEMPTS=0
POLL_SECONDS=30
IDLE_UTIL_PCT=10
IDLE_SAMPLES=3
WAIT_FOR_GPU=1
RESTART=1
STATE_DIR="${ROOT_DIR}/.run"
OUTPUT_ROOT="${ROOT_DIR}/benchmark_artifacts/optimizer"

usage() {
  printf '%s\n' \
    "Usage: ${BASH_SOURCE[0]} [options]" \
    "  --env-file PATH       Runtime profile (default: .env.superinfer-reload)" \
    "  --max-attempts N      Stop after N attempts; 0 means continuous" \
    "  --poll-seconds N      GPU/source polling interval (default: 30)" \
    "  --idle-util-pct N     Max utilization per GPU before a run (default: 10)" \
    "  --idle-samples N      Consecutive idle samples (default: 3)" \
    "  --no-wait-gpu         Do not wait for GPU utilization to fall" \
    "  --no-restart          Use the existing server instead of restarting" \
    "  --output-root PATH    Optimizer artifacts directory" \
    "  -h, --help            Show this help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --idle-util-pct) IDLE_UTIL_PCT="$2"; shift 2 ;;
    --idle-samples) IDLE_SAMPLES="$2"; shift 2 ;;
    --no-wait-gpu) WAIT_FOR_GPU=0; shift ;;
    --no-restart) RESTART=0; shift ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "${ENV_FILE}" ]]; then
  printf 'Missing environment file: %s\n' "${ENV_FILE}" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a
NATIVE_COPY_BACKEND="${NATIVE_COPY_BACKEND:-0}"

PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
mkdir -p "${STATE_DIR}" "${OUTPUT_ROOT}"
STATE_FILE="${OUTPUT_ROOT}/optimizer.state"
LOG_FILE="${OUTPUT_ROOT}/optimizer.log"

fingerprint() {
  {
    sha256sum \
    "${ROOT_DIR}/vllm/v1/simple_kv_offload/manager.py" \
    "${ROOT_DIR}/vllm/v1/simple_kv_offload/worker.py" \
    "${ROOT_DIR}/vllm/v1/simple_kv_offload/copy_backend.py" \
    "${ROOT_DIR}/vllm/v1/simple_kv_offload/native_copy.py" \
    "${ROOT_DIR}/vllm/v1/core/kv_cache_coordinator.py" \
    "${ROOT_DIR}/tests/v1/simple_kv_offload/test_scheduler.py" \
    "${ROOT_DIR}/scripts/run_reload_validation.sh" \
    "${ENV_FILE}"
    printf 'native_copy_backend=%s\n' "${NATIVE_COPY_BACKEND}"
  } | sha256sum | cut -d' ' -f1
}

gpu_is_idle() {
  local values value
  values="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ -n "${values}" ]] || return 1
  while read -r value; do
    [[ -n "${value}" && "${value}" -le "${IDLE_UTIL_PCT}" ]] || return 1
  done <<<"${values}"
}

wait_for_gpu() {
  if (( ! WAIT_FOR_GPU )); then
    return
  fi
  local idle=0
  while (( idle < IDLE_SAMPLES )); do
    if gpu_is_idle; then
      idle=$((idle + 1))
      printf '%s GPU idle sample %d/%d\n' "$(date -Is)" "${idle}" "${IDLE_SAMPLES}" \
        | tee -a "${LOG_FILE}"
    else
      idle=0
      printf '%s GPU busy; waiting\n' "$(date -Is)" | tee -a "${LOG_FILE}"
    fi
    (( idle >= IDLE_SAMPLES )) || sleep "${POLL_SECONDS}"
  done
}

latest_run_dir() {
  local root="$1"
  "${PYTHON}" - "${root}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs = sorted(
    root.glob("*_reload"), key=lambda path: path.stat().st_mtime
) if root.exists() else []
print(runs[-1] if runs else "")
PY
}

analyze_run() {
  local run_dir="$1"
  "${PYTHON}" - "${run_dir}" <<'PY'
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
validation = json.loads((run_dir / "validation.json").read_text())
log_path = run_dir / "server.log"
log = log_path.read_text(errors="replace") if log_path.exists() else ""
store_maps = re.findall(
    r"cached_key_groups=(\{[^}]*\}) source_primary_groups=(\{[^}]*\})", log
)
result = {
    "run_dir": str(run_dir),
    "native_copy_backend": __import__("os").environ.get(
        "NATIVE_COPY_BACKEND", "0"
    ) == "1",
    "validation": validation,
    "store_completion_maps": store_maps,
}
print(json.dumps(result, sort_keys=True))
if validation.get("status") == "PASS":
    raise SystemExit(0)
raise SystemExit(1)
PY
}

attempt=0
last_fingerprint=""
if [[ -f "${STATE_FILE}" ]]; then
  last_fingerprint="$(<"${STATE_FILE}")"
fi
printf '%s optimizer started env=%s max_attempts=%s\n' "$(date -Is)" \
  "${ENV_FILE}" "${MAX_ATTEMPTS}" | tee -a "${LOG_FILE}"

while (( MAX_ATTEMPTS == 0 || attempt < MAX_ATTEMPTS )); do
  current_fingerprint="$(fingerprint)"
  if [[ "${current_fingerprint}" == "${last_fingerprint}" && -n "${last_fingerprint}" ]]; then
    printf '%s source unchanged; waiting before retry\n' "$(date -Is)" \
      | tee -a "${LOG_FILE}"
    sleep "${POLL_SECONDS}"
    continue
  fi

  wait_for_gpu
  attempt=$((attempt + 1))
  printf '%s optimizer attempt=%d fingerprint=%s\n' "$(date -Is)" "${attempt}" \
    "${current_fingerprint}" | tee -a "${LOG_FILE}"

  command=("${ROOT_DIR}/scripts/run_reload_validation.sh")
  if (( RESTART )); then
    command+=(--restart)
  fi
  command+=("${ENV_FILE}")
  run_dir="${OUTPUT_ROOT}/reload/$(date +%Y%m%d_%H%M%S)_attempt${attempt}"
  set +e
  NATIVE_COPY_BACKEND="${NATIVE_COPY_BACKEND}" \
    RELOAD_RUN_DIR="${run_dir}" "${command[@]}" 2>&1 \
    | tee -a "${LOG_FILE}"
  command_status=${PIPESTATUS[0]}
  set -e

  if [[ -n "${run_dir}" && -f "${run_dir}/validation.json" ]]; then
    if analyze_run "${run_dir}" | tee -a "${LOG_FILE}"; then
      printf '%s optimizer PASS run=%s\n' "$(date -Is)" "${run_dir}" \
        | tee -a "${LOG_FILE}"
      exit 0
    fi
  fi
  printf '%s optimizer attempt=%d failed status=%d; waiting for source change\n' \
    "$(date -Is)" "${attempt}" "${command_status}" | tee -a "${LOG_FILE}"
  last_fingerprint="${current_fingerprint}"
  printf '%s\n' "${last_fingerprint}" >"${STATE_FILE}"
done

printf '%s optimizer stopped after %d attempts\n' "$(date -Is)" "${attempt}" \
  | tee -a "${LOG_FILE}"
