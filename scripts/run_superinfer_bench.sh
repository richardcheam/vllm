#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env}"
AUTOSTART=1
OUTPUT_DIR_OVERRIDE=""
RUN_ID_OVERRIDE=""
NSYS_PROFILE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --no-start)
      AUTOSTART=0
      shift
      ;;
    --output-dir)
      OUTPUT_DIR_OVERRIDE="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID_OVERRIDE="$2"
      shift 2
      ;;
    --nsys)
      NSYS_PROFILE=1
      shift
      ;;
    -h|--help)
      printf 'Usage: %s [ENV_FILE] [--no-start] [--output-dir PATH] [--run-id ID] [--nsys]\n' "${BASH_SOURCE[0]}"
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

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/..:${PYTHONPATH:-}"

BENCH_REPEATS="${BENCH_REPEATS:-3}"
BENCH_OUTPUT_DIR="${BENCH_OUTPUT_DIR:-${ROOT_DIR}/benchmark_artifacts}"
BENCH_URL="${BENCH_URL:-http://127.0.0.1:${VLLM_PORT:-8202}}"
BENCH_CLIENT="${BENCH_CLIENT:-robust}"
BENCH_SEED="${BENCH_SEED:-20260803}"
WARMUP_ENABLED="${WARMUP_ENABLED:-1}"
WARMUP_MODE="${WARMUP_MODE:-exhaustive}"
WARMUP_STAGES="${WARMUP_STAGES:-6}"
WARMUP_CLIENT="${WARMUP_CLIENT:-robust}"
WARMUP_SEED="${WARMUP_SEED:-20260803}"
WARMUP_SATURATION_SOAK="${WARMUP_SATURATION_SOAK:-1}"
if [[ -n "${OUTPUT_DIR_OVERRIDE}" ]]; then
  BENCH_OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE}"
fi
RUN_ID="${RUN_ID_OVERRIDE:-$(date +%Y%m%d_%H%M%S)_${PROFILE:-superinfer}}"
RUN_DIR="${BENCH_OUTPUT_DIR}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
SERVER_LOG="${LOG_DIR:-${ROOT_DIR}/logs}/server_${PROFILE:-superinfer}_${VLLM_PORT:-8202}.log"
TRANSFER_QUIESCENCE_TIMEOUT_S="${TRANSFER_QUIESCENCE_TIMEOUT_S:-120}"

capture_runtime_snapshot() {
  local output_base="$1"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader >"${output_base}.gpu.csv" 2>/dev/null || true
  curl --noproxy '*' -fsSL "${BENCH_URL}/metrics" \
    >"${output_base}.metrics.txt" 2>/dev/null || {
    : >"${output_base}.metrics.txt"
  }
  if [[ -f "${SERVER_LOG}" ]]; then
    {
      printf 'error_lines='
      grep -c ' ERROR ' "${SERVER_LOG}" || true
      printf 'traceback_lines='
      grep -c 'Traceback (most recent call last)' "${SERVER_LOG}" || true
      printf 'engine_dead_lines='
      grep -c 'EngineDeadError' "${SERVER_LOG}" || true
      printf 'model_404_lines='
      grep -c 'does not exist\.' "${SERVER_LOG}" || true
    } >"${output_base}.log-diagnostics.txt"
  fi
}

finalize_run() {
  local status="${1:-FAIL}"
  trap - ERR INT TERM
  capture_runtime_snapshot "${RUN_DIR}/final"
  if [[ -f "${SERVER_LOG}" ]]; then
    cp "${SERVER_LOG}" "${RUN_DIR}/server.log" 2>/dev/null || true
  fi
  "${PYTHON}" - "${RUN_DIR}" "${status}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
status = sys.argv[2]
path = run_dir / "benchmark-status.json"
data = {"status": status}
if path.exists():
    try:
        data.update(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        pass
data["status"] = status
path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
  if [[ -f "${RUN_DIR}/run_1.json" ]]; then
    "${PYTHON}" "${ROOT_DIR}/scripts/report_superinfer_bench.py" "${RUN_DIR}" \
      --output-dir "${RUN_DIR}/report" >/dev/null 2>&1 || true
  fi
}

trap 'finalize_run FAIL' ERR INT TERM

if (( AUTOSTART )); then
  launch_args=(--env-file "${ENV_FILE}")
  if (( NSYS_PROFILE )); then
    launch_args+=(--nsys --nsys-output-dir "${RUN_DIR}/nsys")
  fi
  "${ROOT_DIR}/scripts/launch_superinfer.sh" "${launch_args[@]}"
  trap '"${ROOT_DIR}/scripts/stop_superinfer.sh" --env-file "${ENV_FILE}" >/dev/null 2>&1 || true' EXIT
else
  if (( NSYS_PROFILE )); then
    printf '%s\n' '--nsys requires an autostarted server.' >&2
    exit 1
  fi
  printf 'Using already-running server at %s.\n' "${BENCH_URL}"
fi

if [[ ! -f "${ROOT_DIR}/../test_concurrent.py" || ! -f "${ROOT_DIR}/../test_concurrent_robust.py" ]]; then
  printf 'Benchmark clients not found under %s/..\n' "${ROOT_DIR}" >&2
  exit 1
fi

validate_run_output() {
  local output="$1"
  "${PYTHON}" - "${output}" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
successful = int(data.get("successful", 0))
failed = int(data.get("failed", 0))
if successful <= 0 or failed > 0:
    raise SystemExit(
        f"invalid benchmark result: successful={successful} failed={failed}"
    )
PY
}

wait_for_transfer_quiescence() {
  local deadline=$((SECONDS + TRANSFER_QUIESCENCE_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if curl --noproxy '*' -fsSL --max-time 10 "${BENCH_URL}/metrics" \
      | "${PYTHON}" -c '
import re
import sys

required = {
    "vllm:simple_cpu_offload_offload_pending_load_reqs",
    "vllm:simple_cpu_offload_offload_pending_store_events",
    "vllm:simple_cpu_offload_offload_load_queue_depth",
    "vllm:simple_cpu_offload_offload_store_queue_depth",
}
values = {}
pattern = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)$"
)
for line in sys.stdin:
    match = pattern.match(line.strip())
    if match:
        name = match.group("name")
        values[name] = max(values.get(name, 0.0), float(match.group("value")))
if not values:
    raise SystemExit(2)
missing = required - values.keys()
if missing:
    raise SystemExit(3)
if any(values[name] > 0 for name in required):
    raise SystemExit(1)
print("transfer queues quiescent")
'; then
      return 0
    fi
    sleep 2
  done
  printf 'Transfer queues did not quiesce within %ss.\n' \
    "${TRANSFER_QUIESCENCE_TIMEOUT_S}" >&2
  return 1
}

bench_client="${ROOT_DIR}/../test_concurrent.py"
bench_extra_args=()
if [[ "${BENCH_CLIENT}" == "robust" ]]; then
  bench_client="${ROOT_DIR}/../test_concurrent_robust.py"
  bench_extra_args+=(--seed "${BENCH_SEED}")
elif [[ "${BENCH_CLIENT}" != "basic" ]]; then
  printf 'Unsupported BENCH_CLIENT=%s; use robust or basic.\n' "${BENCH_CLIENT}" >&2
  exit 1
fi

if [[ "${WARMUP_ENABLED}" == "1" ]]; then
  if [[ "${WARMUP_MODE}" != "exhaustive" ]]; then
    printf 'Unsupported WARMUP_MODE=%s; use exhaustive.\n' "${WARMUP_MODE}" >&2
    exit 1
  fi

  printf 'Running exhaustive %s-stage warmup before measurement.\n' "${WARMUP_STAGES}"
  IFS=',' read -r -a warmup_prompt_lengths <<< "${WARMUP_PROMPT_LENGTHS:-4096,16384,65536,131072,262144,524288}"
  IFS=',' read -r -a warmup_users <<< "${WARMUP_USERS:-4,8,16,24,32,32}"
  IFS=',' read -r -a warmup_requests <<< "${WARMUP_REQUESTS:-8,32,64,96,128,128}"
  IFS=',' read -r -a warmup_tokens <<< "${WARMUP_MAX_TOKENS:-32,64,128,256,256,256}"
  warmup_count=${#warmup_prompt_lengths[@]}
  if (( warmup_count == 0 || ${#warmup_users[@]} != warmup_count ||
    ${#warmup_requests[@]} != warmup_count || ${#warmup_tokens[@]} != warmup_count )); then
    printf 'Warmup stage arrays must have equal non-zero lengths.\n' >&2
    exit 1
  fi
  if (( WARMUP_STAGES < 1 )); then
    printf 'WARMUP_STAGES must be >= 1.\n' >&2
    exit 1
  fi
  warmup_client="${ROOT_DIR}/../test_concurrent.py"
  warmup_extra_args=()
  warmup_label_args=()
  if [[ "${WARMUP_CLIENT}" == "robust" ]]; then
    warmup_client="${ROOT_DIR}/../test_concurrent_robust.py"
    warmup_extra_args+=(--seed "${WARMUP_SEED}")
    warmup_label_args+=(--run-label)
  elif [[ "${WARMUP_CLIENT}" != "basic" ]]; then
    printf 'Unsupported WARMUP_CLIENT=%s; use robust or basic.\n' "${WARMUP_CLIENT}" >&2
    exit 1
  fi

  for stage in $(seq 1 "${WARMUP_STAGES}"); do
    index=$((stage - 1))
    if (( index >= warmup_count )); then
      index=$((warmup_count - 1))
    fi
    output="${RUN_DIR}/warmup_stage_${stage}.json"
    printf 'Warmup stage %s/%s: prompt=%s users=%s requests=%s max_tokens=%s client=%s\n' \
      "${stage}" "${WARMUP_STAGES}" "${warmup_prompt_lengths[$index]}" \
      "${warmup_users[$index]}" "${warmup_requests[$index]}" \
      "${warmup_tokens[$index]}" "${WARMUP_CLIENT}"
    capture_runtime_snapshot "${output%.json}.before"
    "${PYTHON}" "${warmup_client}" \
      --url "${BENCH_URL}" \
      --users "${warmup_users[$index]}" \
      --requests "${warmup_requests[$index]}" \
      --prompt-len "${warmup_prompt_lengths[$index]}" \
      --max-tokens "${warmup_tokens[$index]}" \
      --model "${SERVED_MODEL_NAME}" \
      "${warmup_extra_args[@]}" \
      "${warmup_label_args[@]}" "warmup-stage-${stage}" \
      --output-json "${output}"
    validate_run_output "${output}"
    capture_runtime_snapshot "${output%.json}.after"
  done

  if [[ "${WARMUP_SATURATION_SOAK}" == "1" ]]; then
    soak_output="${RUN_DIR}/warmup_saturation_soak.json"
    printf 'Warmup saturation soak: users=%s requests=%s prompt=%s max_tokens=%s\n' \
      "${BENCH_USERS:-24}" "${BENCH_REQUESTS:-120}" \
      "${BENCH_PROMPT_LEN:-16384}" "${BENCH_MAX_TOKENS:-256}"
    capture_runtime_snapshot "${soak_output%.json}.before"
    "${PYTHON}" "${ROOT_DIR}/../test_concurrent_robust.py" \
      --url "${BENCH_URL}" \
      --users "${BENCH_USERS:-24}" \
      --requests "${BENCH_REQUESTS:-120}" \
      --prompt-len "${BENCH_PROMPT_LEN:-16384}" \
      --max-tokens "${BENCH_MAX_TOKENS:-256}" \
      --model "${SERVED_MODEL_NAME}" \
      --seed "$((WARMUP_SEED + 1000))" \
      --run-label warmup-saturation-soak \
      --output-json "${soak_output}"
    validate_run_output "${soak_output}"
    capture_runtime_snapshot "${soak_output%.json}.after"
  fi
else
  printf 'Warmup disabled.\n'
fi

for repeat in $(seq 1 "${BENCH_REPEATS}"); do
  output="${RUN_DIR}/run_${repeat}.json"
  printf 'Benchmark repeat %s/%s -> %s (client=%s)\n' \
    "${repeat}" "${BENCH_REPEATS}" "${output}" "${BENCH_CLIENT}"
  measurement_label_args=()
  if [[ "${BENCH_CLIENT}" == "robust" ]]; then
    measurement_label_args+=(--run-label "measurement-repeat-${repeat}")
  fi
  capture_runtime_snapshot "${output%.json}.before"
  "${PYTHON}" "${bench_client}" \
    --url "${BENCH_URL}" \
    --users "${BENCH_USERS:-24}" \
    --requests "${BENCH_REQUESTS:-120}" \
    --prompt-len "${BENCH_PROMPT_LEN:-16384}" \
    --max-tokens "${BENCH_MAX_TOKENS:-256}" \
    --model "${SERVED_MODEL_NAME}" \
    "${bench_extra_args[@]}" \
    "${measurement_label_args[@]}" \
    --output-json "${output}"
  validate_run_output "${output}"
  if [[ "${PROFILE:-superinfer}" == "superinfer" ||
    "${PROFILE:-superinfer}" == "superinfer-high-risk" ]]; then
    wait_for_transfer_quiescence
  fi
  capture_runtime_snapshot "${output%.json}.after"
done

trap - ERR INT TERM
finalize_run PASS
"${PYTHON}" "${ROOT_DIR}/scripts/report_superinfer_bench.py" "${RUN_DIR}" \
  --output-dir "${RUN_DIR}/report"
printf 'Benchmark complete: %s\n' "${RUN_DIR}"
