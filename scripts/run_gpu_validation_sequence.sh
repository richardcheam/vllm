#!/usr/bin/env bash
set -euo pipefail

# Sequential GPU validation driver. It intentionally composes the existing
# launch/reload/benchmark harnesses instead of duplicating them.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_ENV="${ROOT_DIR}/.env.superinfer-service128"
OUTPUT_ROOT="${ROOT_DIR}/benchmark_artifacts/gpu_sequence"
POLL_SECONDS=30
IDLE_UTIL_PCT=10
IDLE_SAMPLES=3
PROFILE_PRESSURE=0
SKIP_PRESSURE=0
SKIP_RELOAD=0
SKIP_DEFAULT=0

usage() {
  printf '%s\n' \
    "Usage: ${BASH_SOURCE[0]} [options]" \
    "  --service-env PATH      Unified launch profile (default: .env.superinfer-service128)" \
    "  --output-root PATH      Sequence artifact root" \
    "  --poll-seconds N        GPU idle polling interval (default: 30)" \
    "  --idle-util-pct N       Maximum utilization for idle (default: 10)" \
    "  --idle-samples N        Consecutive idle samples (default: 3)" \
    "  --profile-pressure      Profile each paired server lifetime with Nsight" \
    "  --skip-reload           Skip both reload gates" \
    "  --skip-default          Skip the default backend and start at native" \
    "  --skip-pressure         Skip both pressure gates" \
    "  -h, --help              Show this help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --service-env) SERVICE_ENV="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --idle-util-pct) IDLE_UTIL_PCT="$2"; shift 2 ;;
    --idle-samples) IDLE_SAMPLES="$2"; shift 2 ;;
    --profile-pressure) PROFILE_PRESSURE=1; shift ;;
    --skip-reload) SKIP_RELOAD=1; shift ;;
    --skip-default) SKIP_DEFAULT=1; shift ;;
    --skip-pressure) SKIP_PRESSURE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

for env_file in "${SERVICE_ENV}"; do
  if [[ ! -f "${env_file}" ]]; then
    printf 'Missing environment file: %s\n' "${env_file}" >&2
    exit 2
  fi
done

SEQUENCE_DIR="${OUTPUT_ROOT}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SEQUENCE_DIR}"
RUN_LOG="${SEQUENCE_DIR}/sequence.log"
SUMMARY_JSON="${SEQUENCE_DIR}/sequence-summary.json"
PYTHON="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  printf 'Project-local Python executable not found: %s\n' "${PYTHON}" >&2
  exit 2
fi

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "${RUN_LOG}"
}

gpu_is_idle() {
  local values value
  values="$(nvidia-smi --query-gpu=utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ -n "${values}" ]] || return 1
  while read -r value; do
    [[ -n "${value}" && "${value}" -le "${IDLE_UTIL_PCT}" ]] || return 1
  done <<<"${values}"
}

wait_for_gpu() {
  local idle=0
  while (( idle < IDLE_SAMPLES )); do
    if gpu_is_idle; then
      idle=$((idle + 1))
      log "GPU idle sample ${idle}/${IDLE_SAMPLES}"
    else
      idle=0
      log "GPU busy; waiting"
    fi
    (( idle >= IDLE_SAMPLES )) || sleep "${POLL_SECONDS}"
  done
}

timestamp() {
  date +%Y%m%d_%H%M%S
}

run_reload() {
  local label="$1"
  local native="$2"
  local run_dir="${SEQUENCE_DIR}/reload/${label}_${native}_$(timestamp)"
  local native_env=0
  [[ "${native}" == "native" ]] && native_env=1
  mkdir -p "${run_dir}"
  log "Starting ${label} reload (${native}); artifact=${run_dir}"
  set +e
  env NATIVE_COPY_BACKEND="${native_env}" RELOAD_RUN_DIR="${run_dir}" \
    bash "${ROOT_DIR}/scripts/run_reload_validation.sh" \
    --no-restart "${SERVICE_ENV}" 2>&1 | tee -a "${RUN_LOG}"
  local status="${PIPESTATUS[0]}"
  set -e
  if [[ ! -f "${run_dir}/validation.json" ]]; then
    log "${label} reload (${native}) produced no validation.json"
    return 1
  fi
  "${PYTHON}" - "${run_dir}" "${status}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
command_status = int(sys.argv[2])
validation = json.loads((run_dir / "validation.json").read_text())
reload_result = json.loads((run_dir / "reload.json").read_text())
phase_timings = {
    phase["name"]: phase["elapsed_s"]
    for phase in reload_result.get("phases", [])
}
result = {
    "kind": "reload",
    "backend": run_dir.name.split("_")[1],
    "run_dir": str(run_dir),
    "command_status": command_status,
    "validation": validation,
    "phase_timings_s": phase_timings,
    "prompt_tokens_min": reload_result.get("prompt_tokens_min"),
    "prompt_tokens_max": reload_result.get("prompt_tokens_max"),
}
(run_dir / "sequence-result.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
if command_status != 0 or validation.get("status") != "PASS":
    raise SystemExit(1)
PY
  log "Completed ${label} reload (${native})"
}

run_pressure() {
  local label="$1"
  local native="$2"
  local native_env=0
  local pressure_root="${SEQUENCE_DIR}/pressure"
  local run_dir="${pressure_root}/${label}_${native}_$(timestamp)"
  [[ "${native}" == "native" ]] && native_env=1
  mkdir -p "${pressure_root}"
  log "Starting ${label} pressure (${native}); artifact=${run_dir}"
  set +e
  env NATIVE_COPY_BACKEND="${native_env}" \
    bash "${ROOT_DIR}/scripts/run_superinfer_bench.sh" \
    --no-start \
    --env-file "${SERVICE_ENV}" \
    --output-dir "${pressure_root}" \
    --run-id "$(basename "${run_dir}")" \
    2>&1 | tee -a "${RUN_LOG}"
  local status="${PIPESTATUS[0]}"
  set -e
  local actual_dir="${run_dir}"
  if [[ ! -f "${actual_dir}/benchmark-status.json" ]]; then
    log "${label} pressure (${native}) produced no benchmark-status.json"
    return 1
  fi
  "${PYTHON}" - "${actual_dir}" "${status}" "${native}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
command_status = int(sys.argv[2])
backend = sys.argv[3]
status = json.loads((run_dir / "benchmark-status.json").read_text())
run_files = sorted(run_dir.glob("run_*.json"))
if not run_files:
    raise SystemExit("no benchmark run JSON files found")
runs = [json.loads(path.read_text()) for path in run_files]
if any(int(run.get("successful", 0)) <= 0 or int(run.get("failed", 0)) > 0 for run in runs):
    raise SystemExit("benchmark contains failed or empty repeats")
expected_prompt_len = {int(run.get("prompt_len", 0)) for run in runs}
prompt_ranges = {
    (run.get("prompt_tokens_min"), run.get("prompt_tokens_max"))
    for run in runs
}
if len(expected_prompt_len) != 1 or prompt_ranges != {
    (next(iter(expected_prompt_len)), next(iter(expected_prompt_len)))
}:
    raise SystemExit("benchmark prompt-token lengths are not exact")
result = {
    "kind": "pressure",
    "backend": backend,
    "run_dir": str(run_dir),
    "command_status": command_status,
    "status": status,
    "runs": runs,
    "measurement_scope": "unique_prefix_d2h_pressure",
}
transfer_deltas = run_dir / "report" / "transfer-deltas.json"
if transfer_deltas.exists():
    result["transfer_deltas"] = json.loads(transfer_deltas.read_text())
(run_dir / "sequence-result.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
if command_status != 0 or status.get("status") != "PASS":
    raise SystemExit(1)
PY
  log "Completed ${label} pressure (${native})"
}

stop_backend() {
  bash "${ROOT_DIR}/scripts/stop_superinfer.sh" "${SERVICE_ENV}" \
    >>"${RUN_LOG}" 2>&1 || true
  SERVER_ACTIVE=0
}

SERVER_ACTIVE=0

cleanup() {
  if (( SERVER_ACTIVE )); then
    log "Stopping active server during sequence cleanup"
    stop_backend
  fi
}

trap cleanup EXIT INT TERM

run_backend() {
  local label="$1"
  local native="$2"
  local native_env=0
  [[ "${native}" == "native" ]] && native_env=1
  wait_for_gpu
  log "Starting ${label} server (${native}) once using ${SERVICE_ENV}"
  local -a launch_args=("${SERVICE_ENV}")
  if (( PROFILE_PRESSURE )); then
    launch_args+=(
      --nsys
      --nsys-output-dir "${SEQUENCE_DIR}/nsys/${label}_${native}"
    )
  fi
  if ! env NATIVE_COPY_BACKEND="${native_env}" \
    bash "${ROOT_DIR}/scripts/launch_superinfer.sh" "${launch_args[@]}" \
    2>&1 | tee -a "${RUN_LOG}"; then
    log "${label} server (${native}) failed to start"
    stop_backend
    return 1
  fi
  SERVER_ACTIVE=1
  if (( ! SKIP_RELOAD )); then
    if ! run_reload "${label}" "${native}"; then
      stop_backend
      return 1
    fi
  fi
  if (( ! SKIP_PRESSURE )); then
    if ! run_pressure "${label}" "${native}"; then
      stop_backend
      return 1
    fi
  fi
  stop_backend
  log "Stopped ${label} server (${native}) after paired stages"
}

log "GPU validation sequence started"
if (( ! SKIP_DEFAULT )); then
  run_backend postchange default
fi
run_backend postchange native

"${PYTHON}" - "${SEQUENCE_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
results = []
for path in sorted(root.glob("**/sequence-result.json")):
    results.append(json.loads(path.read_text()))
(root / "sequence-summary.json").write_text(
    json.dumps({"status": "PASS", "results": results}, indent=2) + "\n"
)
PY
log "GPU validation sequence completed successfully"
