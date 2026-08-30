#!/usr/bin/env bash
set -u -o pipefail

# Run a staged, one-server-lifetime serving soak for each candidate profile.
# The phases intentionally move from short decode-heavy traffic to long-context
# pressure and then back to short traffic to test recovery.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_ENV="${ROOT_DIR}/.env.superinfer-serving"
OUTPUT_ROOT="${ROOT_DIR}/benchmark_artifacts/serving_soak"
POLL_SECONDS=30
IDLE_UTIL_PCT=10
IDLE_SAMPLES=3
SETTLE_SECONDS="${SOAK_SETTLE_SECONDS:-10}"
TRANSFER_QUIESCENCE_TIMEOUT_S="${SOAK_TRANSFER_QUIESCENCE_TIMEOUT_S:-30}"
SOAK_SEED="${SOAK_SEED:-20260825}"
CANDIDATE_FILTER=""

# name|max_num_batched_tokens|max_num_seqs|proactive_swap_budget
CANDIDATES=(
  "short_peak_16384_32_600|16384|32|600"
  "current_32768_24_2400|32768|24|2400"
  "balanced_32768_24_600|32768|24|600"
  "long_prefill_65536_24_600|65536|24|600"
  "long_aggressive_65536_24_1200|65536|24|1200"
  "wide_long_131072_16_600|131072|16|600"
)

usage() {
  printf '%s\n' \
    "Usage: ${BASH_SOURCE[0]} [options]" \
    "  --service-env PATH      Base serving profile" \
    "  --output-root PATH      Artifact root" \
    "  --candidate NAME        Run one candidate only" \
    "  --poll-seconds N        GPU idle polling interval (default: 30)" \
    "  --idle-util-pct N       Maximum GPU utilization for idle (default: 10)" \
    "  --idle-samples N        Consecutive idle samples (default: 3)" \
    "  -h, --help              Show this help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --service-env) SERVICE_ENV="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --candidate) CANDIDATE_FILTER="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --idle-util-pct) IDLE_UTIL_PCT="$2"; shift 2 ;;
    --idle-samples) IDLE_SAMPLES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "${SERVICE_ENV}" ]]; then
  printf 'Missing serving profile: %s\n' "${SERVICE_ENV}" >&2
  exit 2
fi
SERVICE_ENV="$(realpath "${SERVICE_ENV}")"

set -a
# shellcheck disable=SC1090
source "${SERVICE_ENV}"
set +a
STATE_DIR="${STATE_DIR:-${ROOT_DIR}/.run}"
PID_FILE="${STATE_DIR}/server.pid"
if [[ -s "${PID_FILE}" ]]; then
  running_pid="$(<"${PID_FILE}")"
  if kill -0 "${running_pid}" 2>/dev/null; then
    printf 'A server is already running (pid=%s). Stop it before the soak.\n' \
      "${running_pid}" >&2
    exit 2
  fi
fi

PYTHON="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  printf 'Project-local Python executable not found: %s\n' "${PYTHON}" >&2
  exit 2
fi
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/..:${PYTHONPATH:-}"

SEQUENCE_DIR="${OUTPUT_ROOT}/$(date +%Y%m%d_%H%M%S)"
PROFILE_ROOT="${SEQUENCE_DIR}/profiles"
mkdir -p "${PROFILE_ROOT}"
RUN_LOG="${SEQUENCE_DIR}/serving-soak.log"

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

capture_snapshot() {
  local base="$1"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader >"${base}.gpu.csv" 2>/dev/null || true
  curl --noproxy '*' -fsSL --max-time 20 \
    "${BENCH_URL}/metrics" >"${base}.metrics.txt" 2>/dev/null || : >"${base}.metrics.txt"
}

health_ok() {
  curl --noproxy '*' -fsSL --max-time 20 \
    "${BENCH_URL}/health" >/dev/null 2>&1
}

wait_for_transfer_quiescence() {
  local deadline=$((SECONDS + TRANSFER_QUIESCENCE_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if "${PYTHON}" - "${BENCH_URL}" <<'PY'
import re
import sys
import urllib.request

url = sys.argv[1]
required = {
    "vllm:simple_cpu_offload_offload_pending_load_reqs",
    "vllm:simple_cpu_offload_offload_pending_store_events",
    "vllm:simple_cpu_offload_offload_load_queue_depth",
    "vllm:simple_cpu_offload_offload_store_queue_depth",
}
try:
    with urllib.request.urlopen(url + "/metrics", timeout=10) as response:
        text = response.read().decode("utf-8")
except Exception:
    raise SystemExit(2)

values = {}
pattern = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)$"
)
for line in text.splitlines():
    match = pattern.match(line.strip())
    if match:
        name = match.group("name")
        values[name] = max(values.get(name, 0.0), float(match.group("value")))
if required - values.keys():
    raise SystemExit(3)
if any(values[name] > 0 for name in required):
    raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep 2
  done
  return 1
}

write_candidate_config() {
  local candidate_dir="$1"
  local name="$2"
  local max_batched_tokens="$3"
  local max_seqs="$4"
  local budget="$5"
  local candidate_env="${candidate_dir}/candidate.env"

  cp "${SERVICE_ENV}" "${candidate_env}"
  printf '\n# Generated by run_serving_soak.sh for candidate %s.\n' "${name}" \
    >>"${candidate_env}"
  printf 'MAX_NUM_BATCHED_TOKENS=%s\n' "${max_batched_tokens}" >>"${candidate_env}"
  printf 'MAX_NUM_SEQS=%s\n' "${max_seqs}" >>"${candidate_env}"
  printf 'PROACTIVE_SWAP_BUDGET=%s\n' "${budget}" >>"${candidate_env}"
  printf 'LAUNCH_WARMUP_OUTPUT_DIR=%s/launch_warmup\n' \
    "${candidate_dir}" >>"${candidate_env}"
  "${PYTHON}" - "${candidate_dir}" "${name}" "${max_batched_tokens}" \
    "${max_seqs}" "${budget}" "${candidate_env}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "candidate-config.json"
path.write_text(
    json.dumps(
        {
            "name": sys.argv[2],
            "max_num_batched_tokens": int(sys.argv[3]),
            "max_num_seqs": int(sys.argv[4]),
            "proactive_swap_budget": int(sys.argv[5]),
            "environment": sys.argv[6],
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

run_phase() {
  local candidate_dir="$1"
  local candidate_name="$2"
  local stage_name="$3"
  local stage_index="$4"
  local users="$5"
  local requests="$6"
  local prompt_len="$7"
  local max_tokens="$8"
  local repeats="$9"
  local phase_root="${candidate_dir}/${stage_name}"
  local stage_status=0
  mkdir -p "${phase_root}"

  for repeat in $(seq 1 "${repeats}"); do
    local repeat_dir="${phase_root}/repeat_${repeat}"
    local run_status=0
    local seed=$((SOAK_SEED + stage_index * 100 + repeat))
    mkdir -p "${repeat_dir}"
    log "${candidate_name}/${stage_name}/repeat_${repeat}: users=${users} requests=${requests} prompt=${prompt_len} max_tokens=${max_tokens}"
    capture_snapshot "${repeat_dir}/before"
    if ! health_ok; then
      run_status=10
    else
      "${PYTHON}" "${ROOT_DIR}/../test_concurrent_robust.py" \
        --url "${BENCH_URL}" \
        --users "${users}" \
        --requests "${requests}" \
        --prompt-len "${prompt_len}" \
        --max-tokens "${max_tokens}" \
        --model "${SERVED_MODEL_NAME}" \
        --seed "${seed}" \
        --run-label "${candidate_name}-${stage_name}-repeat-${repeat}" \
        --output-json "${repeat_dir}/run_1.json" \
        > >(tee "${repeat_dir}/client.log") 2>&1 || run_status=$?
    fi
    sleep "${SETTLE_SECONDS}"
    capture_snapshot "${repeat_dir}/after"
    if [[ -f "${repeat_dir}/run_1.json" ]]; then
      "${PYTHON}" "${ROOT_DIR}/scripts/report_superinfer_bench.py" \
        "${repeat_dir}" --output-dir "${repeat_dir}/report" \
        >"${repeat_dir}/report.log" 2>&1 || true
    fi
    "${PYTHON}" - "${repeat_dir}" "${run_status}" "${stage_name}" \
      "${repeat}" "${users}" "${requests}" "${prompt_len}" "${max_tokens}" <<'PY'
import json
import sys
from pathlib import Path

from scripts.report_superinfer_bench import parse_metric_samples

repeat_dir = Path(sys.argv[1])
command_status = int(sys.argv[2])
stage_name = sys.argv[3]
repeat = int(sys.argv[4])
users = int(sys.argv[5])
requests = int(sys.argv[6])
prompt_len = int(sys.argv[7])
max_tokens = int(sys.argv[8])

metric_bases = {
    "store_events": "vllm:simple_cpu_offload_offload_store_events_total",
    "load_events": "vllm:simple_cpu_offload_offload_load_events_total",
    "store_bytes": "vllm:simple_cpu_offload_offload_store_bytes_total",
    "load_bytes": "vllm:simple_cpu_offload_offload_load_bytes_total",
    "native_store_submissions": (
        "vllm:simple_cpu_offload_offload_native_store_submissions_total"
    ),
    "native_load_submissions": (
        "vllm:simple_cpu_offload_offload_native_load_submissions_total"
    ),
    "native_store_submit_ns": (
        "vllm:simple_cpu_offload_offload_native_store_submit_ns_total"
    ),
    "native_load_submit_ns": (
        "vllm:simple_cpu_offload_offload_native_load_submit_ns_total"
    ),
}
gauge_bases = {
    "pending_store_events":
        "vllm:simple_cpu_offload_offload_pending_store_events",
    "pending_load_reqs":
        "vllm:simple_cpu_offload_offload_pending_load_reqs",
    "pending_store_reqs":
        "vllm:simple_cpu_offload_offload_pending_store_reqs",
    "load_queue_depth":
        "vllm:simple_cpu_offload_offload_load_queue_depth",
    "store_queue_depth":
        "vllm:simple_cpu_offload_offload_store_queue_depth",
    "dirty_blocks":
        "vllm:simple_cpu_offload_offload_residency_dirty_blocks",
    "invalid_blocks":
        "vllm:simple_cpu_offload_offload_residency_invalid_blocks",
    "cpu_used_blocks":
        "vllm:simple_cpu_offload_offload_cpu_used_blocks",
    "cpu_free_blocks":
        "vllm:simple_cpu_offload_offload_cpu_free_blocks",
}


def read(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    return parse_metric_samples(path)


def value(samples: dict[str, float], base: str) -> float:
    return max(
        (sample_value for name, sample_value in samples.items()
         if name.split("{", 1)[0] == base),
        default=0.0,
    )


before = read(repeat_dir / "before.metrics.txt")
after = read(repeat_dir / "after.metrics.txt")
transfer_deltas = {
    key: value(after, base) - value(before, base)
    for key, base in metric_bases.items()
}
final_gauges = {key: value(after, base) for key, base in gauge_bases.items()}
run_path = repeat_dir / "run_1.json"
run = json.loads(run_path.read_text()) if run_path.exists() else {}
successful = int(run.get("successful", 0))
failed = int(run.get("failed", requests))
exact_prompt = (
    run.get("prompt_tokens_min") == prompt_len
    and run.get("prompt_tokens_max") == prompt_len
)
status = (
    "PASS"
    if command_status == 0
    and successful == requests
    and failed == 0
    and exact_prompt
    and final_gauges.get("invalid_blocks", 0) == 0
    else "FAIL"
)
result = {
    "status": status,
    "command_status": command_status,
    "stage": stage_name,
    "repeat": repeat,
    "users": users,
    "requests": requests,
    "prompt_len": prompt_len,
    "max_tokens": max_tokens,
    "successful": successful,
    "failed": failed,
    "exact_prompt_tokens": exact_prompt,
    "transfer_deltas": transfer_deltas,
    "final_gauges": final_gauges,
    "transfer_quiescent_at_capture": (
        final_gauges.get("pending_store_events", 0) == 0
        and final_gauges.get("pending_load_reqs", 0) == 0
        and final_gauges.get("load_queue_depth", 0) == 0
        and final_gauges.get("store_queue_depth", 0) == 0
    ),
    "run": run,
}
(repeat_dir / "phase-result.json").write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps({"status": status, "stage": stage_name, "repeat": repeat}))
PY
    local result_status
    result_status="$(${PYTHON} - "${repeat_dir}/phase-result.json" <<'PY'
import json
import sys
print(json.loads(open(sys.argv[1]).read()).get("status", "FAIL"))
PY
    )"
    if [[ "${result_status}" != "PASS" ]]; then
      stage_status=1
    fi
  done

  return "${stage_status}"
}

run_reload_phase() {
  local candidate_dir="$1"
  local candidate_name="$2"
  local phase_root="${candidate_dir}/reload_131k"
  local run_status=0
  mkdir -p "${phase_root}"
  log "${candidate_name}/reload_131k: repeated-prefix CPU KV reload"
  capture_snapshot "${phase_root}/before"
  if ! health_ok; then
    run_status=10
  else
    RELOAD_RUN_DIR="${phase_root}" \
      TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}" \
      RELOAD_PROMPT_LEN=131072 \
      RELOAD_REQUESTS=4 \
      RELOAD_EVICT_REQUESTS=8 \
      RELOAD_EVICT_CONCURRENCY=24 \
      RELOAD_REPEATS=1 \
      RELOAD_CONCURRENCY=4 \
      RELOAD_MAX_TOKENS=64 \
      bash "${ROOT_DIR}/scripts/run_reload_validation.sh" \
      --no-restart "${candidate_dir}/candidate.env" \
      > >(tee "${phase_root}/client.log") 2>&1 || run_status=$?
  fi
  sleep "${SETTLE_SECONDS}"
  capture_snapshot "${phase_root}/after"
  "${PYTHON}" - "${phase_root}" "${run_status}" "${candidate_name}" <<'PY'
import json
import sys
from pathlib import Path

phase_root = Path(sys.argv[1])
run_status = int(sys.argv[2])
candidate = sys.argv[3]
validation_path = phase_root / "validation.json"
reload_path = phase_root / "reload.json"
validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
reload = json.loads(reload_path.read_text()) if reload_path.exists() else {}
phase_timings = {
    phase["name"]: phase["elapsed_s"]
    for phase in reload.get("phases", [])
}
result = {
    "status": "PASS"
    if run_status == 0 and validation.get("status") == "PASS"
    else "FAIL",
    "command_status": run_status,
    "stage": "reload_131k",
    "candidate": candidate,
    "validation": validation,
    "phase_timings_s": phase_timings,
    "prompt_tokens_min": reload.get("prompt_tokens_min"),
    "prompt_tokens_max": reload.get("prompt_tokens_max"),
    "reload": reload,
}
(phase_root / "phase-result.json").write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps({"status": result["status"], "stage": "reload_131k"}))
PY
  [[ "$("${PYTHON}" - "${phase_root}/phase-result.json" <<'PY'
import json
import sys
print(json.loads(open(sys.argv[1]).read()).get("status", "FAIL"))
PY
  )" == "PASS" ]]
}

run_candidate() (
  local candidate_dir="$1"
  local candidate_name="$2"
  local max_batched_tokens="$3"
  local max_seqs="$4"
  local budget="$5"
  local candidate_env="${candidate_dir}/candidate.env"
  local candidate_failed=0

  set -a
  # shellcheck disable=SC1090
  source "${candidate_env}"
  set +a
  export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/..:${PYTHONPATH:-}"
  BENCH_URL="${BENCH_URL:-http://127.0.0.1:${VLLM_PORT:-8202}}"
  SERVER_LOG="${LOG_DIR:-${ROOT_DIR}/logs}/server_${PROFILE:-superinfer}_${VLLM_PORT:-8202}.log"
  SERVER_ACTIVE=0

  cleanup() {
    if (( SERVER_ACTIVE )); then
      bash "${ROOT_DIR}/scripts/stop_superinfer.sh" "${candidate_env}" \
        >>"${candidate_dir}/candidate.log" 2>&1 || true
      SERVER_ACTIVE=0
    fi
  }
  trap cleanup EXIT INT TERM

  wait_for_gpu
  log "Launching candidate ${candidate_name}: max_tokens=${max_batched_tokens} max_seqs=${max_seqs} budget=${budget}"
  if ! bash "${ROOT_DIR}/scripts/launch_superinfer.sh" "${candidate_env}" \
    > >(tee "${candidate_dir}/launch.log" | tee -a "${RUN_LOG}") 2>&1; then
    candidate_failed=1
  else
    SERVER_ACTIVE=1
  fi

  if (( ! candidate_failed )); then
    run_phase "${candidate_dir}" "${candidate_name}" "warm_4k" 1 8 16 4096 128 1 || candidate_failed=1
  fi
  if (( ! candidate_failed )); then
    run_phase "${candidate_dir}" "${candidate_name}" "warm_8k" 2 16 32 8192 128 1 || candidate_failed=1
  fi
  if (( ! candidate_failed )); then
    run_phase "${candidate_dir}" "${candidate_name}" "short_peak_16k" 3 24 48 16384 256 3 || candidate_failed=1
  fi
  if (( ! candidate_failed )); then
    run_phase "${candidate_dir}" "${candidate_name}" "mid_32k" 4 16 16 32768 128 1 || candidate_failed=1
  fi
  if (( ! candidate_failed )); then
    run_phase "${candidate_dir}" "${candidate_name}" "mid_64k" 5 8 8 65536 128 1 || candidate_failed=1
  fi
  if (( ! candidate_failed )); then
    run_phase "${candidate_dir}" "${candidate_name}" "long_131k_4u" 6 4 8 131072 128 1 || candidate_failed=1
  fi
  if (( ! candidate_failed )); then
    run_phase "${candidate_dir}" "${candidate_name}" "long_131k_8u" 7 8 16 131072 128 1 || candidate_failed=1
  fi
  if (( ! candidate_failed )); then
    run_phase "${candidate_dir}" "${candidate_name}" "long_131k_16u" 8 16 32 131072 128 1 || candidate_failed=1
  fi
  if (( ! candidate_failed )); then
    run_reload_phase "${candidate_dir}" "${candidate_name}" || candidate_failed=1
  fi
  if (( ! candidate_failed )); then
    run_phase "${candidate_dir}" "${candidate_name}" "recovery_16k" 10 24 24 16384 256 2 || candidate_failed=1
  fi

  capture_snapshot "${candidate_dir}/final"
  "${PYTHON}" - "${candidate_dir}" "${candidate_name}" "${candidate_failed}" <<'PY'
import json
import sys
from pathlib import Path

candidate_dir = Path(sys.argv[1])
result = {
    "candidate": sys.argv[2],
    "runner_status": "FAIL" if int(sys.argv[3]) else "PASS",
    "completed": int(sys.argv[3]) == 0,
    "phase_results": [
        str(path.relative_to(candidate_dir))
        for path in sorted(
            list(candidate_dir.glob("*/repeat_*/phase-result.json"))
            + [candidate_dir / "reload_131k" / "phase-result.json"]
        )
        if path.exists()
    ],
}
(candidate_dir / "candidate-result.json").write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf-8"
)
PY
  if (( candidate_failed )); then
    return 1
  fi
  return 0
)

log "Serving soak started: ${SEQUENCE_DIR}"
overall_status=0
candidate_index=0
for definition in "${CANDIDATES[@]}"; do
  IFS='|' read -r name max_batched_tokens max_seqs budget <<<"${definition}"
  if [[ -n "${CANDIDATE_FILTER}" && "${name}" != "${CANDIDATE_FILTER}" ]]; then
    continue
  fi
  candidate_index=$((candidate_index + 1))
  candidate_dir="${PROFILE_ROOT}/${name}"
  mkdir -p "${candidate_dir}"
  write_candidate_config "${candidate_dir}" "${name}" \
    "${max_batched_tokens}" "${max_seqs}" "${budget}"
  if ! run_candidate "${candidate_dir}" "${name}" "${max_batched_tokens}" \
    "${max_seqs}" "${budget}"; then
    overall_status=1
    log "Candidate ${name} failed; continuing to the next candidate"
  else
    log "Candidate ${name} completed"
  fi
done

if (( candidate_index == 0 )); then
  printf 'No candidate matched: %s\n' "${CANDIDATE_FILTER}" >&2
  exit 2
fi

"${PYTHON}" "${ROOT_DIR}/scripts/report_serving_soak.py" \
  "${SEQUENCE_DIR}" --output-dir "${SEQUENCE_DIR}/report" \
  >"${SEQUENCE_DIR}/report.log" 2>&1 || overall_status=1
log "Serving soak report: ${SEQUENCE_DIR}/report"
log "Serving soak completed with status=$([[ ${overall_status} -eq 0 ]] && printf PASS || printf PARTIAL)"
printf 'SERVING_SOAK_DIR=%s\n' "${SEQUENCE_DIR}"
exit "${overall_status}"
