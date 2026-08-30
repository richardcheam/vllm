#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${1:-${ROOT_DIR}/.env.superinfer-pressure}"
RESTART=0

if [[ "${1:-}" == "--restart" ]]; then
  ENV_FILE="${2:-${ROOT_DIR}/.env.superinfer-pressure}"
  RESTART=1
elif [[ "${1:-}" == "--no-restart" ]]; then
  ENV_FILE="${2:-${ROOT_DIR}/.env.superinfer-pressure}"
elif [[ "${2:-}" == "--restart" ]]; then
  RESTART=1
elif [[ "${2:-}" == "--no-restart" ]]; then
  :
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/..:${PYTHONPATH:-}"

PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
OUTPUT_ROOT="${RELOAD_OUTPUT_DIR:-${ROOT_DIR}/benchmark_artifacts/reload_validation}"
RUN_DIR="${RELOAD_RUN_DIR:-${OUTPUT_ROOT}/$(date +%Y%m%d_%H%M%S)_reload}"
SERVER_LOG="${LOG_DIR:-${ROOT_DIR}/logs}/server_${PROFILE:-superinfer}_${VLLM_PORT:-8202}.log"
TRANSFER_QUIESCENCE_TIMEOUT_S="${TRANSFER_QUIESCENCE_TIMEOUT_S:-120}"
mkdir -p "${RUN_DIR}"

capture() {
  local name="$1"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader >"${RUN_DIR}/${name}.gpu.csv" 2>/dev/null || true
  curl --noproxy '*' -fsSL "${BENCH_URL}/metrics" \
    >"${RUN_DIR}/${name}.metrics.txt" 2>/dev/null || : >"${RUN_DIR}/${name}.metrics.txt"
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

if (( RESTART )); then
  "${ROOT_DIR}/scripts/stop_superinfer.sh" "${ENV_FILE}" >/dev/null 2>&1 || true
  if ! SKIP_LAUNCH_WARMUP=1 LAUNCH_WARMUP_ENABLED=0 \
    "${ROOT_DIR}/scripts/launch_superinfer.sh" "${ENV_FILE}"; then
    capture startup_failure
    if [[ -f "${SERVER_LOG}" ]]; then
      cp "${SERVER_LOG}" "${RUN_DIR}/server.log" 2>/dev/null || true
    fi
    cat >"${RUN_DIR}/reload.json" <<EOF
{
  "status": "FAIL",
  "output_mismatches": -1,
  "startup_failed": true
}
EOF
    cat >"${RUN_DIR}/validation.json" <<EOF
{
  "status": "FAIL",
  "failures": ["server startup failed"],
  "metric_deltas": {},
  "output_mismatches": -1
}
EOF
    printf 'Reload validation startup failed: %s\n' "${RUN_DIR}" >&2
    exit 1
  fi
  cleanup() {
    "${ROOT_DIR}/scripts/stop_superinfer.sh" "${ENV_FILE}" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT INT TERM
fi

capture before
client_status=0
quiescence_status=0
"${PYTHON}" "${ROOT_DIR}/scripts/validate_superinfer_reload.py" \
  --url "${BENCH_URL}" \
  --model "${SERVED_MODEL_NAME}" \
  --prompt-len "${RELOAD_PROMPT_LEN:-131072}" \
  --requests "${RELOAD_REQUESTS:-16}" \
  --evict-requests "${RELOAD_EVICT_REQUESTS:-24}" \
  --evict-concurrency "${RELOAD_EVICT_CONCURRENCY:-24}" \
  --reload-repeats "${RELOAD_REPEATS:-1}" \
  --reload-concurrency "${RELOAD_CONCURRENCY:-1}" \
  --max-tokens "${RELOAD_MAX_TOKENS:-64}" \
  --output-json "${RUN_DIR}/reload.json" || client_status=$?
wait_for_transfer_quiescence || quiescence_status=$?
capture after

if [[ ! -f "${RUN_DIR}/reload.json" ]]; then
  cat >"${RUN_DIR}/reload.json" <<EOF
{
  "status": "FAIL",
  "output_mismatches": -1,
  "client_status": ${client_status}
}
EOF
fi

validation_status=0
"${PYTHON}" - "${RUN_DIR}" "${quiescence_status}" <<'PY' || validation_status=$?
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])


def read_metric(path: Path, name: str) -> float:
    pattern = re.compile(
        rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return float(match.group(1))
    return 0.0


metric_names = {
    "store_events": (
        "vllm:simple_cpu_offload_offload_store_events_total"
    ),
    "load_events": "vllm:simple_cpu_offload_offload_load_events_total",
    "store_bytes": "vllm:simple_cpu_offload_offload_store_bytes_total",
    "load_bytes": "vllm:simple_cpu_offload_offload_load_bytes_total",
    "lookup_requests": (
        "vllm:simple_cpu_offload_offload_cpu_lookup_requests_total"
    ),
    "lookup_hits": (
        "vllm:simple_cpu_offload_offload_cpu_lookup_hits_total"
    ),
}
before = run_dir / "before.metrics.txt"
after = run_dir / "after.metrics.txt"
deltas = {
    key: read_metric(after, metric) - read_metric(before, metric)
    for key, metric in metric_names.items()
}
cached_keys_before = read_metric(
    before, "vllm:simple_cpu_offload_offload_cpu_cached_keys"
)
cached_keys_after = read_metric(
    after, "vllm:simple_cpu_offload_offload_cpu_cached_keys"
)
reload_result = json.loads((run_dir / "reload.json").read_text(encoding="utf-8"))
output_mismatches = int(reload_result.get("output_mismatches", -1))
failures = []
if reload_result.get("status") != "PASS" or output_mismatches != 0:
    failures.append("reload output mismatch")
if deltas["store_events"] <= 0 or deltas["store_bytes"] <= 0:
    failures.append("store counters did not increase")
if deltas["load_events"] <= 0 or deltas["load_bytes"] <= 0:
    failures.append("load counters did not increase")
client_status = int(reload_result.get("client_status", 0))
if client_status != 0:
    failures.append(f"reload client failed with status {client_status}")
expected_prompt_tokens = int(reload_result.get("prompt_len", 0))
if (
    reload_result.get("prompt_tokens_min") != expected_prompt_tokens
    or reload_result.get("prompt_tokens_max") != expected_prompt_tokens
):
    failures.append("reload prompt-token lengths are not exact")

validation = {
    "status": "PASS" if not failures else "FAIL",
    "failures": failures,
    "metric_deltas": deltas,
    "cached_keys_before": cached_keys_before,
    "cached_keys_after": cached_keys_after,
    "output_mismatches": output_mismatches,
    "transfer_quiescent": int(sys.argv[2]) == 0,
}
(run_dir / "validation.json").write_text(
    json.dumps(validation, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(validation, sort_keys=True))
if failures:
    raise SystemExit(1)
PY

if [[ -f "${SERVER_LOG}" ]]; then
  cp "${SERVER_LOG}" "${RUN_DIR}/server.log" 2>/dev/null || true
fi
printf 'Reload validation complete: %s\n' "${RUN_DIR}"
exit "${validation_status}"
