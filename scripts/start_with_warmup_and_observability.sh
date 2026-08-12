#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONTAINER_NAME="${CONTAINER_NAME:-richard-base-dev}"
PORT="${PORT:-8202}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/daily_ops}"
OBS_DIR="${OBS_DIR:-${REPO_ROOT}/observability}"

#WARMUP_MODEL="${WARMUP_MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
WARMUP_MODEL="${WARMUP_MODEL:-deepseek-ai/DeepSeek-V4-Flash-0731}"
WARMUP_LABEL="${WARMUP_LABEL:-warmup}"

STAGE_A_ENABLED="${STAGE_A_ENABLED:-1}"
STAGE_A_USERS="${STAGE_A_USERS:-24}"
STAGE_A_REQUESTS="${STAGE_A_REQUESTS:-120}"
STAGE_A_MAX_TOKENS="${STAGE_A_MAX_TOKENS:-256}"
STAGE_A_PROMPT_LEN="${STAGE_A_PROMPT_LEN:-16384}"

STAGE_B_ENABLED="${STAGE_B_ENABLED:-1}"
STAGE_B_USERS="${STAGE_B_USERS:-32}"
STAGE_B_REQUESTS="${STAGE_B_REQUESTS:-200}"
STAGE_B_MAX_TOKENS="${STAGE_B_MAX_TOKENS:-1024}"
STAGE_B_PROMPT_LEN="${STAGE_B_PROMPT_LEN:-2048}"

STAGE_C_ENABLED="${STAGE_C_ENABLED:-1}"
STAGE_C_USERS="${STAGE_C_USERS:-50}"
STAGE_C_REQUESTS="${STAGE_C_REQUESTS:-300}"
STAGE_C_MAX_TOKENS="${STAGE_C_MAX_TOKENS:-1024}"
STAGE_C_PROMPT_LEN="${STAGE_C_PROMPT_LEN:-4096}"

STAGE_D_ENABLED="${STAGE_D_ENABLED:-0}"
STAGE_D_USERS="${STAGE_D_USERS:-32}"
STAGE_D_REQUESTS="${STAGE_D_REQUESTS:-32}"
STAGE_D_MAX_TOKENS="${STAGE_D_MAX_TOKENS:-128}"
STAGE_D_PROMPT_LEN="${STAGE_D_PROMPT_LEN:-65536}"

STAGE_E_ENABLED="${STAGE_E_ENABLED:-0}"
STAGE_E_USERS="${STAGE_E_USERS:-32}"
STAGE_E_REQUESTS="${STAGE_E_REQUESTS:-64}"
STAGE_E_MAX_TOKENS="${STAGE_E_MAX_TOKENS:-512}"
STAGE_E_PROMPT_LEN="${STAGE_E_PROMPT_LEN:-98304}"

STAGE_F_ENABLED="${STAGE_F_ENABLED:-0}"
STAGE_F_USERS="${STAGE_F_USERS:-50}"
STAGE_F_REQUESTS="${STAGE_F_REQUESTS:-300}"
STAGE_F_MAX_TOKENS="${STAGE_F_MAX_TOKENS:-1024}"
STAGE_F_PROMPT_LEN="${STAGE_F_PROMPT_LEN:-4096}"

BENCH_WORKDIR="${BENCH_WORKDIR:-/workspace/re-SuperInfer}"
BENCH_SCRIPT="${BENCH_SCRIPT:-/workspace/re-SuperInfer/test_concurrent_robust.py}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/re-SuperInfer/vllm-modern/.venv/bin/python}"

COLLECT_INTERVAL_S="${COLLECT_INTERVAL_S:-15}"
METRICS_URL="${METRICS_URL:-http://127.0.0.1:${PORT}/metrics}"

# --- concurrency guard: only one warmup at a time per port ---
mkdir -p "${LOG_ROOT}"
WARMUP_LOCK="${LOG_ROOT}/warmup_p${PORT}.lock"
exec 9>"${WARMUP_LOCK}"
if ! flock -n 9; then
  echo "[warmup] another instance is already running for port ${PORT} (lock: ${WARMUP_LOCK}); skipping." >&2
  exit 1
fi
# ---

mkdir -p "${LOG_ROOT}"

RUN_ID="window_$(date +%Y%m%d_%H%M%S)_p${PORT}"
RUN_DIR="${LOG_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

echo "${RUN_ID}" > "${LOG_ROOT}/active_window_id"
echo "${RUN_DIR}" > "${LOG_ROOT}/active_window_dir"
echo "run_id=${RUN_ID}" > "${RUN_DIR}/meta.env"
echo "start_ts=$(date --iso-8601=seconds)" >> "${RUN_DIR}/meta.env"
echo "port=${PORT}" >> "${RUN_DIR}/meta.env"
echo "metrics_url=${METRICS_URL}" >> "${RUN_DIR}/meta.env"
echo "container=${CONTAINER_NAME}" >> "${RUN_DIR}/meta.env"

NUMA_EXPORTER_PID=""
COLLECTOR_PID=""
cleanup_on_exit() {
  local status=$?
  if (( status != 0 )); then
    if [[ -n "${COLLECTOR_PID}" ]] && kill -0 "${COLLECTOR_PID}" >/dev/null 2>&1; then
      kill "${COLLECTOR_PID}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${NUMA_EXPORTER_PID}" ]] && kill -0 "${NUMA_EXPORTER_PID}" >/dev/null 2>&1; then
      kill "${NUMA_EXPORTER_PID}" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup_on_exit EXIT

if [[ -f "${LOG_ROOT}/collector.pid" ]]; then
  old_pid="$(cat "${LOG_ROOT}/collector.pid" || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" >/dev/null 2>&1; then
    kill "${old_pid}" >/dev/null 2>&1 || true
    sleep 1
  fi
fi

echo "[obs] starting local observability stack..."
docker compose -f "${OBS_DIR}/docker-compose.local.yml" up -d

echo "[obs] starting NUMA exporter loop..."
(
  exec 9>&-
  while true; do
    bash "${SCRIPT_DIR}/export_numa_metrics.sh"
    sleep 10
  done
) > "${RUN_DIR}/numa_exporter_loop.log" 2>&1 &
NUMA_EXPORTER_PID="$!"
echo "${NUMA_EXPORTER_PID}" > "${LOG_ROOT}/numa_exporter_loop.pid"

echo "[server] starting vLLM..."
CONTAINER_NAME="${CONTAINER_NAME}" PORT="${PORT}" bash "${REPO_ROOT}/start.sh"

run_warmup_stage() {
  local stage_name="$1"
  local users="$2"
  local requests="$3"
  local max_tokens="$4"
  local prompt_len="$5"
  local out_json="$6"

  echo "[server] warm-up ${stage_name}: users=${users}, requests=${requests}, max_tokens=${max_tokens}, prompt_len=${prompt_len}" | tee -a "${RUN_DIR}/warmup_stdout.log"
  docker exec -w "${BENCH_WORKDIR}" "${CONTAINER_NAME}" \
    "${PYTHON_BIN}" "${BENCH_SCRIPT}" \
    --url "http://127.0.0.1:${PORT}" \
    --users "${users}" \
    --requests "${requests}" \
    --max-tokens "${max_tokens}" \
    --prompt-len "${prompt_len}" \
    --model "${WARMUP_MODEL}" \
    --run-label "${WARMUP_LABEL}_${stage_name}_${RUN_ID}" \
    --output-json "${BENCH_WORKDIR}/benchmark_artifacts/${RUN_ID}/${out_json}" \
    | tee -a "${RUN_DIR}/warmup_stdout.log"
}

echo "[server] warm-up plan: stage A/B/C plus optional D/E/F"
echo "[server] warm-up plan: stage A/B/C plus optional D/E/F" > "${RUN_DIR}/warmup_stdout.log"

if [[ "${STAGE_A_ENABLED}" == "1" || "${STAGE_A_ENABLED}" == "true" || "${STAGE_A_ENABLED}" == "True" ]]; then
  run_warmup_stage "stage_a" "${STAGE_A_USERS}" "${STAGE_A_REQUESTS}" "${STAGE_A_MAX_TOKENS}" "${STAGE_A_PROMPT_LEN}" "warmup_stage_a_summary.json"
fi

if [[ "${STAGE_B_ENABLED}" == "1" || "${STAGE_B_ENABLED}" == "true" || "${STAGE_B_ENABLED}" == "True" ]]; then
  run_warmup_stage "stage_b" "${STAGE_B_USERS}" "${STAGE_B_REQUESTS}" "${STAGE_B_MAX_TOKENS}" "${STAGE_B_PROMPT_LEN}" "warmup_stage_b_summary.json"
fi

if [[ "${STAGE_C_ENABLED}" == "1" || "${STAGE_C_ENABLED}" == "true" || "${STAGE_C_ENABLED}" == "True" ]]; then
  run_warmup_stage "stage_c" "${STAGE_C_USERS}" "${STAGE_C_REQUESTS}" "${STAGE_C_MAX_TOKENS}" "${STAGE_C_PROMPT_LEN}" "warmup_stage_c_summary.json"
fi

if [[ "${STAGE_D_ENABLED}" == "1" || "${STAGE_D_ENABLED}" == "true" || "${STAGE_D_ENABLED}" == "True" ]]; then
  run_warmup_stage "stage_d" "${STAGE_D_USERS}" "${STAGE_D_REQUESTS}" "${STAGE_D_MAX_TOKENS}" "${STAGE_D_PROMPT_LEN}" "warmup_stage_d_summary.json"
fi

if [[ "${STAGE_E_ENABLED}" == "1" || "${STAGE_E_ENABLED}" == "true" || "${STAGE_E_ENABLED}" == "True" ]]; then
  run_warmup_stage "stage_e" "${STAGE_E_USERS}" "${STAGE_E_REQUESTS}" "${STAGE_E_MAX_TOKENS}" "${STAGE_E_PROMPT_LEN}" "warmup_stage_e_summary.json"
fi

if [[ "${STAGE_F_ENABLED}" == "1" || "${STAGE_F_ENABLED}" == "true" || "${STAGE_F_ENABLED}" == "True" ]]; then
  run_warmup_stage "stage_f" "${STAGE_F_USERS}" "${STAGE_F_REQUESTS}" "${STAGE_F_MAX_TOKENS}" "${STAGE_F_PROMPT_LEN}" "warmup_stage_f_summary.json"
fi

echo "[collector] starting background collectors..."
(
  exec 9>&-
  GPU_CSV="${RUN_DIR}/gpu_metrics.csv"
  DOCKER_CSV="${RUN_DIR}/docker_stats.csv"
  METRICS_SNAP="${RUN_DIR}/vllm_metrics_snapshots.prom"
  NUMA_SNAP="${RUN_DIR}/numa_snapshots.prom"
  CPU_SNAP="${RUN_DIR}/cpu_snapshots.log"

  echo "timestamp,index,name,utilization_gpu,utilization_mem,memory_total_mib,memory_used_mib,memory_free_mib,power_w,temperature_c" > "${GPU_CSV}"
  echo "timestamp,container,cpu_perc,mem_usage,mem_perc,net_io,block_io,pids" > "${DOCKER_CSV}"

  while true; do
    ts="$(date --iso-8601=seconds)"
    echo "# ts=${ts}" >> "${METRICS_SNAP}"
    curl -fsS "${METRICS_URL}" >> "${METRICS_SNAP}" 2>/dev/null || true

    echo "# ts=${ts}" >> "${NUMA_SNAP}"
    if [[ -f "${REPO_ROOT}/observability/node_exporter_textfile/numa.prom" ]]; then
      cat "${REPO_ROOT}/observability/node_exporter_textfile/numa.prom" >> "${NUMA_SNAP}"
    fi

    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,power.draw,temperature.gpu --format=csv,noheader,nounits >> "${GPU_CSV}" 2>/dev/null || true

    docker stats --no-stream --format "${ts},{{.Container}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}},{{.NetIO}},{{.BlockIO}},{{.PIDs}}" "${CONTAINER_NAME}" >> "${DOCKER_CSV}" 2>/dev/null || true

    echo "# ts=${ts}" >> "${CPU_SNAP}"
    cat /proc/loadavg >> "${CPU_SNAP}" 2>/dev/null || true
    grep -E "^(MemTotal|MemAvailable|SwapTotal|SwapFree):" /proc/meminfo >> "${CPU_SNAP}" 2>/dev/null || true

    sleep "${COLLECT_INTERVAL_S}"
  done
) > "${RUN_DIR}/collector.log" 2>&1 &

COLLECTOR_PID="$!"
echo "${COLLECTOR_PID}" > "${LOG_ROOT}/collector.pid"

echo "[done] started with warm-up and observability"
echo "[done] run dir: ${RUN_DIR}"
