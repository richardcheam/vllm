#!/usr/bin/env bash
set -euo pipefail

# User-service entrypoint. systemd owns restart/backoff; this process only
# starts the server, observes it, captures failure evidence, and exits.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/config/reasoning-tree-superinfer.env}"
STATE_DIR="${VLLM_STATE_DIR:-${XDG_STATE_HOME:-${HOME}/.local/state}/deepseek-vllm}"
MAINTENANCE_HOLD="${STATE_DIR}/maintenance"
READY_MARKER="${STATE_DIR}/ready"
SUPERVISOR_LOG="${STATE_DIR}/supervisor.log"
HEALTH_INTERVAL_S="${SUPERVISOR_HEALTH_INTERVAL_S:-10}"
HEALTH_FAILURE_THRESHOLD="${SUPERVISOR_HEALTH_FAILURE_THRESHOLD:-3}"
COLLECT_DIAGNOSTICS="${SUPERVISOR_COLLECT_DIAGNOSTICS:-1}"
SUPERVISOR_STARTUP_TIMEOUT_S="${SUPERVISOR_STARTUP_TIMEOUT_S:-600}"
SUPERVISOR_CONTAINER_RECOVERY="${SUPERVISOR_CONTAINER_RECOVERY:-restart}"
SUPERVISOR_CONTAINER_RESTART_TIMEOUT_S="${SUPERVISOR_CONTAINER_RESTART_TIMEOUT_S:-30}"
SUPERVISOR_GPU_RECOVERY_WAIT_S="${SUPERVISOR_GPU_RECOVERY_WAIT_S:-15}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      if [[ $# -lt 2 ]]; then
        printf '%s\n' '--config requires a path' >&2
        exit 2
      fi
      CONFIG_FILE="$2"
      shift 2
      ;;
    -h|--help)
      printf 'Usage: %s [--config PATH]\n' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${CONFIG_FILE}" ]]; then
  printf 'Missing launch configuration: %s\n' "${CONFIG_FILE}" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${CONFIG_FILE}"
set +a

CONTAINER_NAME="${CONTAINER_NAME:?CONTAINER_NAME is required by the launch config}"
PORT="${PORT:?PORT is required by the launch config}"
REASONING_WORKDIR="${WORKDIR:-/workspace/re-SuperInfer/vllm-modern-reasoning}"

mkdir -p "${STATE_DIR}"
exec 9>"${STATE_DIR}/supervisor.lock"
if ! flock -n 9; then
  printf '[supervisor] another supervisor already holds the lock\n' >&2
  exit 1
fi

log() {
  local message="$1"
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${message}" \
    | tee -a "${SUPERVISOR_LOG}"
}

clear_ready_marker() {
  rm -f "${READY_MARKER}"
}

is_true() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "True" ]]
}

hold_active() {
  [[ -e "${MAINTENANCE_HOLD}" ]]
}

ensure_container_running() {
  local running
  running="$(docker inspect --format '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)"
  case "${running}" in
    true)
      return 0
      ;;
    false)
      log "container ${CONTAINER_NAME} is stopped; starting the existing container"
      docker start "${CONTAINER_NAME}" >/dev/null
      ;;
    *)
      log "container ${CONTAINER_NAME} was not found or cannot be inspected"
      return 1
      ;;
  esac
}

server_process_present() {
  docker exec "${CONTAINER_NAME}" \
    pgrep -f "vllm.*serve.*--port ${PORT}" >/dev/null 2>&1
}

server_health_ok() {
  curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1
}

container_gpu_ready() {
  docker exec "${CONTAINER_NAME}" nvidia-smi -L >/dev/null 2>&1
}

recover_container_gpu() {
  if ! is_true "${SUPERVISOR_CONTAINER_RECOVERY}"; then
    log "container recovery disabled; leaving container unchanged"
    return 0
  fi
  if container_gpu_ready; then
    log "container GPU/NVML check is healthy; no container restart needed"
    return 0
  fi
  log "container GPU/NVML check failed; restarting dedicated container"
  if ! docker restart -t "${SUPERVISOR_CONTAINER_RESTART_TIMEOUT_S}" "${CONTAINER_NAME}" >/dev/null; then
    log "dedicated container restart failed"
    return 1
  fi
  sleep "${SUPERVISOR_GPU_RECOVERY_WAIT_S}"
  if container_gpu_ready; then
    log "container GPU/NVML recovered after restart"
    return 0
  fi
  log "container GPU/NVML remains unavailable after restart"
  return 1
}

latest_server_log() {
  local log_dir="${LOG_DIR:-${REPO_ROOT}/logs}"
  local basename="${LOG_BASENAME:-direct_reasoning_superinfer_8202}"
  local candidate
  for candidate in "${log_dir}/${basename}"_*.log; do
    [[ -f "${candidate}" ]] || continue
    printf '%s\n' "${candidate}"
    return 0
  done
  return 1
}

collect_failure_evidence() {
  local server_log=""
  if ! is_true "${COLLECT_DIAGNOSTICS}"; then
    return 0
  fi
  server_log="$(latest_server_log || true)"
  log "collecting read-only failure diagnostics"
  CONTAINER_NAME="${CONTAINER_NAME}" \
    PORT="${PORT}" \
    SERVER_LOG="${server_log}" \
    DOCKER_LOG_SINCE="${DIAGNOSTICS_DOCKER_LOG_SINCE:-15m}" \
    LOG_TAIL_LINES="${DIAGNOSTICS_LOG_TAIL_LINES:-200}" \
    OUTPUT_DIR="${REPO_ROOT}/failure_artifacts/incident_$(date -u +%Y%m%dT%H%M%SZ)" \
    bash "${REPO_ROOT}/scripts/collect_vllm_failure_diagnostics.sh" || true
}

stop_server() {
  VLLM_SUPERVISED_CHILD=1 \
    CONTAINER_NAME="${CONTAINER_NAME}" PORT="${PORT}" \
    REASONING_WORKDIR="${REASONING_WORKDIR}" \
    bash "${REPO_ROOT}/stop.sh" || true
}

handle_signal() {
  log "supervisor received termination signal; stopping the scoped server"
  clear_ready_marker
  stop_server
  exit 0
}

trap handle_signal TERM INT

if hold_active; then
  log "maintenance hold is active; automatic start is disabled"
  clear_ready_marker
  exit 0
fi

if ! ensure_container_running; then
  log "unable to ensure container availability"
  exit 1
fi

log "starting DeepSeek vLLM service"
if ! VLLM_SUPERVISED_CHILD=1 \
    STARTUP_TIMEOUT_S="${SUPERVISOR_STARTUP_TIMEOUT_S}" \
    COLLECT_START_FAILURE_DIAGNOSTICS=1 \
    bash "${REPO_ROOT}/start.sh" --config "${CONFIG_FILE}"; then
  log "start.sh failed"
  clear_ready_marker
  collect_failure_evidence
  stop_server
  recover_container_gpu || true
  exit 1
fi

if hold_active; then
  log "maintenance hold was enabled during startup; leaving service stopped"
  clear_ready_marker
  stop_server
  exit 0
fi

health_failures=0
touch "${READY_MARKER}"
log "server is ready; entering supervised health/process loop"
while true; do
  if hold_active; then
    log "maintenance hold detected; stopping automatic service"
    clear_ready_marker
    stop_server
    exit 0
  fi

  if ! server_process_present; then
    log "vLLM process tree disappeared unexpectedly"
    clear_ready_marker
    collect_failure_evidence
    stop_server
    recover_container_gpu || true
    exit 1
  fi

  if server_health_ok; then
    health_failures=0
  else
    health_failures=$((health_failures + 1))
    log "health check failed (${health_failures}/${HEALTH_FAILURE_THRESHOLD})"
    if (( health_failures >= HEALTH_FAILURE_THRESHOLD )); then
      log "health failure threshold reached"
      clear_ready_marker
      collect_failure_evidence
      stop_server
      recover_container_gpu || true
      exit 1
    fi
  fi

  sleep "${HEALTH_INTERVAL_S}"
done
