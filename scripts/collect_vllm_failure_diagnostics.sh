#!/usr/bin/env bash
set -euo pipefail

# Read-only incident snapshot. It intentionally does not stop processes or
# restart the container, so it can be run before rollback after a worker death.
CONTAINER_NAME="${CONTAINER_NAME:-richard-base-dev-sysnice}"
PORT="${PORT:-8202}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/failure_artifacts/incident_$(date -u +%Y%m%dT%H%M%SZ)}"
SERVER_LOG="${SERVER_LOG:-}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-200}"

mkdir -p "${OUTPUT_DIR}"

run_capture() {
  local output_file="$1"
  shift
  "$@" >"${OUTPUT_DIR}/${output_file}" 2>&1 || true
}

run_container_capture() {
  local output_file="$1"
  shift
  run_capture "${output_file}" docker exec "${CONTAINER_NAME}" bash -lc "$*"
}

run_capture snapshot_metadata bash -c \
  "printf 'collected_at_utc=%s\\n' \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\"; printf 'host=%s\\n' \"\$(hostname)\"; printf 'container=%s\\n' '${CONTAINER_NAME}'; printf 'port=%s\\n' '${PORT}'"
run_capture host_gpu_status \
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free \
  --format=csv,noheader,nounits
run_capture host_compute_processes \
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits
run_capture host_processes ps -eo pid,ppid,stat,etime,cmd
run_capture port_status ss -ltnp
run_capture container_state docker inspect --format \
  'name={{.Name}} image={{.Config.Image}} image_id={{.Image}} status={{.State.Status}} running={{.State.Running}} started={{.State.StartedAt}} finished={{.State.FinishedAt}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} restart_count={{.RestartCount}} restart_policy={{json .HostConfig.RestartPolicy}} runtime={{.HostConfig.Runtime}}' \
  "${CONTAINER_NAME}"
run_container_capture container_processes \
  'ps -eo pid,ppid,stat,etime,comm,args; printf "--- vLLM ancestry ---\n"; for pid in $(pgrep -f "vllm.*serve.*--port ${PORT}" || true); do echo "root_pid=${pid}"; pstree -aps "${pid}" 2>/dev/null || true; done'
run_container_capture container_process_status \
  'for proc in /proc/[0-9]*; do pid=${proc##*/}; comm=$(tr -d "\0" < "${proc}/comm" 2>/dev/null || true); case "${comm}" in python|VLLM::*) printf "pid=%s ppid=%s stat=" "${pid}" "$(awk "{print \$4}" "${proc}/stat" 2>/dev/null || true)"; tr "\0" " " < "${proc}/cmdline" 2>/dev/null; printf "\n";; esac; done'
run_container_capture container_gpu_status \
  'nvidia-smi -L; nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits'
run_container_capture container_cuda_check \
  'printf "CUDA_ARCH_LIST=%s\nMINSMVER=%s\nMAXSMVER=%s\nNVIDIA_DISABLE_REQUIREMENTS=%s\n" "${CUDA_ARCH_LIST-}" "${MINSMVER-}" "${MAXSMVER-}" "${NVIDIA_DISABLE_REQUIREMENTS-}"; nvidia-smi -L; python3 - <<"PY"
import torch
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"device_count={torch.cuda.device_count()}")
PY'
run_container_capture container_shared_memory \
  'df -h /dev/shm; df -i /dev/shm; ls -la /dev/shm; printf "--- open vLLM shm fds ---\n"; for proc in /proc/[0-9]*; do pid=${proc##*/}; comm=$(tr -d "\0" < "${proc}/comm" 2>/dev/null || true); case "${comm}" in python|VLLM::*) ls -l "${proc}/fd" 2>/dev/null | grep -E "/dev/shm|sem" | sed "s#^#pid=${pid} comm=${comm} #" || true;; esac; done'
run_capture container_kernel_messages docker exec "${CONTAINER_NAME}" \
  dmesg --ctime
run_capture host_kernel_gpu_messages bash -lc \
  'dmesg --ctime 2>/dev/null | grep -Ei "NVRM|Xid|NVML|CUDA|out of memory|oom|GPU has fallen" || true'
run_capture endpoint_health curl -fsS -m 5 "http://127.0.0.1:${PORT}/health"
run_capture endpoint_metrics curl -fsS -m 10 "http://127.0.0.1:${PORT}/metrics"

if [[ -n "${SERVER_LOG}" && -f "${SERVER_LOG}" ]]; then
  run_capture server_log_tail bash -lc "tail -n '${LOG_TAIL_LINES}' '${SERVER_LOG}'"
fi

run_capture docker_log_tail docker logs --since "${DOCKER_LOG_SINCE:-10m}" "${CONTAINER_NAME}"

printf 'Failure diagnostics written to %s\n' "${OUTPUT_DIR}"
