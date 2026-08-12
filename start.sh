#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONTAINER_NAME="${CONTAINER_NAME:-richard-base-dev}"
PORT="${PORT:-8202}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
LOG_BASENAME="${LOG_BASENAME:-direct_A_control_topo_superinfer_8201}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-3600}"

mkdir -p "${LOG_DIR}"

if [[ -z "${LOG_NAME:-}" ]]; then
  LOG_NAME="$(basename "${LOG_DIR}")/${LOG_BASENAME}_$(date +%Y%m%d_%H%M%S).log"
fi
LOG_PATH="${SCRIPT_DIR}/${LOG_NAME}"

if docker exec "${CONTAINER_NAME}" pgrep -af "vllm serve .*--port ${PORT}" >/dev/null 2>&1; then
  echo "vLLM already running on port ${PORT} in ${CONTAINER_NAME}."
  echo "Health: http://127.0.0.1:${PORT}/health"
  exit 0
fi

echo "Starting A-profile server on port ${PORT}..."
bash "${SCRIPT_DIR}/scripts/launch_direct_deepseek_superinfer.sh" \
  --container "${CONTAINER_NAME}" \
  --profile superinfer \
  --port "${PORT}" \
  --gpu-memory-utilization 0.94 \
  --max-model-len 1048576 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 16384 \
  --num-speculative-tokens 0 \
  --block-size 256 \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash-0731 \
  --gh200-topology-tuned 1 \
  --enforce-eager 0 \
  --log-name "${LOG_NAME}"

echo "Waiting for readiness (timeout ${STARTUP_TIMEOUT_S}s)..."
deadline=$((SECONDS + STARTUP_TIMEOUT_S))
while (( SECONDS < deadline )); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "READY: http://127.0.0.1:${PORT}/health"
    echo "Log: ${LOG_PATH}"
    exit 0
  fi

  if [[ -f "${LOG_PATH}" ]] && grep -q "Traceback\|ERROR\|RuntimeError\|ValueError\|StrictDataclassClassValidationError" "${LOG_PATH}"; then
    echo "Startup failed. See log: ${LOG_PATH}" >&2
    exit 1
  fi

  sleep 2
done

echo "Timed out waiting for server readiness. See log: ${LOG_PATH}" >&2
exit 1
