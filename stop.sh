#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-richard-base-dev}"
PORT="${PORT:-8202}"

echo "Stopping vLLM server on port ${PORT} in ${CONTAINER_NAME}..."

PIDS="$(docker exec "${CONTAINER_NAME}" pgrep -f "vllm serve .*--port ${PORT}" || true)"
if [[ -z "${PIDS}" ]]; then
  echo "No vLLM process found on port ${PORT}."
  exit 0
fi

docker exec "${CONTAINER_NAME}" pkill -f "vllm serve .*--port ${PORT}" || true
sleep 3

if docker exec "${CONTAINER_NAME}" pgrep -f "vllm serve .*--port ${PORT}" >/dev/null 2>&1; then
  echo "Process still running, sending SIGKILL..."
  docker exec "${CONTAINER_NAME}" pkill -9 -f "vllm serve .*--port ${PORT}" || true
  sleep 1
fi

if docker exec "${CONTAINER_NAME}" pgrep -f "vllm serve .*--port ${PORT}" >/dev/null 2>&1; then
  echo "Failed to stop vLLM on port ${PORT}." >&2
  exit 1
fi

echo "Stopped vLLM on port ${PORT}."
