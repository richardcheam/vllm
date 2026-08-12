#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/daily_ops}"
PORT="${PORT:-8202}"
CONTAINER_NAME="${CONTAINER_NAME:-richard-base-dev}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

mkdir -p "${LOG_ROOT}"

if [[ -f "${LOG_ROOT}/collector.pid" ]]; then
  pid="$(cat "${LOG_ROOT}/collector.pid" || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    sleep 1
  fi
  rm -f "${LOG_ROOT}/collector.pid"
fi

if [[ -f "${LOG_ROOT}/numa_exporter_loop.pid" ]]; then
  pid="$(cat "${LOG_ROOT}/numa_exporter_loop.pid" || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    sleep 1
  fi
  rm -f "${LOG_ROOT}/numa_exporter_loop.pid"
fi

RUN_DIR=""
if [[ -f "${LOG_ROOT}/active_window_dir" ]]; then
  RUN_DIR="$(cat "${LOG_ROOT}/active_window_dir" || true)"
fi

if [[ -n "${RUN_DIR}" ]] && [[ -d "${RUN_DIR}" ]]; then
  echo "end_ts=$(date --iso-8601=seconds)" >> "${RUN_DIR}/meta.env"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/collect_daily_summary.py" "${RUN_DIR}" > "${RUN_DIR}/summary_paths.log" 2>&1 || true
fi

CONTAINER_NAME="${CONTAINER_NAME}" PORT="${PORT}" bash "${REPO_ROOT}/stop.sh"

echo "[done] service stopped and report generated"
