#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    -h|--help)
      printf 'Usage: %s [ENV_FILE] [--env-file PATH]\n' "${BASH_SOURCE[0]}"
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

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

STATE_DIR="${STATE_DIR:-${ROOT_DIR}/.run}"
TIMEOUT_S="${SHUTDOWN_TIMEOUT_S:-60}"
PID_FILE="${STATE_DIR}/server.pid"
PGID_FILE="${STATE_DIR}/server.pgid"
PROFILE_FILE="${STATE_DIR}/server.profile"

if [[ ! -s "${PID_FILE}" ]]; then
  printf 'No server pid file: %s\n' "${PID_FILE}"
  exit 0
fi

pid="$(<"${PID_FILE}")"
pgid="${pid}"
[[ -s "${PGID_FILE}" ]] && pgid="$(<"${PGID_FILE}")"

if ! kill -0 "${pid}" 2>/dev/null; then
  rm -f "${PID_FILE}" "${PGID_FILE}" "${PROFILE_FILE}"
  printf 'Removed stale server state.\n'
  exit 0
fi

printf 'Stopping server process group pgid=%s pid=%s\n' "${pgid}" "${pid}"
kill -TERM -- "-${pgid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true

deadline=$((SECONDS + TIMEOUT_S))
last_report=0
while (( SECONDS < deadline )); do
  group_alive=0
  if kill -0 -- "-${pgid}" 2>/dev/null; then
    group_alive=1
  fi
  if (( ! group_alive )); then
    rm -f "${PID_FILE}" "${PGID_FILE}" "${PROFILE_FILE}"
    printf 'Server stopped cleanly.\n'
    exit 0
  fi
  if (( SECONDS - last_report >= 5 )); then
    last_report=${SECONDS}
    printf 'Waiting for process group shutdown (%ss remaining): ' \
      "$((deadline - SECONDS))"
    ps -o pid=,stat=,wchan=:24,etime=,cmd= -g "${pgid}" 2>/dev/null \
      | head -n 6 || true
  fi
  sleep 1
done

printf 'Graceful stop timed out; sending SIGKILL to process group.\n' >&2
kill -KILL -- "-${pgid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
kill -KILL "${pid}" 2>/dev/null || true
sleep 3
rm -f "${PID_FILE}" "${PGID_FILE}" "${PROFILE_FILE}"
printf 'Server process state removed.\n'
