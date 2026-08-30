#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATRIX_OUTPUT_DIR="${MATRIX_OUTPUT_DIR:-${ROOT_DIR}/benchmark_artifacts/matrix}"
MATRIX_ID="${MATRIX_ID:-$(date +%Y%m%d_%H%M%S)}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  printf 'Usage: %s [PROFILE_ENV_FILE ...]\n' "${BASH_SOURCE[0]}"
  printf 'Default profiles: .env.vanilla .env.native-offload .env.superinfer .env.superinfer-high-risk\n'
  exit 0
fi

if (( $# > 0 )); then
  PROFILES=("$@")
else
  PROFILES=(.env.vanilla .env.native-offload .env.superinfer .env.superinfer-high-risk)
fi
MATRIX_DIR="${MATRIX_OUTPUT_DIR}/${MATRIX_ID}"
mkdir -p "${MATRIX_DIR}"

active_env=""
cleanup() {
  if [[ -n "${active_env}" ]]; then
    "${ROOT_DIR}/scripts/stop_superinfer.sh" "${active_env}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

status=0
for profile_path in "${PROFILES[@]}"; do
  if [[ "${profile_path}" != /* ]]; then
    profile_path="${ROOT_DIR}/${profile_path}"
  fi
  if [[ ! -f "${profile_path}" ]]; then
    printf 'Missing profile: %s\n' "${profile_path}" >&2
    status=1
    continue
  fi

  profile_name="$(basename "${profile_path}" .env)"
  profile_dir="${MATRIX_DIR}/${profile_name}"
  mkdir -p "${profile_dir}"
  active_env="${profile_path}"
  printf '\n=== Starting profile %s ===\n' "${profile_name}"

  if "${ROOT_DIR}/scripts/launch_superinfer.sh" "${profile_path}"; then
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
      --format=csv,noheader >"${profile_dir}/gpu_before_bench.csv" 2>/dev/null || true
    if "${ROOT_DIR}/scripts/run_superinfer_bench.sh" "${profile_path}" \
      --no-start --output-dir "${profile_dir}" --run-id measurement; then
      printf 'Profile %s completed successfully.\n' "${profile_name}"
    else
      printf 'Profile %s benchmark failed.\n' "${profile_name}" >&2
      status=1
    fi
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
      --format=csv,noheader >"${profile_dir}/gpu_after_bench.csv" 2>/dev/null || true
  else
    printf 'Profile %s startup failed.\n' "${profile_name}" >&2
    status=1
  fi

  printf '=== Stopping profile %s ===\n' "${profile_name}"
  if ! "${ROOT_DIR}/scripts/stop_superinfer.sh" "${profile_path}"; then
    printf 'Profile %s shutdown failed.\n' "${profile_name}" >&2
    status=1
  fi
  active_env=""
done

MATRIX_PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
"${MATRIX_PYTHON}" "${ROOT_DIR}/scripts/report_profile_matrix.py" "${MATRIX_DIR}" || status=1
printf '\nMatrix output: %s\n' "${MATRIX_DIR}"
exit "${status}"
