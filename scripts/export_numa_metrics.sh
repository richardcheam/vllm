#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/observability/node_exporter_textfile}"
OUT_FILE="${OUT_DIR}/numa.prom"
TMP_FILE="${OUT_FILE}.tmp"

mkdir -p "${OUT_DIR}"

{
  echo "# HELP numa_node_memfree_bytes Free memory per NUMA node in bytes"
  echo "# TYPE numa_node_memfree_bytes gauge"
  echo "# HELP numa_node_memtotal_bytes Total memory per NUMA node in bytes"
  echo "# TYPE numa_node_memtotal_bytes gauge"

  for meminfo in /sys/devices/system/node/node*/meminfo; do
    [[ -f "${meminfo}" ]] || continue
    node="$(basename "$(dirname "${meminfo}")" | sed 's/node//')"
    memfree_kb="$(grep -E "^Node[[:space:]]+${node}[[:space:]]+MemFree:" "${meminfo}" | tr -s ' ' | cut -d' ' -f4 || true)"
    memtotal_kb="$(grep -E "^Node[[:space:]]+${node}[[:space:]]+MemTotal:" "${meminfo}" | tr -s ' ' | cut -d' ' -f4 || true)"
    if [[ -n "${memfree_kb}" ]]; then
      echo "numa_node_memfree_bytes{node=\"${node}\"} $((memfree_kb * 1024))"
    fi
    if [[ -n "${memtotal_kb}" ]]; then
      echo "numa_node_memtotal_bytes{node=\"${node}\"} $((memtotal_kb * 1024))"
    fi
  done

  echo "# HELP numa_vmstat_total NUMA vmstat counters"
  echo "# TYPE numa_vmstat_total counter"
  for key in numa_hit numa_miss numa_foreign numa_interleave numa_local numa_other; do
    value="$(grep -E "^${key}[[:space:]]+" /proc/vmstat | tr -s ' ' | cut -d' ' -f2 || true)"
    if [[ -n "${value}" ]]; then
      echo "numa_${key}_total ${value}"
    fi
  done
} > "${TMP_FILE}"

mv "${TMP_FILE}" "${OUT_FILE}"
