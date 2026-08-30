# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GH200 topology discovery and locality helpers for simple KV offload."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TopologyGpuInfo:
    rank: int
    uuid: str
    pci_bus_id: str
    numa_node: int | None
    island_id: int
    p2p_peer_ranks: tuple[int, ...] = ()


@dataclass(frozen=True)
class Gh200TopologyMapping:
    enabled: bool
    discovered: bool
    reason: str
    gpus: tuple[TopologyGpuInfo, ...]
    local_cpu_pool_fraction: float
    local_bandwidth_bytes_per_s: float
    remote_bandwidth_bytes_per_s: float

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def fallback_mode(self) -> bool:
        return not self.discovered

    def island_for_rank(self, rank: int) -> int:
        for gpu in self.gpus:
            if gpu.rank == rank:
                return gpu.island_id
        return 0

    def local_fraction_for_rank(self, rank: int) -> float:
        if self.fallback_mode:
            return self.local_cpu_pool_fraction
        island = self.island_for_rank(rank)
        island_count = max(len({gpu.island_id for gpu in self.gpus}), 1)
        if island_count <= 1:
            return 1.0
        return min(max(self.local_cpu_pool_fraction, 0.0), 1.0)

    def to_log_lines(self) -> list[str]:
        lines = [
            (
                "GH200 topology mapping: "
                f"enabled={self.enabled} discovered={self.discovered} "
                f"reason={self.reason} gpu_count={self.gpu_count}"
            )
        ]
        for gpu in self.gpus:
            peers = ",".join(str(rank) for rank in gpu.p2p_peer_ranks) or "none"
            lines.append(
                "GH200 topology GPU "
                f"rank={gpu.rank} uuid={gpu.uuid} pci={gpu.pci_bus_id} "
                f"numa={gpu.numa_node} island={gpu.island_id} p2p={peers}"
            )
        return lines


def _run_cmd(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _detect_numa_for_pci_bus(pci_bus_id: str) -> int | None:
    # NVML may return 8-digit domains (e.g. 00000009:01:00.0) while sysfs
    # usually uses 4-digit domains (e.g. 0009:01:00.0).
    raw = pci_bus_id.strip().lower()
    parts = raw.split(":")

    candidates: list[str] = []
    if len(parts) == 3:
        domain, bus, dev_func = parts
        if "." in dev_func:
            dev, func = dev_func.split(".", 1)
            domain_candidates = [domain]
            if len(domain) == 8:
                domain_candidates.append(domain[-4:])
            elif len(domain) < 4:
                domain_candidates.append(domain.zfill(4))

            for dom in domain_candidates:
                dom_norm = dom[-4:].zfill(4)
                bus_norm = bus[-2:].zfill(2)
                dev_norm = dev[-2:].zfill(2)
                candidates.append(f"{dom_norm}:{bus_norm}:{dev_norm}.{func}")

    candidates.append(raw)
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        sysfs_path = f"/sys/bus/pci/devices/{candidate}/numa_node"
        try:
            with open(sysfs_path, "r", encoding="utf-8") as f:
                value = int(f.read().strip())
        except (FileNotFoundError, OSError, ValueError):
            continue
        if value < 0:
            return None
        return value
    return None


def _parse_gpu_query_csv(raw: str) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        rows.append((index, parts[1], parts[2]))
    return rows


def _parse_topo_matrix(raw: str) -> dict[int, set[int]]:
    peers: dict[int, set[int]] = {}
    if not raw:
        return peers
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    lines = [ansi_re.sub("", line).rstrip() for line in raw.splitlines() if line.strip()]
    header_idx = -1
    for i, line in enumerate(lines):
        normalized = " ".join(line.split())
        if normalized.startswith("GPU") and "CPU Affinity" in normalized:
            header_idx = i
            break
    if header_idx < 0 or header_idx + 1 >= len(lines):
        return peers

    header_tokens = [
        token
        for token in re.split(r"\s+", lines[header_idx].strip())
        if token.startswith("GPU")
    ]
    gpu_indices: list[int] = []
    for token in header_tokens:
        try:
            gpu_indices.append(int(token.replace("GPU", "")))
        except ValueError:
            continue

    for line in lines[header_idx + 1 :]:
        cols = re.split(r"\s+", line.strip())
        if not cols:
            continue
        label = cols[0]
        if not label.startswith("GPU"):
            continue
        try:
            src = int(label.replace("GPU", ""))
        except ValueError:
            continue
        src_peers: set[int] = set()
        for i, dst in enumerate(gpu_indices):
            cell_idx = i + 1
            if cell_idx >= len(cols):
                break
            cell = cols[cell_idx]
            if dst == src:
                continue
            # Treat NV links and direct P2P as same island hints.
            if cell.startswith("NV") or cell in {"PIX", "PXB", "PHB"}:
                src_peers.add(dst)
        peers[src] = src_peers
    return peers


def _assign_islands(
    gpu_rows: list[tuple[int, str, str]],
    p2p_peers: dict[int, set[int]],
) -> dict[int, int]:
    if not gpu_rows:
        return {}

    # Prefer NUMA if available; otherwise use peer graph connected components.
    numa_groups: dict[int, int] = {}
    next_island = 0
    for index, _, pci_bus in gpu_rows:
        numa = _detect_numa_for_pci_bus(pci_bus)
        if numa is None:
            continue
        if numa not in numa_groups:
            numa_groups[numa] = next_island
            next_island += 1

    islands: dict[int, int] = {}
    if numa_groups:
        for index, _, pci_bus in gpu_rows:
            numa = _detect_numa_for_pci_bus(pci_bus)
            if numa is None:
                islands[index] = 0
            else:
                islands[index] = numa_groups[numa]
        return islands

    # Graph-based fallback.
    visited: set[int] = set()
    for index, _, _ in gpu_rows:
        if index in visited:
            continue
        stack = [index]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            islands[node] = next_island
            for peer in p2p_peers.get(node, set()):
                if peer not in visited:
                    stack.append(peer)
        next_island += 1
    return islands


def discover_gh200_topology_mapping(
    *,
    enabled: bool,
    local_cpu_pool_fraction: float,
    local_bandwidth_bytes_per_s: float,
    remote_bandwidth_bytes_per_s: float,
) -> Gh200TopologyMapping:
    bounded_fraction = min(max(float(local_cpu_pool_fraction), 0.0), 1.0)
    if not enabled:
        return Gh200TopologyMapping(
            enabled=False,
            discovered=False,
            reason="disabled",
            gpus=(),
            local_cpu_pool_fraction=bounded_fraction,
            local_bandwidth_bytes_per_s=float(local_bandwidth_bytes_per_s),
            remote_bandwidth_bytes_per_s=float(remote_bandwidth_bytes_per_s),
        )

    raw_query = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id",
            "--format=csv,noheader",
        ]
    )
    if not raw_query:
        return Gh200TopologyMapping(
            enabled=True,
            discovered=False,
            reason="nvidia-smi query unavailable",
            gpus=(),
            local_cpu_pool_fraction=bounded_fraction,
            local_bandwidth_bytes_per_s=float(local_bandwidth_bytes_per_s),
            remote_bandwidth_bytes_per_s=float(remote_bandwidth_bytes_per_s),
        )

    gpu_rows = _parse_gpu_query_csv(raw_query)
    if not gpu_rows:
        return Gh200TopologyMapping(
            enabled=True,
            discovered=False,
            reason="no gpu rows parsed",
            gpus=(),
            local_cpu_pool_fraction=bounded_fraction,
            local_bandwidth_bytes_per_s=float(local_bandwidth_bytes_per_s),
            remote_bandwidth_bytes_per_s=float(remote_bandwidth_bytes_per_s),
        )

    topo_raw = _run_cmd(["nvidia-smi", "topo", "-m"]) or ""
    p2p_map = _parse_topo_matrix(topo_raw)
    islands = _assign_islands(gpu_rows, p2p_map)

    gpus: list[TopologyGpuInfo] = []
    for index, uuid, pci_bus in gpu_rows:
        numa = _detect_numa_for_pci_bus(pci_bus)
        island = islands.get(index, 0)
        peers = tuple(sorted(p2p_map.get(index, set())))
        gpus.append(
            TopologyGpuInfo(
                rank=index,
                uuid=uuid,
                pci_bus_id=pci_bus,
                numa_node=numa,
                island_id=island,
                p2p_peer_ranks=peers,
            )
        )

    # Respect CUDA_VISIBLE_DEVICES rank remap where set.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        remap: dict[int, int] = {}
        for local_rank, token in enumerate(cvd.split(",")):
            token = token.strip()
            if not token:
                continue
            try:
                remap[int(token)] = local_rank
            except ValueError:
                continue
        remapped: list[TopologyGpuInfo] = []
        for gpu in gpus:
            remapped_rank = remap.get(gpu.rank)
            if remapped_rank is None:
                continue
            remapped_peers = tuple(
                sorted(
                    remap[peer_rank]
                    for peer_rank in gpu.p2p_peer_ranks
                    if peer_rank in remap
                )
            )
            remapped.append(
                TopologyGpuInfo(
                    rank=remapped_rank,
                    uuid=gpu.uuid,
                    pci_bus_id=gpu.pci_bus_id,
                    numa_node=gpu.numa_node,
                    island_id=gpu.island_id,
                    p2p_peer_ranks=remapped_peers,
                )
            )
        if remapped:
            gpus = sorted(remapped, key=lambda item: item.rank)

    return Gh200TopologyMapping(
        enabled=True,
        discovered=True,
        reason="nvidia-smi + sysfs",
        gpus=tuple(sorted(gpus, key=lambda item: item.rank)),
        local_cpu_pool_fraction=bounded_fraction,
        local_bandwidth_bytes_per_s=float(local_bandwidth_bytes_per_s),
        remote_bandwidth_bytes_per_s=float(remote_bandwidth_bytes_per_s),
    )


@dataclass
class LocalityPoolPlanner:
    """Per-island local-first block ownership mapper.

    Keeps block ownership metadata without changing BlockPool allocation.
    """

    total_blocks: int
    enabled: bool
    mapping: Gh200TopologyMapping
    local_fraction: float = 0.75
    _owners: dict[int, int] = field(default_factory=dict)
    _placement_labels: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bounded_fraction = min(max(float(self.local_fraction), 0.0), 1.0)
        self.local_fraction = bounded_fraction
        self._island_ids = sorted({gpu.island_id for gpu in self.mapping.gpus})
        if not self._island_ids:
            self._island_ids = [0]
        self._island_capacity: dict[int, int] = {}
        per_island = self.total_blocks // len(self._island_ids)
        extra = self.total_blocks % len(self._island_ids)
        for i, island in enumerate(self._island_ids):
            self._island_capacity[island] = per_island + (1 if i < extra else 0)
        self._island_local_cap: dict[int, int] = {}
        self._island_local_used: dict[int, int] = {}
        self._island_remote_used: dict[int, int] = {}
        for island in self._island_ids:
            local_cap = int(self._island_capacity[island] * bounded_fraction)
            self._island_local_cap[island] = local_cap
            self._island_local_used[island] = 0
            self._island_remote_used[island] = 0

    @property
    def local_total_blocks(self) -> int:
        return sum(self._island_local_cap.values())

    @property
    def remote_total_blocks(self) -> int:
        return max(self.total_blocks - self.local_total_blocks, 0)

    @property
    def island_ids(self) -> list[int]:
        return list(self._island_ids)

    def island_for_gpu_rank(self, gpu_rank: int) -> int:
        return self.mapping.island_for_rank(gpu_rank)

    def assign_block_ids(
        self,
        cpu_block_ids: list[int],
        *,
        source_gpu_rank: int,
        req_id: str,
    ) -> tuple[list[str], int]:
        if not cpu_block_ids:
            return [], 0
        labels = self.plan_labels_for_allocation(
            num_blocks=len(cpu_block_ids),
            source_gpu_rank=source_gpu_rank,
        )
        remote_fallbacks = self.record_block_assignments(
            cpu_block_ids,
            labels=labels,
            source_gpu_rank=source_gpu_rank,
            req_id=req_id,
        )
        return labels, remote_fallbacks

    def plan_labels_for_allocation(
        self,
        *,
        num_blocks: int,
        source_gpu_rank: int,
    ) -> list[str]:
        if num_blocks <= 0:
            return []
        if not self.enabled:
            return ["local"] * num_blocks

        island = self.island_for_gpu_rank(source_gpu_rank)
        local_cap = self._island_local_cap.get(island, 0)
        local_used = self._island_local_used.get(island, 0)
        local_left = max(local_cap - local_used, 0)
        local_count = min(local_left, num_blocks)
        remote_count = max(num_blocks - local_count, 0)
        return ["local"] * local_count + ["remote"] * remote_count

    def record_block_assignments(
        self,
        cpu_block_ids: list[int],
        *,
        labels: list[str],
        source_gpu_rank: int,
        req_id: str,
    ) -> int:
        del req_id
        if not cpu_block_ids:
            return 0
        assert len(cpu_block_ids) == len(labels)

        island = self.island_for_gpu_rank(source_gpu_rank)
        remote_fallbacks = 0
        for block_id, label in zip(cpu_block_ids, labels):
            assigned_label = "local" if label != "remote" else "remote"
            self._owners[block_id] = island
            self._placement_labels[block_id] = assigned_label
            if assigned_label == "local":
                self._island_local_used[island] = self._island_local_used.get(island, 0) + 1
            else:
                self._island_remote_used[island] = self._island_remote_used.get(island, 0) + 1
                remote_fallbacks += 1
        return remote_fallbacks

    def labels_for_block_ids(
        self,
        cpu_block_ids: list[int],
        *,
        source_gpu_rank: int,
    ) -> list[str]:
        if not cpu_block_ids:
            return []
        if not self.enabled:
            return ["local"] * len(cpu_block_ids)

        source_island = self.island_for_gpu_rank(source_gpu_rank)
        labels: list[str] = []
        for block_id in cpu_block_ids:
            owner_island = self._owners.get(block_id)
            if owner_island is None:
                labels.append("local")
                continue
            if owner_island != source_island:
                labels.append("remote")
                continue
            labels.append(self._placement_labels.get(block_id, "local"))
        return labels

    def estimate_remote_blocks_for_gpu(self, *, source_gpu_rank: int, num_blocks: int) -> int:
        if num_blocks <= 0 or not self.enabled:
            return 0
        island = self.island_for_gpu_rank(source_gpu_rank)
        local_cap = self._island_local_cap.get(island, 0)
        local_used = self._island_local_used.get(island, 0)
        local_left = max(local_cap - local_used, 0)
        return max(num_blocks - local_left, 0)

    def release_block_ids(self, cpu_block_ids: list[int], *, source_gpu_rank: int) -> None:
        if not cpu_block_ids:
            return
        labels = self.labels_for_block_ids(cpu_block_ids, source_gpu_rank=source_gpu_rank)
        island = self.island_for_gpu_rank(source_gpu_rank)
        for block_id, label in zip(cpu_block_ids, labels):
            self._owners.pop(block_id, None)
            self._placement_labels.pop(block_id, None)
            if label == "local":
                self._island_local_used[island] = max(
                    self._island_local_used.get(island, 0) - 1,
                    0,
                )
            else:
                self._island_remote_used[island] = max(
                    self._island_remote_used.get(island, 0) - 1,
                    0,
                )

    def pool_sizes_per_island(self) -> dict[int, dict[str, int]]:
        return {
            island: {
                "capacity": self._island_capacity.get(island, 0),
                "local_cap": self._island_local_cap.get(island, 0),
            }
            for island in self._island_ids
        }


def estimate_remote_penalty_seconds(
    remote_bytes: int,
    *,
    local_bandwidth_bytes_per_s: float,
    remote_bandwidth_bytes_per_s: float,
) -> float:
    """Estimate extra delay for remote bytes vs local bytes."""
    if remote_bytes <= 0:
        return 0.0
    if local_bandwidth_bytes_per_s <= 0 or remote_bandwidth_bytes_per_s <= 0:
        return 0.0
    local_time = remote_bytes / local_bandwidth_bytes_per_s
    remote_time = remote_bytes / remote_bandwidth_bytes_per_s
    return max(remote_time - local_time, 0.0)


def split_elapsed_ms_by_bytes(
    elapsed_ms: float,
    *,
    local_bytes: int,
    remote_bytes: int,
) -> tuple[float, float]:
    """Split elapsed milliseconds proportionally by byte volume."""
    total = max(local_bytes + remote_bytes, 0)
    if total == 0 or elapsed_ms <= 0:
        return 0.0, 0.0
    local_ms = elapsed_ms * (local_bytes / total)
    remote_ms = elapsed_ms * (remote_bytes / total)
    return local_ms, remote_ms
