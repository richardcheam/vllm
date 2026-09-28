# SPDX-License-Identifier: Apache-2.0
"""Generic GPU/CPU locality discovery for the SuperInfer connector."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class GpuLocality:
    gpu_index: int
    uuid: str
    pci_bus_id: str
    numa_node: int | None
    island_id: int | None
    peer_indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class GpuCpuLocalityMapping:
    mode: str
    discovered: bool
    reason: str
    locality_complete: bool
    gpus: tuple[GpuLocality, ...]
    local_cpu_pool_fraction: float
    local_bandwidth_bytes_per_s: float | None
    remote_bandwidth_bytes_per_s: float | None

    @classmethod
    def disabled(cls) -> GpuCpuLocalityMapping:
        return cls(
            mode="disabled",
            discovered=False,
            reason="disabled",
            locality_complete=False,
            gpus=(),
            local_cpu_pool_fraction=1.0,
            local_bandwidth_bytes_per_s=None,
            remote_bandwidth_bytes_per_s=None,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "discovered": self.discovered,
            "locality_complete": self.locality_complete,
            "reason": self.reason,
            "gpu_count": len(self.gpus),
            "local_cpu_pool_fraction": self.local_cpu_pool_fraction,
            "local_bandwidth_bytes_per_s": self.local_bandwidth_bytes_per_s,
            "remote_bandwidth_bytes_per_s": self.remote_bandwidth_bytes_per_s,
            "gpus": [
                {
                    "gpu_index": gpu.gpu_index,
                    "uuid": gpu.uuid,
                    "pci_bus_id": gpu.pci_bus_id,
                    "numa_node": gpu.numa_node,
                    "island_id": gpu.island_id,
                    "peer_indices": list(gpu.peer_indices),
                }
                for gpu in self.gpus
            ],
        }


def _parse_gpu_rows(raw: str) -> list[tuple[int, str, str]]:
    rows = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            rank = int(parts[0])
        except ValueError:
            continue
        rows.append((rank, parts[1], parts[2]))
    return rows


def _parse_peer_matrix(
    raw: str, expected_ranks: set[int] | None = None
) -> dict[int, set[int]]:
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    lines = [ansi.sub("", line).strip() for line in raw.splitlines() if line.strip()]
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith("GPU") and "CPU Affinity" in line
        ),
        -1,
    )
    if header_index < 0:
        return {}
    header = [token for token in lines[header_index].split() if token.startswith("GPU")]
    ranks = [int(token[3:]) for token in header if token[3:].isdigit()]
    if expected_ranks is not None and set(ranks) != expected_ranks:
        return {}
    peers: dict[int, set[int]] = {}
    for line in lines[header_index + 1 :]:
        columns = line.split()
        if not columns or not columns[0].startswith("GPU"):
            continue
        try:
            source = int(columns[0][3:])
        except ValueError:
            continue
        if source not in ranks:
            continue
        if len(columns) < len(ranks) + 1 or columns[1 + ranks.index(source)] != "X":
            continue
        peers[source] = {
            target
            for index, target in enumerate(ranks)
            if index + 1 < len(columns)
            and target != source
            and (columns[index + 1].startswith("NV")
                 or columns[index + 1] in {"PIX", "PXB", "PHB"})
        }
    return peers


def _numa_for_pci(pci_bus_id: str) -> int | None:
    parts = pci_bus_id.strip().lower().split(":")
    if len(parts) != 3 or "." not in parts[2]:
        return None
    domain, bus, device_function = parts
    device, function = device_function.split(".", 1)
    domains = [domain[-4:].zfill(4)]
    candidates = [
        f"{candidate}:{bus[-2:].zfill(2)}:{device[-2:].zfill(2)}.{function}"
        for candidate in domains
    ]
    for candidate in candidates:
        try:
            with open(f"/sys/bus/pci/devices/{candidate}/numa_node") as stream:
                node = int(stream.read().strip())
        except (OSError, ValueError):
            continue
        return node if node >= 0 else None
    return None


def discover_gpu_cpu_locality(
    *,
    mode: str = "auto",
    local_cpu_pool_fraction: float = 0.75,
    local_bandwidth_bytes_per_s: float | None = None,
    remote_bandwidth_bytes_per_s: float | None = None,
) -> GpuCpuLocalityMapping:
    """Discover locality metadata without changing allocation policy."""
    if mode not in {"disabled", "auto", "required"}:
        raise ValueError(f"Unsupported topology mode: {mode}")
    if (
        not isfinite(local_cpu_pool_fraction)
        or not 0.0 <= local_cpu_pool_fraction <= 1.0
    ):
        raise ValueError("local_cpu_pool_fraction must be between 0 and 1")
    for name, bandwidth in (
        ("local_bandwidth_bytes_per_s", local_bandwidth_bytes_per_s),
        ("remote_bandwidth_bytes_per_s", remote_bandwidth_bytes_per_s),
    ):
        if bandwidth is not None and (not isfinite(bandwidth) or bandwidth <= 0):
            raise ValueError(f"{name} must be positive when specified")
    if mode == "disabled":
        return GpuCpuLocalityMapping.disabled()

    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        topo = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        if mode == "required":
            raise RuntimeError("GPU topology discovery failed") from exc
        return GpuCpuLocalityMapping(
            mode=mode,
            discovered=False,
            reason="nvidia-smi unavailable",
            locality_complete=False,
            gpus=(),
            local_cpu_pool_fraction=local_cpu_pool_fraction,
            local_bandwidth_bytes_per_s=local_bandwidth_bytes_per_s,
            remote_bandwidth_bytes_per_s=remote_bandwidth_bytes_per_s,
        )

    rows = _parse_gpu_rows(query.stdout) if query.returncode == 0 else []
    gpu_indices = {index for index, _, _ in rows}
    peers = (
        _parse_peer_matrix(topo.stdout, gpu_indices) if topo.returncode == 0 else {}
    )
    if not rows:
        if mode == "required":
            raise RuntimeError("GPU topology discovery returned no devices")
        return GpuCpuLocalityMapping(
            mode=mode,
            discovered=False,
            reason="no GPU query results",
            locality_complete=False,
            gpus=(),
            local_cpu_pool_fraction=local_cpu_pool_fraction,
            local_bandwidth_bytes_per_s=local_bandwidth_bytes_per_s,
            remote_bandwidth_bytes_per_s=remote_bandwidth_bytes_per_s,
        )

    numa_nodes = {index: _numa_for_pci(pci) for index, _, pci in rows}
    peers_complete = all(
        index in peers and peers[index].issubset(gpu_indices)
        for index, _, _ in rows
    )
    numa_complete = all(node is not None for node in numa_nodes.values())
    locality_complete = peers_complete and numa_complete
    if mode == "required" and not locality_complete:
        missing = []
        if not numa_complete:
            missing.append("NUMA mapping")
        if not peers_complete:
            missing.append("peer matrix")
        raise RuntimeError(
            "GPU locality discovery is incomplete: " + " and ".join(missing)
        )

    graph = {index: set(peers.get(index, set())) for index in gpu_indices}
    for index, linked_indices in tuple(graph.items()):
        for linked_index in linked_indices:
            graph.setdefault(linked_index, set()).add(index)

    islands: dict[int, int] = {}
    for index, _, _ in rows:
        if index in islands:
            continue
        component = {index}
        pending = [index]
        while pending:
            current = pending.pop()
            for peer in graph.get(current, set()):
                if peer not in component:
                    component.add(peer)
                    pending.append(peer)
        island_id = min(component)
        islands.update({member: island_id for member in component})

    gpus = tuple(
        GpuLocality(
            gpu_index=index,
            uuid=uuid,
            pci_bus_id=pci,
            numa_node=numa_nodes[index],
            island_id=islands.get(index) if locality_complete else None,
            peer_indices=tuple(sorted(peers.get(index, set()))),
        )
        for index, uuid, pci in rows
    )
    return GpuCpuLocalityMapping(
        mode=mode,
        discovered=locality_complete,
        reason="discovered" if locality_complete else "partial locality data",
        locality_complete=locality_complete,
        gpus=gpus,
        local_cpu_pool_fraction=local_cpu_pool_fraction,
        local_bandwidth_bytes_per_s=local_bandwidth_bytes_per_s,
        remote_bandwidth_bytes_per_s=remote_bandwidth_bytes_per_s,
    )
