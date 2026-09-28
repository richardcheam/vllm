# SPDX-License-Identifier: Apache-2.0
"""Tests for generic SuperInfer locality discovery."""

from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.superinfer_topology import (
    GpuCpuLocalityMapping,
    _parse_gpu_rows,
    _parse_peer_matrix,
    discover_gpu_cpu_locality,
)


def test_parse_gpu_rows() -> None:
    rows = _parse_gpu_rows("0, GPU-a, 0000:01:00.0\n1, GPU-b, 0000:02:00.0\n")

    assert rows == [
        (0, "GPU-a", "0000:01:00.0"),
        (1, "GPU-b", "0000:02:00.0"),
    ]


def test_parse_peer_matrix() -> None:
    raw = """\
        GPU0 GPU1 CPU Affinity NUMA Affinity
        GPU0 X NV18 0
        GPU1 NV18 X 1
    """

    peers = _parse_peer_matrix(raw)

    assert peers == {0: {1}, 1: {0}}


def test_disabled_topology_is_neutral() -> None:
    mapping = discover_gpu_cpu_locality(mode="disabled")

    assert mapping == GpuCpuLocalityMapping.disabled()
    assert mapping.snapshot()["discovered"] is False


def test_manual_topology_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported topology mode"):
        discover_gpu_cpu_locality(mode="manual")


def test_required_topology_fails_on_missing_numa_mapping(monkeypatch) -> None:
    outputs = iter(
        [
            SimpleNamespace(
                returncode=0,
                stdout="0, GPU-a, 0000:01:00.0\n",
            ),
            SimpleNamespace(
                returncode=0,
                stdout="GPU0 CPU Affinity NUMA Affinity\nGPU0 X 0\n",
            ),
        ]
    )
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.superinfer_topology.subprocess.run",
        lambda *args, **kwargs: next(outputs),
    )
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.superinfer_topology._numa_for_pci",
        lambda _: None,
    )

    with pytest.raises(RuntimeError, match="NUMA mapping"):
        discover_gpu_cpu_locality(mode="required")


def test_complete_topology_reports_real_p2p_island_ids(monkeypatch) -> None:
    outputs = iter(
        [
            SimpleNamespace(
                returncode=0,
                stdout=(
                    "0, GPU-a, 0000:01:00.0\n"
                    "1, GPU-b, 0000:02:00.0\n"
                    "2, GPU-c, 0000:03:00.0\n"
                ),
            ),
            SimpleNamespace(
                returncode=0,
                stdout=(
                    "GPU0 GPU1 GPU2 CPU Affinity NUMA Affinity\n"
                    "GPU0 X NV18 SYS 0\n"
                    "GPU1 NV18 X SYS 0\n"
                    "GPU2 SYS SYS X 1\n"
                ),
            ),
        ]
    )
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.superinfer_topology.subprocess.run",
        lambda *args, **kwargs: next(outputs),
    )
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.superinfer_topology._numa_for_pci",
        lambda pci: 1 if "03" in pci else 0,
    )

    mapping = discover_gpu_cpu_locality(mode="required")

    assert mapping.locality_complete
    assert [gpu.island_id for gpu in mapping.gpus] == [0, 0, 2]
    assert [gpu.gpu_index for gpu in mapping.gpus] == [0, 1, 2]
    assert mapping.snapshot()["discovered"] is True
