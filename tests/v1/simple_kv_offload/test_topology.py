# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.simple_kv_offload.topology import (
    Gh200TopologyMapping,
    LocalityPoolPlanner,
    TopologyGpuInfo,
    discover_gh200_topology_mapping,
    estimate_remote_penalty_seconds,
    split_elapsed_ms_by_bytes,
)


def _fake_mapping(discovered: bool = True) -> Gh200TopologyMapping:
    return Gh200TopologyMapping(
        enabled=True,
        discovered=discovered,
        reason="test",
        gpus=(
            TopologyGpuInfo(
                rank=0,
                uuid="GPU-0",
                pci_bus_id="0000:01:00.0",
                numa_node=0,
                island_id=0,
                p2p_peer_ranks=(1,),
            ),
            TopologyGpuInfo(
                rank=1,
                uuid="GPU-1",
                pci_bus_id="0000:81:00.0",
                numa_node=1,
                island_id=1,
                p2p_peer_ranks=(0,),
            ),
        ),
        local_cpu_pool_fraction=0.75,
        local_bandwidth_bytes_per_s=900.0,
        remote_bandwidth_bytes_per_s=300.0,
    )


def test_locality_pool_planner_local_first_with_remote_fallback() -> None:
    planner = LocalityPoolPlanner(
        total_blocks=10,
        enabled=True,
        mapping=_fake_mapping(),
        local_fraction=0.6,
    )

    labels, remote = planner.assign_block_ids(
        [0, 1, 2, 3, 4, 5, 6, 7],
        source_gpu_rank=0,
        req_id="req-1",
    )

    assert labels.count("local") == 3
    assert labels.count("remote") == 5
    assert remote == 5


def test_locality_pool_planner_disabled_marks_all_local() -> None:
    planner = LocalityPoolPlanner(
        total_blocks=10,
        enabled=False,
        mapping=_fake_mapping(discovered=False),
    )

    labels, remote = planner.assign_block_ids(
        [0, 1, 2, 3, 4],
        source_gpu_rank=0,
        req_id="req-2",
    )

    assert labels == ["local"] * 5
    assert remote == 0


def test_locality_pool_planner_labels_for_block_ids() -> None:
    planner = LocalityPoolPlanner(
        total_blocks=10,
        enabled=True,
        mapping=_fake_mapping(),
        local_fraction=0.4,
    )
    planner.assign_block_ids([0, 1, 2, 3], source_gpu_rank=0, req_id="req-a")
    planner.assign_block_ids([4, 5], source_gpu_rank=1, req_id="req-b")

    labels = planner.labels_for_block_ids([0, 1, 4, 5], source_gpu_rank=0)

    assert labels == ["local", "local", "remote", "remote"]


def test_locality_pool_planner_release_labels() -> None:
    planner = LocalityPoolPlanner(
        total_blocks=8,
        enabled=True,
        mapping=_fake_mapping(),
        local_fraction=0.5,
    )

    planner.assign_block_ids([0, 1, 2, 3, 4, 5], source_gpu_rank=0, req_id="req-x")
    assert planner._island_local_used[0] == 2
    assert planner._island_remote_used[0] == 4

    planner.release_block_ids([0, 1, 2, 3, 4, 5], source_gpu_rank=0)

    assert planner._island_local_used[0] == 0
    assert planner._island_remote_used[0] == 0


def test_discover_gh200_topology_mapping_fail_open(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm.v1.simple_kv_offload.topology._run_cmd",
        lambda _cmd: None,
    )

    mapping = discover_gh200_topology_mapping(
        enabled=True,
        local_cpu_pool_fraction=0.7,
        local_bandwidth_bytes_per_s=1000.0,
        remote_bandwidth_bytes_per_s=500.0,
    )

    assert mapping.enabled is True
    assert mapping.discovered is False
    assert mapping.fallback_mode is True


def test_discover_gh200_topology_mapping_parses_gpu_rows(monkeypatch) -> None:
    def fake_run(cmd):
        if cmd[:2] == ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id"]:
            return "0, GPU-0, 0000:01:00.0\n1, GPU-1, 0000:81:00.0"
        if cmd[:3] == ["nvidia-smi", "topo", "-m"]:
            return "GPU0 GPU1 CPU Affinity\nGPU0 X NV2 0-71\nGPU1 NV2 X 72-143"
        return ""

    monkeypatch.setattr("vllm.v1.simple_kv_offload.topology._run_cmd", fake_run)
    monkeypatch.setattr(
        "vllm.v1.simple_kv_offload.topology._detect_numa_for_pci_bus",
        lambda pci: 0 if "01:00.0" in pci else 1,
    )

    mapping = discover_gh200_topology_mapping(
        enabled=True,
        local_cpu_pool_fraction=0.75,
        local_bandwidth_bytes_per_s=900.0,
        remote_bandwidth_bytes_per_s=300.0,
    )

    assert mapping.discovered is True
    assert mapping.gpu_count == 2
    assert mapping.island_for_rank(0) != mapping.island_for_rank(1)


def test_discover_gh200_topology_mapping_remaps_peer_ranks_from_cvd(
    monkeypatch,
) -> None:
    def fake_run(cmd):
        if cmd[:2] == ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id"]:
            return "0, GPU-0, 0000:01:00.0\n1, GPU-1, 0000:81:00.0"
        if cmd[:3] == ["nvidia-smi", "topo", "-m"]:
            return "GPU0 GPU1 CPU Affinity\nGPU0 X NV2 0-71\nGPU1 NV2 X 72-143"
        return ""

    monkeypatch.setattr("vllm.v1.simple_kv_offload.topology._run_cmd", fake_run)
    monkeypatch.setattr(
        "vllm.v1.simple_kv_offload.topology._detect_numa_for_pci_bus",
        lambda _pci: 0,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")

    mapping = discover_gh200_topology_mapping(
        enabled=True,
        local_cpu_pool_fraction=0.75,
        local_bandwidth_bytes_per_s=900.0,
        remote_bandwidth_bytes_per_s=300.0,
    )

    assert mapping.discovered is True
    assert mapping.gpu_count == 2
    assert mapping.gpus[0].uuid == "GPU-1"
    assert mapping.gpus[1].uuid == "GPU-0"
    assert mapping.gpus[0].p2p_peer_ranks == (1,)
    assert mapping.gpus[1].p2p_peer_ranks == (0,)


def test_estimate_remote_penalty_seconds_positive() -> None:
    penalty = estimate_remote_penalty_seconds(
        1000,
        local_bandwidth_bytes_per_s=1000,
        remote_bandwidth_bytes_per_s=500,
    )
    assert penalty > 0


def test_parse_topo_matrix_handles_ansi_and_tabs() -> None:
    from vllm.v1.simple_kv_offload.topology import _parse_topo_matrix

    topo = (
        "\t\x1b[4mGPU0\tGPU1\tNIC0\tCPU Affinity\tNUMA Affinity\x1b[0m\n"
        "GPU0\t X \tNV18\tNODE\t0-71\t0\n"
        "GPU1\tNV18\t X \tSYS\t72-143\t1\n"
    )

    peers = _parse_topo_matrix(topo)
    assert peers[0] == {1}
    assert peers[1] == {0}


def test_split_elapsed_ms_by_bytes_proportional() -> None:
    local_ms, remote_ms = split_elapsed_ms_by_bytes(
        100.0,
        local_bytes=75,
        remote_bytes=25,
    )
    assert round(local_ms, 1) == 75.0
    assert round(remote_ms, 1) == 25.0


def test_detect_numa_for_pci_bus_normalizes_8_digit_domain(monkeypatch) -> None:
    from io import StringIO

    paths_opened: list[str] = []

    def fake_open(path, *_args, **_kwargs):
        paths_opened.append(path)
        if path.endswith("/0009:01:00.0/numa_node"):
            return StringIO("1\n")
        raise FileNotFoundError(path)

    monkeypatch.setattr("builtins.open", fake_open)

    from vllm.v1.simple_kv_offload.topology import _detect_numa_for_pci_bus

    numa = _detect_numa_for_pci_bus("00000009:01:00.0")

    assert numa == 1
    assert any(p.endswith("/0009:01:00.0/numa_node") for p in paths_opened)
