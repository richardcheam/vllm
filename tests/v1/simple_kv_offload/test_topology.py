from vllm.v1.simple_kv_offload.layout import choose_layout_mode
from vllm.v1.simple_kv_offload.topology import (
    Gh200TopologyMapping,
    TopologyGpuInfo,
)


def test_deepseek_tp2_uses_gpu_derived_layout_by_default():
    decision = choose_layout_mode(
        True,
        model_arches=("DeepseekV4ForCausalLM",),
        num_kv_cache_groups=2,
        has_non_tensor_values=False,
        tensor_parallel_size=2,
    )
    assert decision.mode == "gpu_derived"
    assert not decision.block_first_eligible


def test_topology_mapping_reports_gpu_locality():
    mapping = Gh200TopologyMapping(
        enabled=True,
        discovered=True,
        reason="test",
        gpus=(
            TopologyGpuInfo(0, "gpu0", "0000:01:00.0", 0, 0, (1,)),
            TopologyGpuInfo(1, "gpu1", "0000:02:00.0", 1, 1, (0,)),
        ),
        local_cpu_pool_fraction=0.75,
        local_bandwidth_bytes_per_s=900 * (1 << 30),
        remote_bandwidth_bytes_per_s=280 * (1 << 30),
    )
    assert mapping.gpu_count == 2
    assert mapping.island_for_rank(0) == 0
    assert mapping.island_for_rank(1) == 1
