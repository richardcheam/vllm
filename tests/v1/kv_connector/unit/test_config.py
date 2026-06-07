# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for KV cache offloading configuration."""

import pytest

from vllm.config import (
    CacheConfig,
    KVTransferConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def force_cpu_platform(monkeypatch: pytest.MonkeyPatch):
    import vllm.platforms as platforms
    from vllm.platforms.cpu import CpuPlatform

    monkeypatch.setattr(platforms, "current_platform", CpuPlatform(), raising=False)


@pytest.mark.parametrize(
    "kv_offloading_backend,kv_offloading_size,tp,pp,expected_backend,expected_bytes",
    [
        ("native", 4.0, 1, 1, "OffloadingConnector", 4.0 * (1 << 30)),
        # bytes per rank: 8.0 GiB / (2 * 2) = 2.0 GiB
        ("native", 8.0, 2, 2, "OffloadingConnector", 8.0 * (1 << 30)),
        ("lmcache", 4.0, 1, 1, "LMCacheConnectorV1", 4.0),
        # size per rank: 8.0 GiB / (2 * 2) = 2.0 GiB
        ("lmcache", 8.0, 2, 2, "LMCacheConnectorV1", 2.0),
        # When kv_offloading_size is None, offloading is disabled (backend is ignored)
        ("native", None, 1, 1, None, None),
    ],
)
def test_kv_connector(
    kv_offloading_backend, kv_offloading_size, tp, pp, expected_backend, expected_bytes
):
    kv_transfer_config = (
        KVTransferConfig(kv_connector_extra_config={"existing_key": "existing_value"})
        if expected_backend is not None
        else None
    )

    vllm_config = VllmConfig(
        cache_config=CacheConfig(
            kv_offloading_backend=kv_offloading_backend,
            kv_offloading_size=kv_offloading_size,
        ),
        kv_transfer_config=kv_transfer_config,
        parallel_config=ParallelConfig(
            tensor_parallel_size=tp, pipeline_parallel_size=pp
        ),
    )

    # No KV transfer config expected
    if expected_backend is None:
        assert vllm_config.kv_transfer_config is expected_backend
        return

    kv_transfer_config = vllm_config.kv_transfer_config
    kv_connector_extra_config = kv_transfer_config.kv_connector_extra_config

    assert kv_transfer_config.kv_connector == expected_backend
    assert kv_transfer_config.kv_role == "kv_both"

    if kv_offloading_backend == "native":
        assert kv_connector_extra_config["cpu_bytes_to_use"] == expected_bytes
        # Existing config should be preserved
        assert kv_connector_extra_config["existing_key"] == "existing_value"
    elif kv_offloading_backend == "lmcache":
        assert kv_connector_extra_config["lmcache.local_cpu"] is True
        assert kv_connector_extra_config["lmcache.max_local_cpu_size"] == expected_bytes
        # Existing config should be replaced
        assert "existing_key" not in kv_connector_extra_config


def test_kv_offloading_size_only_uses_native_default():
    """Test that setting only kv_offloading_size enables native offloading."""
    vllm_config = VllmConfig(
        cache_config=CacheConfig(
            kv_offloading_size=4.0,
            # kv_offloading_backend not set, should default to "native"
        ),
    )

    kv_transfer_config = vllm_config.kv_transfer_config
    kv_connector_extra_config = kv_transfer_config.kv_connector_extra_config
    assert kv_transfer_config.kv_connector == "OffloadingConnector"
    assert kv_transfer_config.kv_role == "kv_both"
    assert kv_connector_extra_config["cpu_bytes_to_use"] == 4.0 * (1 << 30)


def test_superinfer_knobs_default_off_do_not_enable_kv_connector():
    vllm_config = VllmConfig(
        cache_config=CacheConfig(),
        scheduler_config=SchedulerConfig(
            max_model_len=8192,
            is_encoder_decoder=False,
        ),
    )

    assert vllm_config.kv_transfer_config is None
    assert vllm_config.cache_config.swap_cpu_memory_gb is None
    assert vllm_config.scheduler_config.proactive_swap_budget == 0
    assert vllm_config.scheduler_config.vlt_alpha == 0.0
    assert vllm_config.scheduler_config.vlt_beta_bandwidth == 0.0
    assert vllm_config.scheduler_config.vlt_beta_future == 0.0
    assert vllm_config.scheduler_config.slo_ttft is None
    assert vllm_config.scheduler_config.slo_tbt is None


def test_swap_cpu_memory_gb_enables_simple_offload_connector():
    vllm_config = VllmConfig(
        cache_config=CacheConfig(
            swap_cpu_memory_gb=12.5,
        ),
    )

    kv_transfer_config = vllm_config.kv_transfer_config
    assert kv_transfer_config is not None
    assert kv_transfer_config.kv_connector == "SimpleCPUOffloadConnector"
    assert kv_transfer_config.kv_role == "kv_both"
    assert kv_transfer_config.kv_connector_extra_config["cpu_bytes_to_use"] == (
        12.5 * (1 << 30)
    )
    assert kv_transfer_config.kv_connector_extra_config["lazy_offload"] is True
    assert kv_transfer_config.kv_connector_extra_config["debug_single_request_swap"] is False


def test_swap_cpu_memory_gb_overrides_existing_connector_preserving_extra_config():
    vllm_config = VllmConfig(
        cache_config=CacheConfig(
            swap_cpu_memory_gb=2.0,
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExistingConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={"existing_key": "existing_value"},
        ),
    )

    kv_transfer_config = vllm_config.kv_transfer_config
    assert kv_transfer_config is not None
    assert kv_transfer_config.kv_connector == "SimpleCPUOffloadConnector"
    assert kv_transfer_config.kv_role == "kv_both"
    assert kv_transfer_config.kv_connector_extra_config["existing_key"] == (
        "existing_value"
    )
    assert kv_transfer_config.kv_connector_extra_config["cpu_bytes_to_use"] == (
        2.0 * (1 << 30)
    )
    assert kv_transfer_config.kv_connector_extra_config["lazy_offload"] is True


def test_swap_cpu_memory_gb_with_proactive_budget_enables_debug_single_request_swap():
    vllm_config = VllmConfig(
        cache_config=CacheConfig(
            swap_cpu_memory_gb=8.0,
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=8192,
            is_encoder_decoder=False,
            proactive_swap_budget=4,
        ),
    )

    kv_transfer_config = vllm_config.kv_transfer_config
    assert kv_transfer_config is not None
    assert kv_transfer_config.kv_connector == "SimpleCPUOffloadConnector"
    assert kv_transfer_config.kv_connector_extra_config["debug_single_request_swap"] is True


def test_swap_cpu_memory_gb_rejects_non_positive_values():
    with pytest.raises(ValueError, match="swap_cpu_memory_gb"):
        _ = VllmConfig(cache_config=CacheConfig(swap_cpu_memory_gb=0.0))
