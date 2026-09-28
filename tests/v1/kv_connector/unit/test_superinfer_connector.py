# SPDX-License-Identifier: Apache-2.0
"""Tests for the opt-in SuperInfer V1 connector entry point."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.superinfer_connector import (
    SuperInferConnector,
    prepare_superinfer_extra_config,
    validate_superinfer_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.superinfer_policy import (
    SuperInferVLTInputs,
    compute_vlt_score,
    rank_vlt_candidates,
)
from vllm.distributed.kv_transfer.kv_connector.v1.superinfer_stats import (
    SuperInferConnectorStats,
)
from vllm.v1.simple_kv_offload.manager import (
    BoundaryStoreStats,
    SimpleCPUOffloadScheduler,
)
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker


def test_default_superinfer_config_is_valid() -> None:
    validate_superinfer_config({})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("copy_backend", "unknown"),
        ("copy_backend", "inline"),
        ("capacity_gb", 0),
    ],
)
def test_invalid_superinfer_config_is_rejected(key: str, value) -> None:
    with pytest.raises(ValueError):
        validate_superinfer_config({key: value})


def test_nested_superinfer_config_is_valid() -> None:
    validate_superinfer_config(
        {
            "schema_version": 1,
            "capacity_gb": 128,
            "copy_backend": "dma",
            "allocation": {"mode": "empty"},
            "topology": {"mode": "auto"},
            "policy": {"mode": "conservative", "proactive_budget": 0},
        }
    )


def test_v030_connector_rejects_unimplemented_copy_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported SuperInfer copy_backend"):
        validate_superinfer_config({"copy_backend": "native_dma"})


@pytest.mark.parametrize(
    "config",
    [
        {"transfer_queue_depth": 8},
        {"metrics": {"mode": "basic"}},
        {"policy": {"mode": "proactive", "proactive_budget": 1}},
        {"policy": {"mode": "high_risk"}},
    ],
)
def test_unimplemented_superinfer_options_fail_closed(config: dict) -> None:
    with pytest.raises(ValueError):
        validate_superinfer_config(config)


def test_capacity_conflict_is_rejected() -> None:
    with pytest.raises(ValueError, match="Specify only one"):
        validate_superinfer_config({"capacity_gb": 128, "cpu_bytes_to_use": 1024})


def test_capacity_conflict_with_per_rank_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="Specify only one"):
        validate_superinfer_config(
            {"capacity_gb": 128, "cpu_bytes_to_use_per_rank": 1024}
        )


def test_capacity_and_allocation_translate_to_cpu_connector_config() -> None:
    prepared = prepare_superinfer_extra_config(
        {"capacity_gb": 2, "allocation": {"mode": "empty"}}
    )

    assert prepared["cpu_bytes_to_use"] == 2 * 1024**3
    assert prepared["cpu_kv_allocation_mode"] == "empty"
    assert prepared["kv_offload_backend"] == "cpu"


def test_auto_allocation_explicitly_maps_to_zero_filled() -> None:
    prepared = prepare_superinfer_extra_config({"allocation": {"mode": "auto"}})

    assert prepared["cpu_kv_allocation_mode"] == "zero"


def test_connector_constructor_applies_translated_config(monkeypatch) -> None:
    captured = {}

    def fake_base_init(self, vllm_config, role, kv_cache_config):
        captured["extra_config"] = (
            vllm_config.kv_transfer_config.kv_connector_extra_config
        )
        self.scheduler_manager = None
        self.worker_handler = None

    monkeypatch.setattr(SimpleCPUOffloadConnector, "__init__", fake_base_init)
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "capacity_gb": 2,
                "allocation": {"mode": "empty"},
                "topology": {"mode": "disabled"},
            }
        )
    )

    connector = SuperInferConnector(config, role=None, kv_cache_config=None)

    assert captured["extra_config"]["cpu_bytes_to_use"] == 2 * 1024**3
    assert captured["extra_config"]["cpu_kv_allocation_mode"] == "empty"
    assert connector.get_superinfer_capabilities()["proactive_policy_active"] is False


def test_superinfer_stats_aggregate_and_reduce() -> None:
    first = SuperInferConnectorStats()
    first.record({"store_bytes": 10, "pending_store_events": 1})
    second = SuperInferConnectorStats()
    second.record({"store_bytes": 20, "pending_store_events": 0})

    first.aggregate(second)

    assert first.reduce()["store_bytes"] == 30
    assert first.reduce()["pending_store_events"] == 0


def test_superinfer_telemetry_snapshot_shape() -> None:
    expected = {
        "store_events",
        "load_events",
        "store_bytes",
        "load_bytes",
        "cpu_lookup_requests",
        "cpu_lookup_hits",
        "cpu_lookup_hit_tokens",
        "pending_store_events",
        "pending_store_reqs",
        "pending_load_reqs",
        "cpu_total_blocks",
        "cpu_free_blocks",
        "cpu_used_blocks",
        "boundary_stores_published",
        "boundary_stores_stored",
        "boundary_stores_dropped_cpu_full",
        "boundary_stores_skipped_cached",
    }

    stats = SuperInferConnectorStats()
    stats.record({name: 1 for name in expected})
    assert set(stats.reduce()) == expected


def test_boundary_cumulative_counters_are_counted_once_per_interval() -> None:
    manager = SimpleCPUOffloadScheduler.__new__(SimpleCPUOffloadScheduler)
    manager.cpu_block_pool = MagicMock()
    manager.cpu_block_pool.get_num_free_blocks.return_value = 5
    manager.num_cpu_blocks = 8
    manager._store_event_to_blocks = {}
    manager._reqs_to_store = {}
    manager._reqs_to_load = {}
    manager._telemetry_store_events = 0
    manager._telemetry_load_events = 0
    manager._telemetry_store_bytes = 0
    manager._telemetry_load_bytes = 0
    manager._telemetry_cpu_lookup_requests = 0
    manager._telemetry_cpu_lookup_hits = 0
    manager._telemetry_cpu_lookup_hit_tokens = 0
    manager.boundary_store_stats = BoundaryStoreStats(published=3, stored=2)
    manager._telemetry_boundary_last = BoundaryStoreStats()

    first = manager.take_superinfer_telemetry()
    second = manager.take_superinfer_telemetry()

    assert first["boundary_stores_published"] == 3
    assert first["boundary_stores_stored"] == 2
    assert second["boundary_stores_published"] == 0
    assert second["boundary_stores_stored"] == 0


def test_worker_transfer_byte_telemetry_uses_registered_region_size() -> None:
    worker = SimpleCPUOffloadWorker.__new__(SimpleCPUOffloadWorker)
    worker.bytes_per_block = 64
    worker._telemetry_store_events = 0
    worker._telemetry_load_events = 0
    worker._telemetry_store_bytes = 0
    worker._telemetry_load_bytes = 0

    worker._record_transfer(is_store=True, num_blocks=2)
    worker._record_transfer(is_store=False, num_blocks=1)
    telemetry = worker.take_superinfer_telemetry()

    assert telemetry == {
        "store_events": 1,
        "load_events": 1,
        "store_bytes": 128,
        "load_bytes": 64,
    }


def test_vlt_ranking_prefers_higher_predicted_slo_penalty() -> None:
    scores = rank_vlt_candidates(
        {
            "within_slo": SuperInferVLTInputs(predicted_ttft_s=1.0),
            "over_slo": SuperInferVLTInputs(predicted_ttft_s=10.0),
        }
    )

    assert [item.request_id for item in scores] == ["over_slo", "within_slo"]
    assert compute_vlt_score(SuperInferVLTInputs(predicted_ttft_s=1.0)) == 0


def test_superinfer_connector_is_lazy_registered() -> None:
    connector_cls = KVConnectorFactory.get_connector_class_by_name(
        "SuperInferConnector"
    )

    assert connector_cls.__name__ == "SuperInferConnector"


def test_superinfer_connector_preserves_hma_capability() -> None:
    connector_cls = KVConnectorFactory.get_connector_class_by_name(
        "SuperInferConnector"
    )

    assert issubclass(connector_cls, SupportsHMA)
