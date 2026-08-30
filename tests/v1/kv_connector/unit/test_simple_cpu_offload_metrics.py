# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SimpleCPUOffloadConnector telemetry."""

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

from prometheus_client import Counter, Gauge, Histogram

from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
    SimpleCPUOffloadConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler


class RecordingMetric:
    instances = []

    def __init__(self, **kwargs):
        self.name = kwargs["name"]
        self.values = defaultdict(float)
        type(self).instances.append(self)

    def labels(self, *labelvalues):
        return RecordingMetricChild(self)


class RecordingMetricChild:
    def __init__(self, parent):
        self.parent = parent

    def inc(self, value):
        self.parent.values["counter"] += value

    def set(self, value):
        self.parent.values["gauge"] = value

    def observe(self, value):
        self.parent.values["histogram"] += value


def _metric_types():
    return {Gauge: RecordingMetric, Counter: RecordingMetric, Histogram: RecordingMetric}


def test_stats_aggregate_counters_and_replace_gauges():
    first = SimpleCPUOffloadConnectorStats(
        data={
            "offload_store_bytes": 10,
            "local_swap_out_bytes": 20,
            "offload_cpu_free_blocks": 8,
        }
    )
    second = SimpleCPUOffloadConnectorStats(
        data={
            "offload_store_bytes": 30,
            "local_swap_out_bytes": 40,
            "offload_cpu_free_blocks": 4,
        }
    )

    first.aggregate(second)

    assert first.reduce() == {
        "offload_store_bytes": 40,
        "local_swap_out_bytes": 60,
        "offload_cpu_free_blocks": 4,
    }


def test_prom_metrics_export_all_telemetry_types():
    RecordingMetric.instances.clear()
    config = SimpleNamespace(kv_transfer_config=None)
    metrics = SimpleCPUOffloadConnector.build_prom_metrics(
        config,
        _metric_types(),
        ["engine"],
        {0: ["0"]},
    )

    metrics.observe(
        {
            "offload_store_bytes": 128,
            "offload_cpu_free_blocks": 7,
            "local_bandwidth_gbps": 900.0,
        }
    )

    by_name = {metric.name: metric for metric in RecordingMetric.instances}
    assert (
        by_name["vllm:simple_cpu_offload_offload_store_bytes"].values["counter"]
        == 128
    )
    assert (
        by_name["vllm:simple_cpu_offload_offload_cpu_free_blocks"].values["gauge"]
        == 7
    )
    assert (
        by_name["vllm:simple_cpu_offload_local_bandwidth_gbps"].values["gauge"]
        == 900
    )


def test_telemetry_keys_cover_snapshot_fields():
    expected = {
        "offload_store_bytes",
        "local_swap_out_bytes",
        "remote_swap_in_time_ms",
        "local_bandwidth_gbps",
        "offload_cpu_cached_keys",
    }
    assert expected.issubset(set(SimpleCPUOffloadScheduler.telemetry_keys()))


def test_start_load_kv_delegates_to_worker():
    connector = object.__new__(SimpleCPUOffloadConnector)
    connector.worker_handler = Mock()

    connector.start_load_kv(None)

    connector.worker_handler.start_load_kv.assert_called_once_with()


def test_residency_state_counts_are_observable():
    manager = object.__new__(SimpleCPUOffloadScheduler)
    manager._residency = {}
    manager._gpu_block_pool = SimpleNamespace(
        blocks=[SimpleNamespace(block_hash=None) for _ in range(5)]
    )
    manager._record_gpu_blocks("req", ([3, 4],))
    manager._mark_residency_for_gpu_blocks([3], "store_in_flight", cpu_block_ids=[8])
    manager._mark_residency_for_gpu_blocks([4], "cpu_only", cpu_block_ids=[9])

    snapshot = manager.residency_snapshot()
    assert snapshot["offload_residency_store_in_flight_blocks"] == 1
    assert snapshot["offload_residency_cpu_only_blocks"] == 1


def test_residency_reconciliation_invalidates_evicted_cpu_block():
    class CacheIndex:
        def __init__(self):
            self.present = True

        def get_one_block(self, block_hash):
            return object() if self.present else None

    manager = object.__new__(SimpleCPUOffloadScheduler)
    manager._residency = {}
    manager.cpu_block_pool = SimpleNamespace(
        cached_block_hash_to_block=CacheIndex()
    )
    manager._gpu_block_pool = SimpleNamespace(
        blocks=[SimpleNamespace(block_hash=b"hash")]
    )
    manager._record_gpu_blocks("req", ([0],))
    manager._mark_residency_for_gpu_blocks(
        [0], "cpu_only", cpu_block_ids=[3]
    )

    manager.cpu_block_pool.cached_block_hash_to_block.present = False
    manager._reconcile_request_residency("req")

    record = manager._residency[("req", 0, 0)]
    assert record.state == "gpu_only"
    assert record.cpu_block_id is None


def test_residency_cleanup_removes_finished_request_records():
    manager = object.__new__(SimpleCPUOffloadScheduler)
    manager._residency = {}
    manager._residency["key"] = SimpleNamespace(
        request_id="finished",
        state="gpu_only",
    )
    manager._residency["inflight"] = SimpleNamespace(
        request_id="finished",
        state="store_in_flight",
    )

    manager._drop_residency_request("finished")

    assert "key" not in manager._residency
    assert "inflight" in manager._residency
