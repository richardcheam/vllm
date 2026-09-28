# SPDX-License-Identifier: Apache-2.0
"""Typed telemetry for the optional SuperInfer connector."""

from dataclasses import dataclass
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.metrics.utils import create_metric_per_engine

_COUNTERS = (
    ("store_events", "SuperInfer KV store events."),
    ("load_events", "SuperInfer KV load events."),
    ("store_bytes", "SuperInfer KV bytes stored."),
    ("load_bytes", "SuperInfer KV bytes loaded."),
    ("cpu_lookup_requests", "SuperInfer CPU KV lookup requests."),
    ("cpu_lookup_hits", "SuperInfer CPU KV lookup hits."),
    ("cpu_lookup_hit_tokens", "SuperInfer tokens matched from CPU KV."),
    ("boundary_stores_published", "SuperInfer boundary stores published."),
    ("boundary_stores_stored", "SuperInfer boundary stores stored."),
    (
        "boundary_stores_dropped_cpu_full",
        "SuperInfer boundary stores dropped for CPU capacity.",
    ),
    (
        "boundary_stores_skipped_cached",
        "SuperInfer boundary stores skipped as cached.",
    ),
)
_GAUGES = (
    ("pending_store_events", "SuperInfer pending store events."),
    ("pending_store_reqs", "SuperInfer pending store requests."),
    ("pending_load_reqs", "SuperInfer pending load requests."),
    ("cpu_total_blocks", "SuperInfer total CPU KV blocks."),
    ("cpu_free_blocks", "SuperInfer free CPU KV blocks."),
    ("cpu_used_blocks", "SuperInfer used CPU KV blocks."),
)


@dataclass
class SuperInferConnectorStats(KVConnectorStats):
    """Serializable interval observations from SuperInfer."""

    def __post_init__(self) -> None:
        if not self.data:
            self.reset()

    def reset(self) -> None:
        self.data = {name: [] for name, _ in (*_COUNTERS, *_GAUGES)}

    def record(self, values: dict[str, int | float]) -> None:
        for name in self.data:
            if name in values:
                self.data[name].append(values[name])

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            for name, values in other.data.items():
                self.data[name].extend(values)
        return self

    def reduce(self) -> dict[str, int | float]:
        reduced: dict[str, int | float] = {}
        for name, values in self.data.items():
            if not values:
                continue
            if name in {gauge[0] for gauge in _GAUGES}:
                reduced[name] = values[-1]
            else:
                reduced[name] = sum(values)
        return reduced

    def is_empty(self) -> bool:
        return not any(self.data.values())


class SuperInferPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self._counters = {
            name: create_metric_per_engine(
                self._counter_cls(
                    name=f"vllm:superinfer_{name}",
                    documentation=documentation,
                    labelnames=labelnames,
                ),
                per_engine_labelvalues,
            )
            for name, documentation in _COUNTERS
        }
        self._gauges = {
            name: create_metric_per_engine(
                self._gauge_cls(
                    name=f"vllm:superinfer_{name}",
                    documentation=documentation,
                    labelnames=labelnames,
                ),
                per_engine_labelvalues,
            )
            for name, documentation in _GAUGES
        }

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for name, metric in self._counters.items():
            metric[engine_idx].inc(float(transfer_stats_data.get(name, 0)))
        for name, metric in self._gauges.items():
            metric[engine_idx].set(float(transfer_stats_data.get(name, 0)))
