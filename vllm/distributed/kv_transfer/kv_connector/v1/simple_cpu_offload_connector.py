# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SimpleCPUOffloadConnector: minimal CPU KV cache offloading."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    PromMetric,
    PromMetricT,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.simple_kv_offload.manager import (
    SimpleCPUOffloadScheduler,
)
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
)
from vllm.v1.simple_kv_offload.worker import (
    SimpleCPUOffloadWorker,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Default CPU capacity: 8 GB
DEFAULT_CPU_CAPACITY_BYTES = 8 * (1024**3)

_COUNTER_TELEMETRY_KEYS = frozenset(
    {
        "offload_store_events",
        "offload_load_events",
        "offload_store_blocks",
        "offload_load_blocks",
        "offload_store_bytes",
        "offload_load_bytes",
        "offload_store_events_ge_16_blocks",
        "offload_store_events_lt_4_blocks",
        "offload_load_events_ge_16_blocks",
        "offload_load_events_lt_4_blocks",
        "local_swap_out_bytes",
        "local_swap_in_bytes",
        "remote_swap_out_bytes",
        "remote_swap_in_bytes",
        "num_remote_fallbacks",
        "swap_out_time_ms",
        "swap_in_time_ms",
        "local_swap_out_time_ms",
        "local_swap_in_time_ms",
        "remote_swap_out_time_ms",
        "remote_swap_in_time_ms",
        "offload_cpu_lookup_requests",
        "offload_cpu_lookup_hits",
        "offload_cpu_lookup_hit_tokens",
        "offload_load_requests_created",
        "offload_load_events_assigned",
        "offload_speculative_boundary_requests",
        "offload_speculative_accepted_tokens",
        "offload_speculative_dirty_tail_tokens",
        "offload_load_enqueue_wait_ns",
        "offload_store_enqueue_wait_ns",
        "offload_load_enqueue_full",
        "offload_store_enqueue_full",
        "offload_load_blocks_submitted",
        "offload_store_blocks_submitted",
        "offload_load_descriptors_submitted",
        "offload_store_descriptors_submitted",
        "offload_load_bytes_submitted",
        "offload_store_bytes_submitted",
        "offload_load_coalesced_spans",
        "offload_store_coalesced_spans",
        "offload_load_workspace_reallocations",
        "offload_store_workspace_reallocations",
        "offload_native_load_submissions",
        "offload_native_store_submissions",
        "offload_native_load_descriptors",
        "offload_native_store_descriptors",
        "offload_native_load_bytes",
        "offload_native_store_bytes",
        "offload_native_load_submit_ns",
        "offload_native_store_submit_ns",
        "offload_native_load_errors",
        "offload_native_store_errors",
        "offload_native_load_workspace_reallocations",
        "offload_native_store_workspace_reallocations",
        "offload_cpu_kv_capacity_bytes",
        "offload_cpu_kv_allocated_bytes",
        "offload_cpu_kv_pinned_bytes",
        "offload_cpu_kv_allocation_time_ms",
        "offload_cpu_kv_allocation_mode_empty",
        "offload_load_coalesced_spans",
        "offload_store_coalesced_spans",
    }
)


class SimpleCPUOffloadPromMetrics(KVConnectorPromMetrics):
    """Prometheus adapter for SimpleCPUOffloadScheduler snapshots."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self._metrics: dict[tuple[int, str], PromMetricT] = {}
        self._counter_keys = _COUNTER_TELEMETRY_KEYS

        for key in SimpleCPUOffloadScheduler.telemetry_keys():
            metric_cls = (
                self._counter_cls if key in self._counter_keys else self._gauge_cls
            )
            metric = metric_cls(
                name=f"vllm:simple_cpu_offload_{key}",
                documentation=f"Simple CPU KV offload {key.replace('_', ' ')}.",
                labelnames=labelnames,
            )
            for engine_idx, labelvalues in per_engine_labelvalues.items():
                self._metrics[(engine_idx, key)] = metric.labels(*labelvalues)

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for key, value in transfer_stats_data.items():
            if key not in SimpleCPUOffloadScheduler.telemetry_keys():
                continue
            if not isinstance(value, (int, float)):
                continue
            metric = self._metrics.get((engine_idx, key))
            if metric is None:
                continue
            if key in self._counter_keys:
                metric.inc(value)
            else:
                metric.set(value)


@dataclass
class SimpleCPUOffloadConnectorStats(KVConnectorStats):
    """Serializable scheduler-side transfer telemetry."""

    _activity_keys = (
        "offload_pending_store_events",
        "offload_pending_load_reqs",
        "offload_pending_store_reqs",
        "offload_store_events",
        "offload_load_events",
        "offload_store_blocks",
        "offload_load_blocks",
        "offload_store_bytes",
        "offload_load_bytes",
        "local_swap_out_bytes",
        "local_swap_in_bytes",
        "remote_swap_out_bytes",
        "remote_swap_in_bytes",
        "num_remote_fallbacks",
        "swap_out_time_ms",
        "swap_in_time_ms",
        "local_swap_out_time_ms",
        "local_swap_in_time_ms",
        "remote_swap_out_time_ms",
        "remote_swap_in_time_ms",
    ) + tuple(_COUNTER_TELEMETRY_KEYS)
    _sum_keys = _COUNTER_TELEMETRY_KEYS

    def reset(self) -> None:
        self.data = {}

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        assert isinstance(other, SimpleCPUOffloadConnectorStats)
        merged = dict(self.data)
        for key, value in other.data.items():
            if isinstance(value, (int, float)) and key in self._sum_keys:
                merged[key] = merged.get(key, 0) + value
            else:
                merged[key] = value
        self.data = merged
        return self

    def reduce(self) -> dict[str, int | float]:
        return {
            key: value
            for key, value in self.data.items()
            if isinstance(value, (int, float))
        }

    def is_empty(self) -> bool:
        return not self.data


class SimpleCPUOffloadConnector(KVConnectorBase_V1, SupportsHMA):
    """CPU KV cache offloading with custom kernel transfers and BlockPool LRU."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        enable_prefix_caching = vllm_config.cache_config.enable_prefix_caching
        extra_config = self._kv_transfer_config.kv_connector_extra_config or {}

        cpu_capacity_bytes = int(
            extra_config.get("cpu_bytes_to_use", DEFAULT_CPU_CAPACITY_BYTES)
        )
        # cpu_bytes_to_use is server-wide for compatibility;
        # cpu_bytes_to_use_per_rank overrides for per-rank capacity.
        world_size = vllm_config.parallel_config.world_size
        cpu_capacity_per_rank = cpu_capacity_bytes // world_size
        if "cpu_bytes_to_use_per_rank" in extra_config:
            explicit = int(extra_config["cpu_bytes_to_use_per_rank"])
            if explicit != cpu_capacity_per_rank:
                logger.warning(
                    "cpu_bytes_to_use_per_rank (%.2f GB) != "
                    "cpu_bytes_to_use/world_size (%.2f GB). Using per-rank value.",
                    explicit / (1024**3),
                    cpu_capacity_per_rank / (1024**3),
                )
            cpu_capacity_per_rank = explicit

        lazy_offload = bool(extra_config.get("lazy_offload", False))
        debug_single_request_swap = bool(
            extra_config.get("debug_single_request_swap", False)
        )
        pin_memory_fix = bool(extra_config.get("pin_memory_fix", False))
        swapper_block_first = bool(extra_config.get("swapper_block_first", False))
        min_lazy_store_batch_blocks = int(
            extra_config.get("min_lazy_store_batch_blocks", 1)
        )
        transfer_queue_depth = int(extra_config.get("transfer_queue_depth", 8))
        cpu_kv_allocation_mode = str(
            extra_config.get("cpu_kv_allocation_mode", "zero")
        )

        self.scheduler_manager: SimpleCPUOffloadScheduler | None = None
        self.worker_handler: SimpleCPUOffloadWorker | None = None

        if not enable_prefix_caching:
            logger.warning(
                "Detected prefix caching disabled, disabling CPU offload "
                "since it requires prefix caching."
            )
            return

        logger.info(
            "SimpleCPUOffloadConnector: role=%s, "
            "per_rank=%.2f GB, world_size=%d, mode=%s",
            role.name,
            cpu_capacity_per_rank / (1024**3),
            world_size,
            "lazy" if lazy_offload else "eager",
        )

        if role == KVConnectorRole.SCHEDULER:
            from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

            assert kv_cache_config is not None
            scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
                kv_cache_config, vllm_config
            )
            self.scheduler_manager = SimpleCPUOffloadScheduler(
                vllm_config,
                kv_cache_config,
                cpu_capacity_per_rank,
                scheduler_block_size=scheduler_block_size,
                hash_block_size=hash_block_size,
                lazy_offload=lazy_offload,
                debug_single_request_swap=debug_single_request_swap,
                pin_memory_fix=pin_memory_fix,
                swapper_block_first=swapper_block_first,
                min_lazy_store_batch_blocks=min_lazy_store_batch_blocks,
                transfer_queue_depth=transfer_queue_depth,
            )
        elif role == KVConnectorRole.WORKER:
            self.worker_handler = SimpleCPUOffloadWorker(
                vllm_config,
                kv_cache_config,
                cpu_capacity_per_rank,
                pin_memory_fix=pin_memory_fix,
                swapper_block_first=swapper_block_first,
                transfer_queue_depth=transfer_queue_depth,
                cpu_kv_allocation_mode=cpu_kv_allocation_mode,
            )

    # --- Worker-side methods ---

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self.worker_handler is not None:
            self.worker_handler.register_kv_caches(kv_caches)

    def bind_connector_metadata(
        self,
        connector_metadata: KVConnectorMetadata,
    ) -> None:
        super().bind_connector_metadata(connector_metadata)
        if self.worker_handler is not None:
            assert isinstance(connector_metadata, SimpleCPUOffloadMetadata)
            self.worker_handler.bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        if self.worker_handler is not None:
            self.worker_handler.clear_connector_metadata()

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        if self.worker_handler is not None:
            assert isinstance(kv_connector_metadata, SimpleCPUOffloadMetadata)
            self.worker_handler.handle_preemptions(kv_connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        del forward_context, kwargs
        if self.worker_handler is not None:
            self.worker_handler.start_load_kv()

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name
        # The connector's load event is polled after forward. Attention-layer
        # waits are not needed because this connector restores complete blocks
        # before the request is admitted to execution.

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        pass  # Always save asynchronously and deferred to get_finished()

    def wait_for_save(self) -> None:
        pass  # All stores are driven by get_finished() and no wait needed

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        if self.worker_handler is not None:
            return self.worker_handler.get_finished(finished_req_ids)
        return None, None

    def build_connector_worker_meta(self):
        if self.worker_handler is not None:
            return self.worker_handler.build_connector_worker_meta()
        return None

    # --- Scheduler-side methods ---

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.bind_gpu_block_pool(gpu_block_pool)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.get_num_new_matched_tokens(
                request, num_computed_tokens
            )
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.update_state_after_alloc(
                request, blocks, num_external_tokens
            )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.build_connector_meta(scheduler_output)
        return SimpleCPUOffloadMetadata()

    def update_connector_output(
        self,
        connector_output: KVConnectorOutput,
    ) -> None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.update_connector_output(connector_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.request_finished(request, block_ids)
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.request_finished_all_groups(
                request, block_ids
            )
        return False, None

    # NOTE: New API only for SimpleCPUOffloadConnector.
    def has_pending_transfers(self) -> bool:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.has_pending_transfers()
        return False

    def has_pending_store_transfers(self) -> bool:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.has_pending_store_transfers()
        return False

    def get_estimated_swap_bandwidth_bytes_per_s(self) -> float | None:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.get_estimated_swap_bandwidth_bytes_per_s()
        return None

    def get_num_free_cpu_blocks(self) -> int | None:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.get_num_free_cpu_blocks()
        return None

    def get_num_total_cpu_blocks(self) -> int | None:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.get_num_total_cpu_blocks()
        return None

    def get_num_cpu_resident_blocks(self, request: "Request") -> int:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.get_num_cpu_resident_blocks(request)
        return 0

    def get_num_cpu_resident_owned_blocks(self, request: "Request") -> int:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.get_num_cpu_resident_owned_blocks(request)
        return 0

    def estimate_request_remote_penalty_seconds(self, request: "Request") -> float:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.estimate_request_remote_penalty_seconds(request)
        return 0.0

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self.scheduler_manager is None:
            return None
        stats = SimpleCPUOffloadConnectorStats(
            data=self.scheduler_manager.take_telemetry_stats()
        )
        return None if stats.is_empty() else stats

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        return SimpleCPUOffloadConnectorStats(data=data or {})

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return SimpleCPUOffloadPromMetrics(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
        )

    def take_events(self) -> Iterable[KVCacheEvent]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.take_events()
        return []

    # NOTE: Workers are not contacted. In-flight transfers drain naturally,
    # and stale completions are ignored by the guarded
    # SimpleCPUOffloadScheduler._process_store_event().
    def reset_cache(self) -> bool | None:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.reset()
        return None
