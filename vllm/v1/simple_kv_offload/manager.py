# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side manager for SimpleCPUOffloadConnector."""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    KVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm.v1.core.kv_cache_utils import (
    make_block_hash_with_group_id,
    resolve_block_hashes,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.simple_kv_offload.topology import (
    Gh200TopologyMapping,
    LocalityPoolPlanner,
    discover_gh200_topology_mapping,
    estimate_remote_penalty_seconds,
    split_elapsed_ms_by_bytes,
)
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class TransferMeta:
    gpu_block_ids: list[int]
    cpu_block_ids: list[int]
    cpu_block_localities: list[str] = field(default_factory=list)
    source_gpu_rank: int = 0


@dataclass
class LoadRequestState:
    request: "Request"
    transfer_meta: TransferMeta
    load_event: int | None = None
    finished: bool = False


# NOTE: This per-request state is only used in eager mode.
@dataclass
class StoreRequestState:
    request: "Request"
    # Accumulated block IDs from scheduler_output via yield_req_data.
    block_ids: tuple[list[int], ...]
    # Per-group cursors tracking how many blocks have been stored/skipped.
    num_stored_blocks: list[int]
    store_events: set[int] = field(default_factory=set)
    finish_touched_gpu_block_ids: set[int] = field(default_factory=set)
    finished: bool = False
    accepted_tokens: int = 0
    dirty_tail_tokens: int = 0


@dataclass
class TransferPerfState:
    started_at_s: float
    bytes_to_copy: int


@dataclass
class ResidencyRecord:
    request_id: str
    group_idx: int
    ordinal: int
    gpu_block_id: int
    block_hash: bytes | None = None
    cpu_block_id: int | None = None
    state: str = "gpu_only"
    event_idx: int | None = None


class SimpleCPUOffloadScheduler:
    """Scheduler-side manager for CPU offloading."""

    _TELEMETRY_KEYS = (
        "offload_pending_store_events",
        "offload_pending_load_reqs",
        "offload_pending_store_reqs",
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
        "offload_cpu_total_blocks",
        "offload_cpu_free_blocks",
        "offload_cpu_used_blocks",
        "offload_lazy_target_free_blocks",
        "offload_proactive_swap_budget",
        "offload_gh200_topology_fallback_mode",
        "offload_pin_memory_fix",
        "offload_swapper_block_first",
        "offload_gh200_topology_tuned",
        "offload_gh200_topology_discovered",
        "offload_gh200_gpu_rank",
        "offload_local_cpu_pool_blocks",
        "offload_remote_cpu_pool_blocks",
        "offload_min_lazy_store_batch_blocks",
        "offload_transfer_queue_depth",
        "offload_load_queue_depth",
        "offload_store_queue_depth",
        "offload_cpu_kv_capacity_bytes",
        "offload_cpu_kv_allocated_bytes",
        "offload_cpu_kv_pinned_bytes",
        "offload_cpu_kv_allocation_time_ms",
        "offload_cpu_kv_allocation_mode_empty",
        "offload_load_queue_peak_depth",
        "offload_store_queue_peak_depth",
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
        "offload_load_workspace_reallocations",
        "offload_store_workspace_reallocations",
        "offload_native_load_submissions",
        "offload_native_store_submissions",
        "offload_native_load_blocks",
        "offload_native_store_blocks",
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
        "offload_load_coalesced_spans",
        "offload_store_coalesced_spans",
        "offload_estimated_swap_bandwidth_bytes_per_s",
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
        "local_bandwidth_gbps",
        "remote_bandwidth_gbps",
        "offload_residency_gpu_only_blocks",
        "offload_residency_store_in_flight_blocks",
        "offload_residency_cpu_only_blocks",
        "offload_residency_load_in_flight_blocks",
        "offload_residency_synced_blocks",
        "offload_residency_dirty_blocks",
        "offload_residency_invalid_blocks",
        "offload_cpu_lookup_requests",
        "offload_cpu_lookup_hits",
        "offload_cpu_lookup_hit_tokens",
        "offload_cpu_cached_keys",
        "offload_load_requests_created",
        "offload_load_events_assigned",
        "offload_speculative_boundary_requests",
        "offload_speculative_accepted_tokens",
        "offload_speculative_dirty_tail_tokens",
    )

    @classmethod
    def telemetry_keys(cls) -> tuple[str, ...]:
        return cls._TELEMETRY_KEYS

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        scheduler_block_size: int,
        hash_block_size: int,
        lazy_offload: bool = False,
        debug_single_request_swap: bool = False,
        pin_memory_fix: bool = False,
        swapper_block_first: bool = False,
        min_lazy_store_batch_blocks: int = 1,
        transfer_queue_depth: int = 8,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.enable_kv_cache_events = (
            vllm_config.kv_events_config is not None
            and vllm_config.kv_events_config.enable_kv_cache_events
        )
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.cp_world_size = dcp_world_size
        self.block_size = scheduler_block_size
        self.hash_block_size = hash_block_size
        assert self.block_size % self.hash_block_size == 0
        # Derive a CPU KVCacheConfig from the GPU config and build a coordinator
        assert kv_cache_config is not None
        self.cpu_kv_cache_config = self._derive_cpu_config(
            kv_cache_config, cpu_capacity_bytes
        )
        self.num_cpu_blocks = self.cpu_kv_cache_config.num_blocks
        # Find the full attention kv group for prefix cache matching.
        self.fa_gidx = -1
        for g_idx, g in enumerate(self.cpu_kv_cache_config.kv_cache_groups):
            if isinstance(g.kv_cache_spec, FullAttentionSpec):
                self.fa_gidx = g_idx
                break
        assert 0 <= self.fa_gidx < len(self.cpu_kv_cache_config.kv_cache_groups)
        # FA group's own block_size; divides scheduler_block_size (the LCM)
        # but is NOT assumed to equal it.
        self.fa_block_size: int = (
            self.cpu_kv_cache_config.kv_cache_groups[
                self.fa_gidx
            ].kv_cache_spec.block_size
            * self.cp_world_size
        )
        assert self.block_size % self.fa_block_size == 0

        logger.info(
            "SimpleCPUOffloadScheduler: Allocating %d CPU blocks (%.2f GB, mode=%s)",
            self.num_cpu_blocks,
            cpu_capacity_bytes / (1024**3),
            "lazy" if lazy_offload else "eager",
        )

        # TODO (yifan): maybe need to enable kv_cache_events and metrics_collector here.
        self.cpu_coordinator: KVCacheCoordinator = get_kv_cache_coordinator(
            kv_cache_config=self.cpu_kv_cache_config,
            max_model_len=vllm_config.model_config.max_model_len,
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=1,
            scheduler_block_size=self.block_size,
            hash_block_size=self.hash_block_size,
        )
        self.cpu_block_pool: BlockPool = self.cpu_coordinator.block_pool

        # GPU block pool reference - bound after scheduler builds kv_cache_manager
        self._gpu_block_pool: BlockPool | None = None

        # Load metadata
        self._reqs_to_load: dict[str, LoadRequestState] = {}
        # Inverse map: load_event_idx -> req_ids. Keyed by load_event_idx because
        # the worker reports completions by event index, not request id.
        self._load_event_to_reqs: dict[int, list[str]] = {}

        # Pending (cpu_hit_blocks, hit_length) tuples from find_longest_cache_hit,
        # kept pinned via touch() while awaiting update_state_after_alloc().
        self._pending_cpu_hits: dict[
            str, tuple[tuple[list[KVCacheBlock], ...], int]
        ] = {}
        self._residency: dict[tuple[str, int, int], ResidencyRecord] = {}
        self._residency_by_gpu_block: dict[int, set[tuple[str, int, int]]] = {}
        self._residency_keys_by_request: dict[str, set[tuple[str, int, int]]] = {}
        self._requests_by_id: dict[str, Request] = {}
        self._gpu_block_hash_aliases: dict[int, set[bytes]] = {}
        self._gpu_block_primary_hashes: dict[int, bytes | None] = {}

        # Store metadata
        self._lazy_mode = lazy_offload
        self._debug_single_request_swap = debug_single_request_swap
        self._pin_memory_fix = pin_memory_fix
        self._swapper_block_first = swapper_block_first
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self._gh200_topology_tuned = bool(
            extra_config.get(
                "gh200_topology_tuned",
                getattr(vllm_config.cache_config, "gh200_topology_tuned", False),
            )
        )
        self._gpu_rank = int(getattr(vllm_config.parallel_config, "rank", 0))
        self._local_fraction = float(
            extra_config.get(
                "local_cpu_pool_fraction",
                getattr(vllm_config.cache_config, "local_cpu_pool_fraction", 0.75),
            )
        )
        self._local_bw_bytes_per_s = float(
            extra_config.get(
                "local_swap_bandwidth_bytes_per_s",
                getattr(
                    vllm_config.cache_config,
                    "local_swap_bandwidth_bytes_per_s",
                    900 * (1024**3),
                ),
            )
        )
        self._remote_bw_bytes_per_s = float(
            extra_config.get(
                "remote_swap_bandwidth_bytes_per_s",
                getattr(
                    vllm_config.cache_config,
                    "remote_swap_bandwidth_bytes_per_s",
                    300 * (1024**3),
                ),
            )
        )
        self._topology_mapping: Gh200TopologyMapping = discover_gh200_topology_mapping(
            enabled=self._gh200_topology_tuned,
            local_cpu_pool_fraction=self._local_fraction,
            local_bandwidth_bytes_per_s=self._local_bw_bytes_per_s,
            remote_bandwidth_bytes_per_s=self._remote_bw_bytes_per_s,
        )
        for line in self._topology_mapping.to_log_lines():
            logger.warning(line) if (
                self._topology_mapping.fallback_mode and self._gh200_topology_tuned
            ) else logger.info(line)
        self._min_lazy_store_batch_blocks = max(1, int(min_lazy_store_batch_blocks))
        self._transfer_queue_depth = max(1, int(transfer_queue_depth))
        self._telemetry_load_queue_depth = 0
        self._telemetry_store_queue_depth = 0
        self._telemetry_cpu_kv_capacity_bytes = 0
        self._telemetry_cpu_kv_allocated_bytes = 0
        self._telemetry_cpu_kv_pinned_bytes = 0
        self._telemetry_cpu_kv_allocation_time_ms = 0
        self._telemetry_cpu_kv_allocation_mode_empty = 0
        self._telemetry_load_queue_peak_depth = 0
        self._telemetry_store_queue_peak_depth = 0
        self._telemetry_load_enqueue_wait_ns = 0
        self._telemetry_store_enqueue_wait_ns = 0
        self._telemetry_load_enqueue_full = 0
        self._telemetry_store_enqueue_full = 0
        self._telemetry_load_blocks_submitted = 0
        self._telemetry_store_blocks_submitted = 0
        self._telemetry_load_descriptors_submitted = 0
        self._telemetry_store_descriptors_submitted = 0
        self._telemetry_load_bytes_submitted = 0
        self._telemetry_store_bytes_submitted = 0
        self._telemetry_load_workspace_reallocations = 0
        self._telemetry_store_workspace_reallocations = 0
        self._telemetry_native_load_submissions = 0
        self._telemetry_native_store_submissions = 0
        self._telemetry_native_load_blocks = 0
        self._telemetry_native_store_blocks = 0
        self._telemetry_native_load_descriptors = 0
        self._telemetry_native_store_descriptors = 0
        self._telemetry_native_load_bytes = 0
        self._telemetry_native_store_bytes = 0
        self._telemetry_native_load_submit_ns = 0
        self._telemetry_native_store_submit_ns = 0
        self._telemetry_native_load_errors = 0
        self._telemetry_native_store_errors = 0
        self._telemetry_native_load_workspace_reallocations = 0
        self._telemetry_native_store_workspace_reallocations = 0
        self._telemetry_load_coalesced_spans = 0
        self._telemetry_store_coalesced_spans = 0
        self._telemetry_cpu_lookup_requests = 0
        self._telemetry_cpu_lookup_hits = 0
        self._telemetry_cpu_lookup_hit_tokens = 0
        self._telemetry_load_requests_created = 0
        self._telemetry_load_events_assigned = 0
        self._logged_cpu_lookup_diagnostic = False
        self._accepted_token_boundaries: dict[str, int] = {}
        self._dirty_tail_tokens: dict[str, int] = {}
        self._store_completion_log_count = 0
        self._alias_tracking_log_count = 0
        self._proactive_swap_budget = max(
            0,
            int(
                extra_config.get(
                    "proactive_swap_budget",
                    getattr(vllm_config.scheduler_config, "proactive_swap_budget", 0),
                )
            ),
        )
        # Lazy mode: use a cursor to track the last scanned block in the GPU free queue.
        self._cursor: KVCacheBlock | None = None
        if self._lazy_mode:
            estimated_target_free = self._estimate_lazy_target_blocks(
                kv_cache_config,
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.cp_world_size,
            )
            # The proactive budget is a transfer/rotation budget, not a GPU
            # free-block watermark. Using it as the watermark forces a large
            # lazy scan even when the workload is not under KV pressure.
            self._target_free = estimated_target_free
        else:
            estimated_target_free = 0
            self._target_free = 0
        logger.info(
            "SimpleCPUOffloadScheduler: lazy_target_free_blocks=%d "
            "(estimated=%d, proactive_swap_budget=%d)",
            self._target_free,
            estimated_target_free,
            self._proactive_swap_budget,
        )
        logger.info(
            "SimpleCPUOffloadScheduler: cache geometry block_size=%d "
            "hash_block_size=%d cpu_hash_block_size=%d cpu_manager_block_sizes=%s",
            self.block_size,
            self.hash_block_size,
            self.cpu_block_pool.hash_block_size,
            [manager.block_size for manager in self.cpu_coordinator.single_type_managers],
        )
        logger.info(
            "SimpleCPUOffloadScheduler: max_inflight_store_events=%d",
            self._transfer_queue_depth,
        )
        self._store_event_to_blocks: dict[int, TransferMeta] = {}
        self._abandoned_store_event_to_blocks: dict[int, TransferMeta] = {}
        # Eager mode only
        self._reqs_to_store: dict[str, StoreRequestState] = {}
        self._store_event_to_reqs: dict[int, list[str]] = {}
        self._in_flight_store_gpu_blocks: set[int] = set()
        self._abandoned_reqs_to_load: dict[str, LoadRequestState] = {}

        # Event counters
        self._load_event_counter: int = 0
        self._store_event_counter: int = 0

        # For TP/PP: track partial store completions across steps.
        # Events must be reported by all world_size workers before considered complete.
        self._expected_worker_count = vllm_config.parallel_config.world_size
        self._store_event_pending_counts: dict[int, int] = {}

        self._telemetry_store_events = 0
        self._telemetry_load_events = 0
        self._telemetry_store_blocks = 0
        self._telemetry_load_blocks = 0
        self._telemetry_store_bytes = 0
        self._telemetry_load_bytes = 0
        self._telemetry_store_events_ge_16_blocks = 0
        self._telemetry_store_events_lt_4_blocks = 0
        self._telemetry_load_events_ge_16_blocks = 0
        self._telemetry_load_events_lt_4_blocks = 0
        self._telemetry_local_swap_out_bytes = 0
        self._telemetry_local_swap_in_bytes = 0
        self._telemetry_remote_swap_out_bytes = 0
        self._telemetry_remote_swap_in_bytes = 0
        self._telemetry_num_remote_fallbacks = 0
        self._telemetry_swap_out_time_ms = 0.0
        self._telemetry_swap_in_time_ms = 0.0
        self._telemetry_local_swap_out_time_ms = 0.0
        self._telemetry_local_swap_in_time_ms = 0.0
        self._telemetry_remote_swap_out_time_ms = 0.0
        self._telemetry_remote_swap_in_time_ms = 0.0
        self._estimated_swap_bandwidth_bytes_per_s: float | None = None
        self._bandwidth_ema_alpha = 0.2
        self._store_event_perf: dict[int, TransferPerfState] = {}
        self._load_event_perf: dict[int, TransferPerfState] = {}
        self._store_event_localities: dict[int, list[str]] = {}
        self._load_event_localities: dict[int, list[str]] = {}
        self._locality_planner = LocalityPoolPlanner(
            total_blocks=max(self.num_cpu_blocks - 1, 0),
            enabled=self._gh200_topology_tuned,
            mapping=self._topology_mapping,
            local_fraction=self._local_fraction,
        )
        self._cpu_block_island_map = self._build_cpu_block_island_map()

    @property
    def telemetry_stats(self) -> dict[str, int | float]:
        free_blocks = self.cpu_block_pool.get_num_free_blocks()
        return {
            "offload_pending_store_events": len(self._store_event_to_blocks),
            "offload_pending_load_reqs": len(self._reqs_to_load),
            "offload_pending_store_reqs": len(self._reqs_to_store),
            "offload_store_events": self._telemetry_store_events,
            "offload_load_events": self._telemetry_load_events,
            "offload_store_blocks": self._telemetry_store_blocks,
            "offload_load_blocks": self._telemetry_load_blocks,
            "offload_store_bytes": self._telemetry_store_bytes,
            "offload_load_bytes": self._telemetry_load_bytes,
            "offload_store_events_ge_16_blocks": self._telemetry_store_events_ge_16_blocks,
            "offload_store_events_lt_4_blocks": self._telemetry_store_events_lt_4_blocks,
            "offload_load_events_ge_16_blocks": self._telemetry_load_events_ge_16_blocks,
            "offload_load_events_lt_4_blocks": self._telemetry_load_events_lt_4_blocks,
            "offload_cpu_total_blocks": self.num_cpu_blocks,
            "offload_cpu_free_blocks": free_blocks,
            "offload_cpu_used_blocks": self.num_cpu_blocks - free_blocks,
            "offload_lazy_target_free_blocks": self._target_free,
            "offload_proactive_swap_budget": self._proactive_swap_budget,
            "offload_gh200_topology_fallback_mode": int(
                self._topology_mapping.fallback_mode
            ),
            "offload_pin_memory_fix": int(self._pin_memory_fix),
            "offload_swapper_block_first": int(self._swapper_block_first),
            "offload_gh200_topology_tuned": int(self._gh200_topology_tuned),
            "offload_gh200_topology_discovered": int(
                self._topology_mapping.discovered
            ),
            "offload_gh200_gpu_rank": self._gpu_rank,
            "offload_local_cpu_pool_blocks": self._locality_planner.local_total_blocks,
            "offload_remote_cpu_pool_blocks": self._locality_planner.remote_total_blocks,
            "offload_min_lazy_store_batch_blocks": self._min_lazy_store_batch_blocks,
            "offload_transfer_queue_depth": self._transfer_queue_depth,
            "offload_load_queue_depth": self._telemetry_load_queue_depth,
            "offload_store_queue_depth": self._telemetry_store_queue_depth,
            "offload_cpu_kv_capacity_bytes": self._telemetry_cpu_kv_capacity_bytes,
            "offload_cpu_kv_allocated_bytes": self._telemetry_cpu_kv_allocated_bytes,
            "offload_cpu_kv_pinned_bytes": self._telemetry_cpu_kv_pinned_bytes,
            "offload_cpu_kv_allocation_time_ms": self._telemetry_cpu_kv_allocation_time_ms,
            "offload_cpu_kv_allocation_mode_empty": self._telemetry_cpu_kv_allocation_mode_empty,
            "offload_load_queue_peak_depth": self._telemetry_load_queue_peak_depth,
            "offload_store_queue_peak_depth": self._telemetry_store_queue_peak_depth,
            "offload_load_enqueue_wait_ns": self._telemetry_load_enqueue_wait_ns,
            "offload_store_enqueue_wait_ns": self._telemetry_store_enqueue_wait_ns,
            "offload_load_enqueue_full": self._telemetry_load_enqueue_full,
            "offload_store_enqueue_full": self._telemetry_store_enqueue_full,
            "offload_load_blocks_submitted": self._telemetry_load_blocks_submitted,
            "offload_store_blocks_submitted": self._telemetry_store_blocks_submitted,
            "offload_load_descriptors_submitted": self._telemetry_load_descriptors_submitted,
            "offload_store_descriptors_submitted": self._telemetry_store_descriptors_submitted,
            "offload_load_bytes_submitted": self._telemetry_load_bytes_submitted,
            "offload_store_bytes_submitted": self._telemetry_store_bytes_submitted,
            "offload_load_workspace_reallocations": self._telemetry_load_workspace_reallocations,
            "offload_store_workspace_reallocations": self._telemetry_store_workspace_reallocations,
            "offload_native_load_submissions": self._telemetry_native_load_submissions,
            "offload_native_store_submissions": self._telemetry_native_store_submissions,
            "offload_native_load_blocks": self._telemetry_native_load_blocks,
            "offload_native_store_blocks": self._telemetry_native_store_blocks,
            "offload_native_load_descriptors": self._telemetry_native_load_descriptors,
            "offload_native_store_descriptors": self._telemetry_native_store_descriptors,
            "offload_native_load_bytes": self._telemetry_native_load_bytes,
            "offload_native_store_bytes": self._telemetry_native_store_bytes,
            "offload_native_load_submit_ns": self._telemetry_native_load_submit_ns,
            "offload_native_store_submit_ns": self._telemetry_native_store_submit_ns,
            "offload_native_load_errors": self._telemetry_native_load_errors,
            "offload_native_store_errors": self._telemetry_native_store_errors,
            "offload_native_load_workspace_reallocations": self._telemetry_native_load_workspace_reallocations,
            "offload_native_store_workspace_reallocations": self._telemetry_native_store_workspace_reallocations,
            "offload_load_coalesced_spans": self._telemetry_load_coalesced_spans,
            "offload_store_coalesced_spans": self._telemetry_store_coalesced_spans,
            "offload_estimated_swap_bandwidth_bytes_per_s": int(
                self._estimated_swap_bandwidth_bytes_per_s or 0
            ),
            "local_swap_out_bytes": self._telemetry_local_swap_out_bytes,
            "local_swap_in_bytes": self._telemetry_local_swap_in_bytes,
            "remote_swap_out_bytes": self._telemetry_remote_swap_out_bytes,
            "remote_swap_in_bytes": self._telemetry_remote_swap_in_bytes,
            "num_remote_fallbacks": self._telemetry_num_remote_fallbacks,
            "swap_out_time_ms": int(self._telemetry_swap_out_time_ms),
            "swap_in_time_ms": int(self._telemetry_swap_in_time_ms),
            "local_swap_out_time_ms": int(self._telemetry_local_swap_out_time_ms),
            "local_swap_in_time_ms": int(self._telemetry_local_swap_in_time_ms),
            "remote_swap_out_time_ms": int(self._telemetry_remote_swap_out_time_ms),
            "remote_swap_in_time_ms": int(self._telemetry_remote_swap_in_time_ms),
            "local_bandwidth_gbps": self._local_bw_bytes_per_s * 8 / 1e9,
            "remote_bandwidth_gbps": self._remote_bw_bytes_per_s * 8 / 1e9,
            **self._residency_stats(),
            "offload_cpu_lookup_requests": self._telemetry_cpu_lookup_requests,
            "offload_cpu_lookup_hits": self._telemetry_cpu_lookup_hits,
            "offload_cpu_lookup_hit_tokens": self._telemetry_cpu_lookup_hit_tokens,
            "offload_cpu_cached_keys": len(
                self.cpu_block_pool.cached_block_hash_to_block._cache
            ),
            "offload_load_requests_created": self._telemetry_load_requests_created,
            "offload_load_events_assigned": self._telemetry_load_events_assigned,
            "offload_speculative_boundary_requests": len(
                self._accepted_token_boundaries
            ),
            "offload_speculative_accepted_tokens": sum(
                self._accepted_token_boundaries.values()
            ),
            "offload_speculative_dirty_tail_tokens": sum(
                self._dirty_tail_tokens.values()
            ),
        }

    def _residency_stats(self) -> dict[str, int]:
        counts = {
            "gpu_only": 0,
            "store_in_flight": 0,
            "cpu_only": 0,
            "load_in_flight": 0,
            "synced": 0,
            "dirty": 0,
            "invalid": 0,
        }
        for record in self._residency.values():
            counts[record.state] = counts.get(record.state, 0) + 1
        return {
            "offload_residency_gpu_only_blocks": counts["gpu_only"],
            "offload_residency_store_in_flight_blocks": counts["store_in_flight"],
            "offload_residency_cpu_only_blocks": counts["cpu_only"],
            "offload_residency_load_in_flight_blocks": counts["load_in_flight"],
            "offload_residency_synced_blocks": counts["synced"],
            "offload_residency_dirty_blocks": counts["dirty"],
            "offload_residency_invalid_blocks": counts["invalid"],
        }

    def _record_gpu_blocks(
        self,
        req_id: str,
        block_id_groups: tuple[list[int], ...],
        request: "Request | None" = None,
    ) -> None:
        if self._gpu_block_pool is None:
            return
        store_states = getattr(self, "_reqs_to_store", {})
        state = store_states.get(req_id)
        request = request or (state.request if state is not None else None)
        if request is not None:
            self._requests_by_id[req_id] = request
        groups = (
            self.cpu_kv_cache_config.kv_cache_groups
            if request is not None
            else ()
        )
        aliases_by_block = getattr(self, "_gpu_block_hash_aliases", None)
        if aliases_by_block is None:
            aliases_by_block = self._gpu_block_hash_aliases = {}
        primary_hashes = getattr(self, "_gpu_block_primary_hashes", None)
        if primary_hashes is None:
            primary_hashes = self._gpu_block_primary_hashes = {}
        aliases_for_call: dict[int, set[bytes]] = {}
        for group_idx, block_ids in enumerate(block_id_groups):
            for ordinal, block_id in enumerate(block_ids):
                key = (req_id, group_idx, ordinal)
                block = self._gpu_block_pool.blocks[block_id]
                if getattr(block, "is_null", False):
                    self._set_residency_record(
                        key,
                        ResidencyRecord(
                            request_id=req_id,
                            group_idx=group_idx,
                            ordinal=ordinal,
                            gpu_block_id=block_id,
                            block_hash=None,
                        ),
                    )
                    continue
                previous_primary = primary_hashes.get(block_id)
                current_primary = block.block_hash
                if block_id in primary_hashes and (
                    previous_primary != current_primary
                ):
                    aliases_by_block.pop(block_id, None)
                primary_hashes[block_id] = current_primary
                aliases = aliases_for_call.setdefault(block_id, set())
                if block.block_hash is not None:
                    aliases.add(block.block_hash)
                if request is not None and group_idx < len(groups):
                    group_block_size = (
                        groups[group_idx].kv_cache_spec.block_size
                        * self.cp_world_size
                    )
                    group_hashes = resolve_block_hashes(
                        request.block_hashes,
                        self.hash_block_size,
                        group_block_size,
                    )
                    if ordinal < len(group_hashes):
                        aliases.add(
                            make_block_hash_with_group_id(
                                group_hashes[ordinal], group_idx
                            )
                        )
                record = self._residency.get(key)
                if record is None:
                    self._set_residency_record(
                        key,
                        ResidencyRecord(
                        request_id=req_id,
                        group_idx=group_idx,
                        ordinal=ordinal,
                        gpu_block_id=block_id,
                        block_hash=block.block_hash,
                        ),
                    )
                else:
                    record.gpu_block_id = block_id
                    record.block_hash = block.block_hash

        # Replace, rather than extend, aliases for IDs observed in this request.
        # GPU block IDs are recycled; retaining aliases from an older owner can
        # make unrelated requests appear to have enormous group coverage.
        for block_id, aliases in aliases_for_call.items():
            aliases_by_block.setdefault(block_id, set()).update(aliases)

        if request is not None and self._alias_tracking_log_count < 8:
            self._alias_tracking_log_count += 1
            tracked_groups: dict[int, int] = {}
            for block_ids in block_id_groups:
                for block_id in block_ids:
                    for alias in aliases_by_block.get(block_id, ()):
                        group_id = int.from_bytes(alias[-4:], "big", signed=False)
                        tracked_groups[group_id] = tracked_groups.get(group_id, 0) + 1
            logger.info(
                "SimpleCPUOffloadScheduler: tracked request aliases req=%s "
                "group_block_counts=%s tracked_alias_groups=%s",
                req_id,
                [len(block_ids) for block_ids in block_id_groups],
                tracked_groups,
            )

    def _mark_residency_for_gpu_blocks(
        self,
        gpu_block_ids: list[int],
        state: str,
        *,
        cpu_block_ids: list[int] | None = None,
        event_idx: int | None = None,
    ) -> None:
        cpu_block_ids = cpu_block_ids or []
        for index, gpu_block_id in enumerate(gpu_block_ids):
            matched = [
                self._residency[key]
                for key in getattr(self, "_residency_by_gpu_block", {}).get(
                    gpu_block_id, ()
                )
                if key in self._residency
            ]
            if not matched:
                key = ("<unowned>", 0, gpu_block_id)
                matched = [
                    self._set_residency_record(
                        key,
                        ResidencyRecord(
                            request_id="<unowned>",
                            group_idx=0,
                            ordinal=gpu_block_id,
                            gpu_block_id=gpu_block_id,
                        ),
                    )
                ]
            for record in matched:
                record.state = state
                record.event_idx = event_idx
                if index < len(cpu_block_ids):
                    record.cpu_block_id = cpu_block_ids[index]

    def _reconcile_request_residency(self, req_id: str) -> None:
        """Invalidate tracked CPU residency after an individual cache eviction."""
        cached = self.cpu_block_pool.cached_block_hash_to_block
        for key in self._residency_keys_by_request.get(req_id, ()):
            record = self._residency.get(key)
            if record is None:
                continue
            if record.state not in ("cpu_only", "synced"):
                continue
            if record.block_hash is None or cached.get_one_block(record.block_hash) is None:
                record.cpu_block_id = None
                record.event_idx = None
                record.state = "gpu_only"

    def _residency_record(
        self, req_id: str, group_idx: int, ordinal: int
    ) -> ResidencyRecord | None:
        return self._residency.get((req_id, group_idx, ordinal))

    def _set_residency_record(
        self, key: tuple[str, int, int], record: ResidencyRecord
    ) -> ResidencyRecord:
        if not hasattr(self, "_residency_by_gpu_block"):
            self._residency_by_gpu_block = {}
        if not hasattr(self, "_residency_keys_by_request"):
            self._residency_keys_by_request = {}
        previous = self._residency.get(key)
        if previous is not None and previous.gpu_block_id != record.gpu_block_id:
            self._residency_by_gpu_block.get(previous.gpu_block_id, set()).discard(key)
        self._residency[key] = record
        self._residency_by_gpu_block.setdefault(record.gpu_block_id, set()).add(key)
        self._residency_keys_by_request.setdefault(record.request_id, set()).add(key)
        return record

    def _remove_residency_record(self, key: tuple[str, int, int]) -> None:
        if not hasattr(self, "_residency_by_gpu_block"):
            self._residency_by_gpu_block = {}
        if not hasattr(self, "_residency_keys_by_request"):
            self._residency_keys_by_request = {}
        record = self._residency.pop(key, None)
        if record is None:
            return
        block_keys = self._residency_by_gpu_block.get(record.gpu_block_id)
        if block_keys is not None:
            block_keys.discard(key)
            if not block_keys:
                self._residency_by_gpu_block.pop(record.gpu_block_id, None)
        request_keys = self._residency_keys_by_request.get(record.request_id)
        if request_keys is not None:
            request_keys.discard(key)
            if not request_keys:
                self._residency_keys_by_request.pop(record.request_id, None)

    def _drop_residency_request(self, req_id: str) -> None:
        """Remove request-scoped records after no transfer still depends on them."""
        in_flight_states = {"store_in_flight", "load_in_flight"}
        if not hasattr(self, "_residency_keys_by_request"):
            for key, record in list(self._residency.items()):
                if (
                    getattr(record, "request_id", None) == req_id
                    and getattr(record, "state", None) not in in_flight_states
                ):
                    self._residency.pop(key, None)
            return
        for key in list(self._residency_keys_by_request.get(req_id, ())):
            record = self._residency.get(key)
            if record is not None and record.state not in in_flight_states:
                self._remove_residency_record(key)
        if not self._residency_keys_by_request.get(req_id):
            self._requests_by_id.pop(req_id, None)

    def take_telemetry_stats(self) -> dict[str, int | float]:
        stats = self.telemetry_stats
        for name in (
            "store_events",
            "load_events",
            "store_blocks",
            "load_blocks",
            "store_bytes",
            "load_bytes",
            "store_events_ge_16_blocks",
            "store_events_lt_4_blocks",
            "load_events_ge_16_blocks",
            "load_events_lt_4_blocks",
            "local_swap_out_bytes",
            "local_swap_in_bytes",
            "remote_swap_out_bytes",
            "remote_swap_in_bytes",
            "num_remote_fallbacks",
            "native_load_submissions",
            "native_store_submissions",
            "native_load_blocks",
            "native_store_blocks",
            "native_load_blocks",
            "native_store_blocks",
            "native_load_descriptors",
            "native_store_descriptors",
            "native_load_bytes",
            "native_store_bytes",
            "native_load_submit_ns",
            "native_store_submit_ns",
            "native_load_errors",
            "native_store_errors",
            "native_load_workspace_reallocations",
            "native_store_workspace_reallocations",
            "load_coalesced_spans",
            "store_coalesced_spans",
            "load_queue_peak_depth",
            "store_queue_peak_depth",
            "load_enqueue_wait_ns",
            "store_enqueue_wait_ns",
            "load_enqueue_full",
            "store_enqueue_full",
            "load_blocks_submitted",
            "store_blocks_submitted",
            "load_descriptors_submitted",
            "store_descriptors_submitted",
            "load_bytes_submitted",
            "store_bytes_submitted",
            "load_workspace_reallocations",
            "store_workspace_reallocations",
        ):
            setattr(self, f"_telemetry_{name}", 0)
        for name in (
            "swap_out_time_ms",
            "swap_in_time_ms",
            "local_swap_out_time_ms",
            "local_swap_in_time_ms",
            "remote_swap_out_time_ms",
            "remote_swap_in_time_ms",
        ):
            setattr(self, f"_telemetry_{name}", 0.0)
        return stats

    def _build_cpu_block_island_map(self) -> dict[int, int]:
        islands = self._locality_planner.island_ids
        block_ids = [block.block_id for block in self.cpu_block_pool.blocks if not block.is_null]
        if not islands or not block_ids:
            return {}
        per_island, extra = divmod(len(block_ids), len(islands))
        result: dict[int, int] = {}
        cursor = 0
        for index, island in enumerate(islands):
            count = per_island + (index < extra)
            for block_id in block_ids[cursor : cursor + count]:
                result[block_id] = island
            cursor += count
        return result

    def _choose_cpu_blocks_for_localities(self, labels: list[str]) -> list[int]:
        free_blocks = self.cpu_block_pool.free_block_queue.get_all_free_blocks()
        source_island = self._locality_planner.island_for_gpu_rank(self._gpu_rank)
        local = [
            block.block_id
            for block in free_blocks
            if self._cpu_block_island_map.get(block.block_id, source_island)
            == source_island
        ]
        remote = [
            block.block_id
            for block in free_blocks
            if self._cpu_block_island_map.get(block.block_id, source_island)
            != source_island
        ]
        selected: list[int] = []
        for label in labels:
            candidates = local if label == "local" else remote
            fallback = remote if label == "local" else local
            if candidates:
                selected.append(candidates.pop(0))
            elif fallback:
                selected.append(fallback.pop(0))
            else:
                break
        if len(selected) != len(labels):
            return [block.block_id for block in free_blocks[: len(labels)]]
        return selected

    def _allocate_cpu_blocks_locality_aware(
        self, num_blocks: int, req_id: str
    ) -> tuple[list[KVCacheBlock], list[str]]:
        if num_blocks <= 0:
            return [], []
        labels = self._locality_planner.plan_labels_for_allocation(
            num_blocks=num_blocks, source_gpu_rank=self._gpu_rank
        )
        selected_ids = self._choose_cpu_blocks_for_localities(labels)
        blocks = [self.cpu_block_pool.blocks[block_id] for block_id in selected_ids]
        for block in blocks:
            self.cpu_block_pool.free_block_queue.remove(block)
            if self.cpu_block_pool.enable_caching:
                self.cpu_block_pool._maybe_evict_cached_block(block)
            assert block.ref_cnt == 0
            block.ref_cnt += 1
            if self.cpu_block_pool.metrics_collector:
                self.cpu_block_pool.metrics_collector.on_block_allocated(block)
        fallback_count = self._locality_planner.record_block_assignments(
            selected_ids,
            labels=labels,
            source_gpu_rank=self._gpu_rank,
            req_id=req_id,
        )
        self._telemetry_num_remote_fallbacks += fallback_count
        return blocks, labels

    def _count_transfer_bytes(self, num_blocks: int) -> int:
        if num_blocks <= 0 or self.cpu_kv_cache_config.num_blocks <= 0:
            return 0
        tensors = self.cpu_kv_cache_config.kv_cache_tensors
        total_bytes = tensors[0].size if tensors[0].block_stride else sum(t.size for t in tensors)
        return num_blocks * (total_bytes // self.cpu_kv_cache_config.num_blocks)

    def get_estimated_swap_bandwidth_bytes_per_s(self) -> float | None:
        return self._estimated_swap_bandwidth_bytes_per_s

    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_block_pool.get_num_free_blocks()

    def get_num_total_cpu_blocks(self) -> int:
        return self.num_cpu_blocks

    def get_num_cpu_resident_blocks(self, request: "Request") -> int:
        if not request.block_hashes or request.num_tokens <= 1:
            return 0
        hit_blocks, hit_length, _ = self.cpu_coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_tokens - 1
        )
        if hit_length <= 0:
            return 0
        return sum(1 for group in hit_blocks for block in group if not block.is_null)

    def get_num_cpu_resident_owned_blocks(self, request: "Request") -> int:
        state = self._reqs_to_store.get(request.request_id)
        if state is None or self._gpu_block_pool is None:
            return 0
        cached = self.cpu_block_pool.cached_block_hash_to_block
        count = 0
        for group_ids in state.block_ids:
            for block_id in group_ids:
                block = self._gpu_block_pool.blocks[block_id]
                if (
                    not block.is_null
                    and block.block_hash is not None
                    and block.ref_cnt <= 1
                    and cached.get_one_block(block.block_hash) is not None
                ):
                    count += 1
        return count

    def estimate_request_remote_penalty_seconds(self, request: "Request") -> float:
        if not self._gh200_topology_tuned:
            return 0.0
        state = self._reqs_to_store.get(request.request_id)
        if state is None:
            return 0.0
        unsynced = max(
            sum(len(group) for group in state.block_ids)
            - self.get_num_cpu_resident_owned_blocks(request),
            0,
        )
        remote = self._locality_planner.estimate_remote_blocks_for_gpu(
            source_gpu_rank=self._gpu_rank, num_blocks=unsynced
        )
        return estimate_remote_penalty_seconds(
            remote * self._count_transfer_bytes(1),
            local_bandwidth_bytes_per_s=self._local_bw_bytes_per_s,
            remote_bandwidth_bytes_per_s=self._remote_bw_bytes_per_s,
        )

    def _update_estimated_swap_bandwidth(
        self, bytes_to_copy: int, elapsed_s: float
    ) -> None:
        if bytes_to_copy <= 0 or elapsed_s <= 0:
            return
        measured = bytes_to_copy / elapsed_s
        if not math.isfinite(measured) or measured <= 0:
            return
        if self._estimated_swap_bandwidth_bytes_per_s is None:
            self._estimated_swap_bandwidth_bytes_per_s = measured
        else:
            alpha = self._bandwidth_ema_alpha
            self._estimated_swap_bandwidth_bytes_per_s = (
                (1 - alpha) * self._estimated_swap_bandwidth_bytes_per_s
                + alpha * measured
            )

    @staticmethod
    def _split_local_remote_bytes(
        labels: list[str], bytes_per_block: int
    ) -> tuple[int, int]:
        local = sum(label == "local" for label in labels) * bytes_per_block
        return local, len(labels) * bytes_per_block - local

    @staticmethod
    def _derive_cpu_config(
        gpu_config: "KVCacheConfig", cpu_capacity_bytes: int
    ) -> "KVCacheConfig":
        """Derive a CPU KVCacheConfig from the GPU config.
        Same kv_cache_groups, num_blocks scaled by CPU/GPU memory ratio."""
        # Import here to avoid potential circular imports
        from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfigCls
        from vllm.v1.kv_cache_interface import KVCacheTensor

        assert len(gpu_config.kv_cache_tensors) > 0

        is_packed = any(t.block_stride for t in gpu_config.kv_cache_tensors)
        assert not is_packed or all(t.block_stride for t in gpu_config.kv_cache_tensors)
        gpu_total_bytes = (
            gpu_config.kv_cache_tensors[0].size
            if is_packed
            else sum(t.size for t in gpu_config.kv_cache_tensors)
        )
        num_gpu_blocks = gpu_config.num_blocks
        num_cpu_blocks = max(1, num_gpu_blocks * cpu_capacity_bytes // gpu_total_bytes)
        # Create CPU kv_cache_tensors mirroring GPU by scaling size proportionally.
        cpu_tensors = [
            KVCacheTensor(
                size=t.size // num_gpu_blocks * num_cpu_blocks,
                shared_by=list(t.shared_by),
                offset=t.offset,
                block_stride=t.block_stride,
            )
            for t in gpu_config.kv_cache_tensors
        ]

        return KVCacheConfigCls(
            num_blocks=num_cpu_blocks,
            kv_cache_tensors=cpu_tensors,
            kv_cache_groups=gpu_config.kv_cache_groups,
        )

    @staticmethod
    def _estimate_lazy_target_blocks(
        kv_cache_config: "KVCacheConfig",
        max_num_batched_tokens: int,
        cp_world_size: int = 1,
    ) -> int:
        """GPU blocks to keep available (free/offloaded) per step in lazy mode."""
        WATERMARK_RATIO = 1.0  # Reserve larger space to avoid running out of GPU blocks
        target = 0
        for g in kv_cache_config.kv_cache_groups:
            spec = g.kv_cache_spec
            block_size = spec.block_size * cp_world_size
            if isinstance(spec, MambaSpec):
                target += 2
            elif isinstance(spec, SlidingWindowSpec):
                target += cdiv(spec.sliding_window, block_size) + 1
            else:
                target += cdiv(max_num_batched_tokens, block_size)
        return int(target * (1 + WATERMARK_RATIO))

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        """Bind GPU block pool so that we can touch blocks during stores.
        Called by Scheduler after kv_cache_manager is ready."""
        self._gpu_block_pool = gpu_block_pool

    def residency_snapshot(self) -> dict[str, int]:
        """Return current per-block residency counts for diagnostics/tests."""
        return self._residency_stats()

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Return (num_new_tokens, is_async) from consecutive CPU cache hits."""

        # Pins found CPU blocks so they survive LRU eviction until
        # update_state_after_alloc() consumes them. Any pin from an earlier
        # call on the same request (e.g. retry after a failed allocate_slots)
        # is dropped first.
        if stale := self._pending_cpu_hits.pop(request.request_id, None):
            self._free_pending_cpu_hit(stale)

        num_skipped_hashes = num_computed_tokens // self.hash_block_size
        remaining_hashes = request.block_hashes[num_skipped_hashes:]

        if not remaining_hashes:
            return 0, False
        # Must recompute at least the last token, matching the logic in
        # kv_cache_manager.get_computed_blocks().
        max_hit_len = request.num_tokens - 1 - num_computed_tokens
        self._telemetry_cpu_lookup_requests += 1
        if max_hit_len <= 0:
            return 0, False
        cached_keys = list(self.cpu_block_pool.cached_block_hash_to_block._cache)
        # The first lookup often happens while the store phase is still
        # populating CPU memory. Keep the diagnostic armed until keys exist so
        # it describes the reload phase rather than startup traffic.
        log_cpu_lookup = (
            not self._logged_cpu_lookup_diagnostic and len(cached_keys) >= 300
        )
        if log_cpu_lookup:
            logger.info(
                "SimpleCPUOffloadScheduler: first CPU lookup request=%s "
                "request_hash_len=%s cached_key_len=%s request_hash_prefix=%s "
                "cached_key_prefix=%s",
                request.request_id,
                len(remaining_hashes[0]) if remaining_hashes else 0,
                len(cached_keys[0]) if cached_keys else 0,
                remaining_hashes[0][:8].hex() if remaining_hashes else "",
                cached_keys[0][:8].hex() if cached_keys else "",
            )
        cpu_hit_blocks, hit_length, _ = self.cpu_coordinator.find_longest_cache_hit(
            remaining_hashes, max_hit_len
        )
        per_group_lengths: tuple[int, ...] = ()
        if hit_length <= 0:
            # Hybrid cache reconciliation can reject an otherwise usable
            # per-group prefix when one sparse group tightens the fixed point.
            # Retry independently, but admit only the common scheduler-aligned
            # prefix so every DeepSeek KV group is restored together.
            find_per_group = getattr(
                self.cpu_coordinator, "find_longest_cache_hit_per_group", None
            )
            if find_per_group is not None:
                per_group_blocks, per_group_lengths = find_per_group(
                    remaining_hashes, max_hit_len
                )
                if per_group_lengths:
                    common_hit_length = min(per_group_lengths)
                    common_hit_length = (
                        common_hit_length // self.block_size * self.block_size
                    )
                    if common_hit_length > 0:
                        cpu_hit_blocks = tuple(
                            group[: cdiv(common_hit_length, manager.block_size)]
                            for group, manager in zip(
                                per_group_blocks,
                                self.cpu_coordinator.single_type_managers,
                            )
                        )
                        hit_length = common_hit_length
                        if log_cpu_lookup:
                            logger.info(
                                "SimpleCPUOffloadScheduler: hybrid per-group "
                                "lookup recovered common_hit_length=%d "
                                "group_hit_lengths=%s",
                                hit_length,
                                per_group_lengths,
                            )
        if log_cpu_lookup:
            cached_key_groups: dict[int, int] = {}
            for key in self.cpu_block_pool.cached_block_hash_to_block._cache:
                group_id = int.from_bytes(key[-4:], "big", signed=False)
                cached_key_groups[group_id] = cached_key_groups.get(group_id, 0) + 1
            direct_group_first_hits: list[bool] = []
            direct_group_hash_indices: list[int] = []
            hash_unit = self.cpu_block_pool.hash_block_size
            for group_id, manager in enumerate(
                self.cpu_coordinator.single_type_managers
            ):
                hash_index = manager.block_size // hash_unit - 1
                direct_group_hash_indices.append(hash_index)
                direct_group_first_hits.append(
                    hash_index < len(remaining_hashes)
                    and bool(
                        self.cpu_block_pool.get_cached_block(
                            remaining_hashes[hash_index], [group_id]
                        )
                    )
                )
            logger.info(
                "SimpleCPUOffloadScheduler: CPU lookup result request=%s "
                "hit_length=%d group_hit_blocks=%s group_cache_keys=%s "
                "direct_group_first_hits=%s direct_group_hash_indices=%s "
                "per_group_hit_lengths=%s",
                request.request_id,
                hit_length,
                [
                    sum(not block.is_null for block in group)
                    for group in cpu_hit_blocks
                ],
                cached_key_groups,
                direct_group_first_hits,
                direct_group_hash_indices,
                per_group_lengths,
            )
            self._logged_cpu_lookup_diagnostic = True

        if hit_length > 0:
            self._telemetry_cpu_lookup_hits += 1
            self._telemetry_cpu_lookup_hit_tokens += hit_length
            pin_blocks = [
                blk for grp in cpu_hit_blocks for blk in grp if not blk.is_null
            ]
            self.cpu_block_pool.touch(pin_blocks)
            self._pending_cpu_hits[request.request_id] = (
                cpu_hit_blocks,
                hit_length,
            )
            return hit_length, True
        return 0, False

    # TODO(yifan): this API now only matches the suffix part of the prefix cache. A more
    # general API should scan blocks in both GPU and CPU block pool in a single pass.
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        req_id = request.request_id
        block_ids_by_group = blocks.get_block_ids()
        num_groups = len(block_ids_by_group)

        # Eager mode tracks every request. Lazy mode starts tracking when
        # pressure makes request-owned residency useful to the scheduler.
        if (
            not self._lazy_mode
            or req_id in self._reqs_to_store
            or self._is_lazy_store_pressure_active()
        ) and req_id not in self._reqs_to_store:
            self._reqs_to_store[req_id] = StoreRequestState(
                request=request,
                block_ids=tuple([] for _ in range(num_groups)),
                num_stored_blocks=[0] * num_groups,
            )
        self._record_gpu_blocks(req_id, block_ids_by_group, request=request)

        # Pop the CPU hit cached by get_num_new_matched_tokens(). The
        # found blocks were pinned there to survive LRU eviction in the window
        # between get_num_new_matched_tokens() and this matching call.
        pending = self._pending_cpu_hits.pop(req_id, None)

        if num_external_tokens == 0:
            if pending is not None:
                logger.warning(
                    "SimpleCPUOffloadScheduler: update_state_after_alloc "
                    "called for req_id=%s with no external tokens but "
                    "get_num_new_matched_tokens() unexpectedly recorded "
                    "a pending CPU hit; releasing the stale pin.",
                    req_id,
                )
                self._free_pending_cpu_hit(pending)
            return

        if pending is None:
            logger.warning(
                "SimpleCPUOffloadScheduler: update_state_after_alloc called "
                "for req_id=%s with num_external_tokens=%d but no pending "
                "CPU hit from get_num_new_matched_tokens(); skipping load.",
                req_id,
                num_external_tokens,
            )
            return

        cpu_hit_blocks_full, _ = pending

        # ``num_external_tokens`` is LCM-aligned (checked per-group below),
        # so this counts whole scheduler-aligned chunks of incoming tokens.
        num_blocks_to_load = num_external_tokens // self.block_size
        assert num_blocks_to_load > 0
        num_cached_fa_blocks = sum(
            blk.block_hash is not None for blk in blocks.blocks[self.fa_gidx]
        )
        num_computed_tokens = num_cached_fa_blocks * self.fa_block_size

        # Build transfer pairs across all groups.
        total_computed_tokens = num_computed_tokens + num_external_tokens
        kv_cache_groups = self.cpu_kv_cache_config.kv_cache_groups

        # The scheduler may have accepted fewer blocks than
        # get_num_new_matched_tokens() reported.
        # (e.g. due to token budget in test_partial_gpu_prefix_plus_cpu_load).
        # Take only the leading N blocks per group matching num_external_tokens;
        # the rest will be released along with the temp pin below.
        cpu_hit_blocks: list[list[KVCacheBlock]] = []
        for g in range(num_groups):
            g_block_size = (
                kv_cache_groups[g].kv_cache_spec.block_size * self.cp_world_size
            )
            assert num_external_tokens % g_block_size == 0, (
                f"num_external_tokens={num_external_tokens} not aligned to "
                f"group {g} block_size={g_block_size}"
            )
            n_take_g = num_external_tokens // g_block_size
            cpu_hit_blocks.append(cpu_hit_blocks_full[g][:n_take_g])

        gpu_block_ids: list[int] = []
        cpu_block_ids: list[int] = []
        cpu_blocks_to_touch: list[KVCacheBlock] = []

        for g in range(num_groups):
            cpu_blocks_g = cpu_hit_blocks[g]
            n_ext_g = len(cpu_blocks_g)
            if n_ext_g == 0:
                continue

            # Number of blocks in the computed range for this group.
            g_block_size = (
                kv_cache_groups[g].kv_cache_spec.block_size * self.cp_world_size
            )
            n_computed_g = cdiv(total_computed_tokens, g_block_size)

            # Back-trace: ext blocks sit at the tail of the computed range.
            gpu_ext_start = n_computed_g - n_ext_g
            group_gpu_ids = block_ids_by_group[g]

            for i, cpu_blk in enumerate(cpu_blocks_g):
                # Skip null blocks (e.g. sliding window or mamba padding).
                if cpu_blk.is_null:
                    continue
                gpu_block_ids.append(group_gpu_ids[gpu_ext_start + i])
                cpu_block_ids.append(cpu_blk.block_id)
                cpu_blocks_to_touch.append(cpu_blk)

        # Touch CPU blocks to prevent eviction during async load.
        self.cpu_block_pool.touch(cpu_blocks_to_touch)
        # Release the temporary pin held since get_num_new_matched_tokens().
        self._free_pending_cpu_hit(pending)

        # Touch GPU blocks to prevent freeing during async load
        assert self._gpu_block_pool is not None
        self._gpu_block_pool.touch(
            [self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
        )

        assert self._reqs_to_load.get(req_id) is None
        self._reqs_to_load[req_id] = LoadRequestState(
            request=request,
            transfer_meta=TransferMeta(
                gpu_block_ids,
                cpu_block_ids,
                self._locality_planner.labels_for_block_ids(
                    cpu_block_ids, source_gpu_rank=self._gpu_rank
                ),
                self._gpu_rank,
            ),
        )
        self._telemetry_load_requests_created += 1

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> SimpleCPUOffloadMetadata:
        # --- Stores ---
        store_event = -1
        store_gpu, store_cpu, store_req_ids = self.prepare_store_specs(scheduler_output)
        if store_gpu:
            store_event = self._store_event_counter
            self._store_event_counter += 1
            self._store_event_to_blocks[store_event] = TransferMeta(
                store_gpu,
                store_cpu,
                self._locality_planner.labels_for_block_ids(
                    store_cpu, source_gpu_rank=self._gpu_rank
                ),
                self._gpu_rank,
            )
            self._mark_residency_for_gpu_blocks(
                store_gpu,
                "store_in_flight",
                cpu_block_ids=store_cpu,
                event_idx=store_event,
            )
            store_bytes = self._count_transfer_bytes(len(store_gpu))
            store_localities = self._store_event_to_blocks[store_event].cpu_block_localities
            self._store_event_localities[store_event] = store_localities
            self._store_event_perf[store_event] = TransferPerfState(
                time.monotonic(), store_bytes
            )
            self._record_transfer_telemetry(is_store=True, num_blocks=len(store_gpu), bytes_to_copy=store_bytes)
            if store_req_ids:  # For eager mode only, track req->blocks mapping
                self._store_event_to_reqs[store_event] = store_req_ids
                for req_id in store_req_ids:
                    store_state = self._reqs_to_store.get(req_id)
                    if store_state is not None:
                        store_state.store_events.add(store_event)

        # --- Loads ---
        load_event = -1
        load_gpu: list[int] = []
        load_cpu: list[int] = []
        load_localities: list[str] = []
        load_req_ids: list[str] = []
        for req_id, load_state in self._reqs_to_load.items():
            if load_state.load_event is not None:
                continue
            assert load_state.transfer_meta is not None
            load_gpu.extend(load_state.transfer_meta.gpu_block_ids)
            load_cpu.extend(load_state.transfer_meta.cpu_block_ids)
            load_localities.extend(load_state.transfer_meta.cpu_block_localities)
            load_req_ids.append(req_id)
        if load_req_ids:
            load_event = self._load_event_counter
            self._load_event_counter += 1
            for req_id in load_req_ids:
                self._reqs_to_load[req_id].load_event = load_event
            self._load_event_to_reqs[load_event] = load_req_ids
            self._telemetry_load_events_assigned += 1
            self._mark_residency_for_gpu_blocks(
                load_gpu,
                "load_in_flight",
                cpu_block_ids=load_cpu,
                event_idx=load_event,
            )
            load_bytes = self._count_transfer_bytes(len(load_gpu))
            self._load_event_localities[load_event] = load_localities
            self._load_event_perf[load_event] = TransferPerfState(
                time.monotonic(), load_bytes
            )
            self._record_transfer_telemetry(is_store=False, num_blocks=len(load_gpu), bytes_to_copy=load_bytes)

        result = SimpleCPUOffloadMetadata(
            load_event=load_event,
            load_gpu_blocks=load_gpu,
            load_cpu_blocks=load_cpu,
            load_cpu_block_localities={
                load_event: load_localities
            }
            if load_event >= 0
            else {},
            load_event_to_reqs={
                event_idx: list(req_ids)
                for event_idx, req_ids in self._load_event_to_reqs.items()
            },
            store_event=store_event,
            store_gpu_blocks=store_gpu,
            store_cpu_blocks=store_cpu,
            store_cpu_block_localities=(
                self._store_event_localities.get(store_event, [])
                if store_event >= 0
                else []
            ),
            need_flush=bool(scheduler_output.preempted_req_ids),
            pin_memory_fix=self._pin_memory_fix,
            swapper_block_first=self._swapper_block_first,
        )
        return result

    def prepare_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[list[int], list[int], list[str]]:
        """Prepare store specs for the store event."""
        if self._lazy_mode:
            if not self._store_transfer_capacity_available():
                return [], [], []
            if scheduler_output.finished_req_ids:
                finished_specs = self._prepare_eager_store_specs(scheduler_output)
                if finished_specs[0]:
                    return finished_specs
            if not self._has_lazy_store_pressure(scheduler_output):
                return [], [], []
            return self._prepare_lazy_store_specs()
        return self._prepare_eager_store_specs(scheduler_output)

    def _has_lazy_store_pressure(self, scheduler_output: SchedulerOutput) -> bool:
        if scheduler_output.preempted_req_ids:
            return True
        if self._gpu_block_pool is None or self._target_free <= 0:
            return False
        return self._gpu_block_pool.get_num_free_blocks() <= self._target_free + 1

    def _is_lazy_store_pressure_active(self) -> bool:
        return (
            self._gpu_block_pool is not None
            and self._target_free > 0
            and self._gpu_block_pool.get_num_free_blocks() <= self._target_free + 1
        )

    def _prepare_lazy_store_specs(
        self,
    ) -> tuple[list[int], list[int], list[str]]:
        """Single-pass cursor walk: offload cached GPU blocks near eviction.

        Walks the GPU free queue from the cursor, counting blocks that are
        free-or-offloaded (safe for the allocator to evict). Stops when
        target_free blocks are covered or CPU capacity is reached.
        """
        gpu_pool = self._gpu_block_pool
        if gpu_pool is None or self._target_free <= 0:
            return [], [], []

        free_queue = gpu_pool.free_block_queue
        cpu_pool = self.cpu_block_pool
        num_cpu_free = cpu_pool.get_num_free_blocks()

        # Validate cursor: stale if block was removed from free queue.
        if self._cursor is not None and self._cursor.ref_cnt > 0:
            self._cursor = None

        gpu_ids: list[int] = []
        block_hashes: list[bytes] = []
        last_visited = self._cursor
        candidate_groups: dict[int, int] = {}
        candidate_count = 0
        scan_limit = self._target_free * max(
            2, len(self.cpu_kv_cache_config.kv_cache_groups)
        )

        for covered, node in enumerate(free_queue.iter_blocks_after(self._cursor)):
            if (
                covered >= scan_limit
                or len(gpu_ids) >= self._target_free
                or len(gpu_ids) >= num_cpu_free
            ):
                break

            last_visited = node
            bhash = node.block_hash
            aliases = getattr(self, "_gpu_block_hash_aliases", {}).get(
                node.block_id, ()
            )
            store_hash = bhash or next(iter(aliases), None)
            if store_hash is not None and not node.is_null:
                candidate_count += 1
                for alias in aliases:
                    group_id = int.from_bytes(alias[-4:], "big", signed=False)
                    candidate_groups[group_id] = candidate_groups.get(group_id, 0) + 1

            if (
                store_hash is not None
                and not node.is_null
                and cpu_pool.cached_block_hash_to_block.get_one_block(store_hash)
                is None
            ):
                for companion_id in self._lazy_companion_block_ids(node.block_id):
                    if len(gpu_ids) >= num_cpu_free or companion_id in gpu_ids:
                        break
                    companion = gpu_pool.blocks[companion_id]
                    if companion.ref_cnt != 0 or companion.is_null:
                        continue
                    companion_aliases = getattr(
                        self, "_gpu_block_hash_aliases", {}
                    ).get(companion_id, ())
                    companion_hash = companion.block_hash or next(
                        iter(companion_aliases), None
                    )
                    if companion_hash is None:
                        continue
                    if (
                        cpu_pool.cached_block_hash_to_block.get_one_block(
                            companion_hash
                        )
                        is not None
                    ):
                        continue
                    gpu_ids.append(companion_id)
                    block_hashes.append(companion_hash)

        self._cursor = last_visited

        if self._store_completion_log_count < 8:
            logger.info(
                "SimpleCPUOffloadScheduler: lazy scan candidates=%d "
                "candidate_alias_groups=%s selected=%d selected_hash_groups=%s",
                candidate_count,
                candidate_groups,
                len(gpu_ids),
                {
                    int.from_bytes(hash_value[-4:], "big", signed=False): sum(
                        value == hash_value for value in block_hashes
                    )
                    for hash_value in set(block_hashes)
                },
            )

        if self._debug_single_request_swap:
            gpu_ids = gpu_ids[:1]
            block_hashes = block_hashes[:1]
        if (
            self._target_free > 1
            and len(gpu_ids) < self._min_lazy_store_batch_blocks
        ):
            return [], [], []

        # Batch-allocate CPU blocks and stamp hashes.
        if gpu_ids:
            cpu_blocks, _ = self._allocate_cpu_blocks_locality_aware(
                len(gpu_ids), "<lazy-scan>"
            )
            cpu_ids = [blk.block_id for blk in cpu_blocks]
            for cpu_blk, bhash in zip(cpu_blocks, block_hashes):  # type: ignore[assignment]
                cpu_blk._block_hash = bhash  # type: ignore[assignment]
            # Touch GPU blocks to prevent eviction during async copy.
            gpu_pool.touch([gpu_pool.blocks[bid] for bid in gpu_ids])
        else:
            cpu_ids = []

        return gpu_ids, cpu_ids, []

    def _lazy_companion_block_ids(self, gpu_block_id: int) -> list[int]:
        """Return free blocks covering the same token segment in each group."""
        gpu_pool = self._gpu_block_pool
        if gpu_pool is None:
            return [gpu_block_id]
        records = [
            record
            for record in self._residency.values()
            if record.gpu_block_id == gpu_block_id
            and record.request_id != "<unowned>"
        ]
        if not records:
            return [gpu_block_id]

        groups = self.cpu_kv_cache_config.kv_cache_groups
        selected: list[int] = []
        for source in records:
            source_block_size = (
                groups[source.group_idx].kv_cache_spec.block_size
                * self.cp_world_size
            )
            source_start = source.ordinal * source_block_size
            source_end = source_start + source_block_size
            for candidate in self._residency.values():
                if candidate.request_id != source.request_id:
                    continue
                group_block_size = (
                    groups[candidate.group_idx].kv_cache_spec.block_size
                    * self.cp_world_size
                )
                candidate_start = candidate.ordinal * group_block_size
                candidate_end = candidate_start + group_block_size
                if candidate_end <= source_start or candidate_start >= source_end:
                    continue
                if candidate.gpu_block_id not in selected:
                    selected.append(candidate.gpu_block_id)

        return selected or [gpu_block_id]

    def _prepare_eager_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[list[int], list[int], list[str]]:
        """Identify newly computed blocks to offload from scheduler requests.

        Only considers blocks whose KV data has been **confirmed computed** by
        the GPU. This means blocks from the current step are NOT stored until the
        next step. If a request finishes in the same step as its last full block,
        that block may be missed. (TODO: flush on finish.)

        Returns:
            (gpu_block_ids, cpu_block_ids, req_ids) for the store event.
        """

        merged_gpu_block_ids: list[int] = []
        merged_cpu_block_ids: list[int] = []
        req_ids: list[str] = []

        gpu_block_pool = self._gpu_block_pool
        if gpu_block_pool is None:
            return [], [], []
        cpu_block_pool = self.cpu_block_pool
        num_free = cpu_block_pool.get_num_free_blocks()
        kv_cache_groups = self.cpu_kv_cache_config.kv_cache_groups
        num_groups = len(kv_cache_groups)
        # Dedup against blocks already scheduled.
        in_flight = self._in_flight_store_gpu_blocks

        finished_req_ids = scheduler_output.finished_req_ids or set()
        scheduled_req_ids: set[str] = set()
        request_data = []
        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            scheduled_req_ids.add(req_id)
            request_data.append((req_id, new_block_id_groups, preempted))
        request_data.extend(
            (req_id, None, False)
            for req_id in finished_req_ids - scheduled_req_ids
        )

        for req_id, new_block_id_groups, preempted in request_data:
            state = self._reqs_to_store.get(req_id)
            if state is None or (state.finished and req_id not in finished_req_ids):
                continue

            # Accumulate new block IDs.
            if preempted:
                state.block_ids = tuple([] for _ in range(num_groups))
                state.num_stored_blocks = [0] * num_groups
                for record in self._residency.values():
                    if record.request_id == req_id:
                        record.state = "invalid"
                        record.cpu_block_id = None
                        record.event_idx = None
            if new_block_id_groups:
                for g in range(min(num_groups, len(new_block_id_groups))):
                    if new_block_id_groups[g] is not None:
                        state.block_ids[g].extend(new_block_id_groups[g])
            self._record_gpu_blocks(req_id, state.block_ids, request=state.request)

            num_new_tokens = scheduler_output.num_scheduled_tokens.get(
                req_id, state.request.num_tokens if req_id in finished_req_ids else 0
            )
            if num_new_tokens == 0:
                continue

            block_ids_by_group = state.block_ids
            if not block_ids_by_group:
                continue

            self._reconcile_request_residency(req_id)

            # --- Phase 1: Scan blocks, classify as cached vs to-store ---
            gpu_block_ids: list[int] = []
            block_hashes_to_store: list[bytes] = []
            advanced_per_group: list[int] = [0] * num_groups
            out_of_space = False
            # Confirmed tokens: KV data written and visible to all streams.
            req = state.request
            confirmed_tokens = (
                req.num_tokens
                if req_id in finished_req_ids or req.is_finished()
                else req.num_computed_tokens - req.num_output_placeholders
            )
            # Cap to blocks with confirmed KV data.
            aligned_tokens = confirmed_tokens // self.block_size * self.block_size

            for g in range(num_groups):
                # FIXME (yifan): handle CPU cache eviction, where
                # num_stored_blocks can be stale and omit evicted blocks in
                # the middle of the request.
                group_gpu_ids = block_ids_by_group[g]

                g_block_size = (
                    kv_cache_groups[g].kv_cache_spec.block_size * self.cp_world_size
                )
                request_group_hashes = resolve_block_hashes(
                    req.block_hashes,
                    self.hash_block_size,
                    g_block_size,
                )
                ready_blocks_g = aligned_tokens // g_block_size
                scannable = group_gpu_ids[:ready_blocks_g]
                for dirty_ordinal in range(ready_blocks_g, len(group_gpu_ids)):
                    dirty_record = self._residency_record(
                        req_id, g, dirty_ordinal
                    )
                    if dirty_record is not None and dirty_record.state not in (
                        "cpu_only",
                        "synced",
                    ):
                        dirty_record.state = "dirty"

                for ordinal, gpu_block_id in enumerate(scannable):
                    gpu_block = gpu_block_pool.blocks[gpu_block_id]
                    if gpu_block.is_null:
                        advanced_per_group[g] += 1
                        continue

                    bhash_with_group = (
                        make_block_hash_with_group_id(
                            request_group_hashes[ordinal], g
                        )
                        if ordinal < len(request_group_hashes)
                        else None
                    )
                    if bhash_with_group is None:
                        # No request hash exists for this position, so it cannot
                        # serve a prefix-cache hit or a stable CPU reload.
                        advanced_per_group[g] += 1
                        continue

                    record = self._residency_record(req_id, g, ordinal)
                    # Skip if already scheduled for store or already cached in CPU.
                    if (
                        gpu_block_id in in_flight
                        or cpu_block_pool.cached_block_hash_to_block.get_one_block(
                            bhash_with_group
                        )
                        is not None
                    ):
                        if record is not None:
                            record.state = "synced"
                        advanced_per_group[g] += 1
                        continue

                    if num_free <= 0:
                        out_of_space = True
                        break
                    num_free -= 1

                    gpu_block_ids.append(gpu_block_id)
                    block_hashes_to_store.append(bhash_with_group)
                    if record is not None:
                        record.state = "store_in_flight"
                    advanced_per_group[g] += 1

                if out_of_space:
                    break

            # --- Phase 2: Batch allocate CPU blocks and stamp hashes ---
            n_to_alloc = len(gpu_block_ids)
            if n_to_alloc > 0:
                cpu_blocks_alloc, _ = self._allocate_cpu_blocks_locality_aware(
                    n_to_alloc, req_id
                )
                cpu_block_ids = [blk.block_id for blk in cpu_blocks_alloc]
                for cpu_blk, bhash in zip(cpu_blocks_alloc, block_hashes_to_store):
                    cpu_blk._block_hash = bhash  # type: ignore[assignment]
            else:
                cpu_block_ids = []

            if cpu_block_ids:
                req_ids.append(req_id)
                merged_gpu_block_ids.extend(gpu_block_ids)
                merged_cpu_block_ids.extend(cpu_block_ids)
                in_flight.update(gpu_block_ids)
                self._mark_residency_for_gpu_blocks(
                    gpu_block_ids,
                    "store_in_flight",
                    cpu_block_ids=cpu_block_ids,
                )

                # Touch GPU blocks to prevent freeing during async copy
                gpu_block_pool.touch(
                    [gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
                )

                logger.debug(
                    "Request %s: Scheduling store of %d blocks to CPU (%d groups)",
                    req_id,
                    len(cpu_block_ids),
                    num_groups,
                )

            # Advance per-group cursors (includes cached hits + newly stored)
            for g in range(num_groups):
                state.num_stored_blocks[g] += advanced_per_group[g]

        return merged_gpu_block_ids, merged_cpu_block_ids, req_ids

    def _record_transfer_telemetry(
        self, *, is_store: bool, num_blocks: int, bytes_to_copy: int
    ) -> None:
        prefix = "store" if is_store else "load"
        for suffix, increment in (
            ("events", 1),
            ("blocks", num_blocks),
            ("bytes", bytes_to_copy),
        ):
            name = f"_telemetry_{prefix}_{suffix}"
            setattr(self, name, getattr(self, name) + increment)
        if num_blocks >= 16:
            name = f"_telemetry_{prefix}_events_ge_16_blocks"
            setattr(self, name, getattr(self, name) + 1)
        if 0 < num_blocks < 4:
            name = f"_telemetry_{prefix}_events_lt_4_blocks"
            setattr(self, name, getattr(self, name) + 1)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Handle async transfer completions from worker.

        Load completions arrive via finished_recving (real req_ids).
        Store completions arrive via kv_connector_worker_meta as
        per-event worker counts. We accumulate across steps and process
        a store event only when all workers have reported completion.
        """
        # --- Load completions ---
        for req_id in list(connector_output.finished_recving or []):
            self._cleanup_load_request(req_id)

        # --- Store completions ---
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, SimpleCPUOffloadWorkerMetadata):
            return
        self._telemetry_load_queue_depth = meta.load_queue_depth
        self._telemetry_store_queue_depth = meta.store_queue_depth
        self._telemetry_cpu_kv_capacity_bytes = max(
            self._telemetry_cpu_kv_capacity_bytes, meta.cpu_kv_capacity_bytes
        )
        self._telemetry_cpu_kv_allocated_bytes = max(
            self._telemetry_cpu_kv_allocated_bytes, meta.cpu_kv_allocated_bytes
        )
        self._telemetry_cpu_kv_pinned_bytes = max(
            self._telemetry_cpu_kv_pinned_bytes, meta.cpu_kv_pinned_bytes
        )
        self._telemetry_cpu_kv_allocation_time_ms = max(
            self._telemetry_cpu_kv_allocation_time_ms,
            meta.cpu_kv_allocation_time_ms,
        )
        self._telemetry_cpu_kv_allocation_mode_empty = max(
            self._telemetry_cpu_kv_allocation_mode_empty,
            meta.cpu_kv_allocation_mode_empty,
        )
        for name in (
            "load_queue_peak_depth",
            "store_queue_peak_depth",
            "load_enqueue_wait_ns",
            "store_enqueue_wait_ns",
            "load_enqueue_full",
            "store_enqueue_full",
            "load_blocks_submitted",
            "store_blocks_submitted",
            "load_descriptors_submitted",
            "store_descriptors_submitted",
            "load_bytes_submitted",
            "store_bytes_submitted",
            "load_coalesced_spans",
            "store_coalesced_spans",
            "load_workspace_reallocations",
            "store_workspace_reallocations",
        ):
            setattr(
                self,
                f"_telemetry_{name}",
                getattr(self, f"_telemetry_{name}") + getattr(meta, name),
            )
        for name in (
            "native_load_submissions",
            "native_store_submissions",
            "native_load_descriptors",
            "native_store_descriptors",
            "native_load_bytes",
            "native_store_bytes",
            "native_load_submit_ns",
            "native_store_submit_ns",
            "native_load_errors",
            "native_store_errors",
            "native_load_workspace_reallocations",
            "native_store_workspace_reallocations",
        ):
            setattr(
                self,
                f"_telemetry_{name}",
                getattr(self, f"_telemetry_{name}") + getattr(meta, name),
            )
        for event_idx, count in meta.completed_store_events.items():
            total = self._store_event_pending_counts.get(event_idx, 0) + count
            if total >= self._expected_worker_count:
                self._store_event_pending_counts.pop(event_idx, None)
                self._process_store_event(event_idx)
            else:
                self._store_event_pending_counts[event_idx] = total

    def record_speculative_boundary(
        self, request_id: str, *, accepted_tokens: int, dirty_tail_tokens: int
    ) -> None:
        """Record DSpark boundary state without changing admission decisions.

        Rotation remains disabled until GPU rollback/restore evidence exists. The
        bookkeeping makes accepted-token and dirty-tail state available for that
        later gate without coupling speculative execution to DMA scheduling.
        """
        accepted_tokens = max(int(accepted_tokens), 0)
        dirty_tail_tokens = max(int(dirty_tail_tokens), 0)
        self._accepted_token_boundaries[request_id] = accepted_tokens
        self._dirty_tail_tokens[request_id] = dirty_tail_tokens
        state = self._reqs_to_store.get(request_id)
        if state is not None:
            state.accepted_tokens = accepted_tokens
            state.dirty_tail_tokens = dirty_tail_tokens

    def speculative_boundary_snapshot(self) -> dict[str, int]:
        """Return diagnostic DSpark boundary counters."""
        return {
            "requests": len(self._accepted_token_boundaries),
            "accepted_tokens": sum(self._accepted_token_boundaries.values()),
            "dirty_tail_tokens": sum(self._dirty_tail_tokens.values()),
        }

    def _process_store_event(self, event_idx: int) -> None:
        """Process a fully-completed store event."""
        transfer = self._store_event_to_blocks.pop(event_idx, None)
        if transfer is None:
            transfer = self._abandoned_store_event_to_blocks.pop(event_idx, None)
            if transfer is None:
                return  # guard stale events from before a reset() call
            self._release_transfer_refs(transfer)
            self._store_event_perf.pop(event_idx, None)
            self._store_event_localities.pop(event_idx, None)
            self._locality_planner.release_block_ids(
                transfer.cpu_block_ids, source_gpu_rank=transfer.source_gpu_rank
            )
            return

        if not self._lazy_mode:
            self._in_flight_store_gpu_blocks.difference_update(transfer.gpu_block_ids)

        self._process_store_completion(
            event_idx, transfer.gpu_block_ids, transfer.cpu_block_ids
        )
        self._mark_residency_for_gpu_blocks(
            transfer.gpu_block_ids,
            "cpu_only",
            cpu_block_ids=transfer.cpu_block_ids,
        )
        self._locality_planner.release_block_ids(
            transfer.cpu_block_ids, source_gpu_rank=transfer.source_gpu_rank
        )
        perf = self._store_event_perf.pop(event_idx, None)
        if perf is not None:
            elapsed_s = max(time.monotonic() - perf.started_at_s, 1e-9)
            self._update_estimated_swap_bandwidth(perf.bytes_to_copy, elapsed_s)
            elapsed_ms = elapsed_s * 1000
            self._telemetry_swap_out_time_ms += elapsed_ms
            local_bytes, remote_bytes = self._split_local_remote_bytes(
                self._store_event_localities.pop(event_idx, []),
                self._count_transfer_bytes(1),
            )
            self._telemetry_local_swap_out_bytes += local_bytes
            self._telemetry_remote_swap_out_bytes += remote_bytes
            local_ms, remote_ms = split_elapsed_ms_by_bytes(
                elapsed_ms, local_bytes=local_bytes, remote_bytes=remote_bytes
            )
            self._telemetry_local_swap_out_time_ms += local_ms
            self._telemetry_remote_swap_out_time_ms += remote_ms
        logger.debug(
            "Store event %d completed: cached %d blocks to CPU",
            event_idx,
            len(transfer.cpu_block_ids),
        )

        for req_id in self._store_event_to_reqs.pop(event_idx, []):
            state = self._reqs_to_store.get(req_id)
            if state is None:
                continue
            state.store_events.discard(event_idx)
            if state.finished and not state.store_events:
                self._cleanup_store_request(req_id)

    def _process_store_completion(
        self, event_idx: int, gpu_block_ids: list[int], cpu_block_ids: list[int]
    ) -> None:
        """Cache CPU blocks per-group and release GPU refs.

        Block hashes were stamped on CPU blocks at allocation time (in
        ``_prepare_*_store_specs``).  Here we just register them in the
        cache map so they become discoverable by the load path.
        """
        assert len(cpu_block_ids) == len(gpu_block_ids)

        assert self._gpu_block_pool is not None
        cpu_blocks = [self.cpu_block_pool.blocks[bid] for bid in cpu_block_ids]

        for gpu_block_id, cpu_block in zip(gpu_block_ids, cpu_blocks):
            bhash = cpu_block.block_hash
            assert bhash is not None
            cpu_pool = self.cpu_block_pool
            cpu_pool.cached_block_hash_to_block.insert(bhash, cpu_block)

            # DeepSeek-V4 packs several KV groups into one physical block.
            # BlockPool keeps the non-primary group hashes as aliases; retain
            # those aliases on the CPU block so every group can find the same
            # copied payload during reload.
            gpu_block = self._gpu_block_pool.blocks[gpu_block_id]
            for alias in self._gpu_block_pool.cached_block_hashes_by_block.get(
                gpu_block_id, ()
            ):
                if alias != bhash:
                    cpu_pool._insert_block_hash(alias, cpu_block, num_tokens=None)
            for alias in self._gpu_block_hash_aliases.get(gpu_block_id, ()):
                if alias != bhash:
                    cpu_pool._insert_block_hash(alias, cpu_block, num_tokens=None)
            for alias in self._request_hash_aliases_for_gpu_block(gpu_block_id):
                if alias != bhash:
                    cpu_pool._insert_block_hash(alias, cpu_block, num_tokens=None)

        self._store_completion_log_count += 1
        key_groups: dict[int, int] = {}
        for key in self.cpu_block_pool.cached_block_hash_to_block._cache:
            group_id = int.from_bytes(key[-4:], "big", signed=False)
            key_groups[group_id] = key_groups.get(group_id, 0) + 1
        source_groups: dict[int, int] = {}
        tracked_groups: dict[int, int] = {}
        for block_id in gpu_block_ids:
            source_hash = self._gpu_block_pool.blocks[block_id].block_hash
            if source_hash is None:
                continue
            group_id = int.from_bytes(source_hash[-4:], "big", signed=False)
            source_groups[group_id] = source_groups.get(group_id, 0) + 1
            for alias in self._gpu_block_hash_aliases.get(block_id, ()):
                group_id = int.from_bytes(alias[-4:], "big", signed=False)
                tracked_groups[group_id] = tracked_groups.get(group_id, 0) + 1
        logger.info(
            "SimpleCPUOffloadScheduler: store completion keys event=%d "
            "completion_count=%d cached_keys=%d cached_key_groups=%s "
            "source_primary_groups=%s tracked_alias_groups=%s",
            event_idx,
            self._store_completion_log_count,
            len(self.cpu_block_pool.cached_block_hash_to_block._cache),
            key_groups,
            source_groups,
            tracked_groups,
        )

        # Free CPU and GPU blocks' ref counts to turn them into prefix cache
        self.cpu_block_pool.free_blocks(cpu_blocks)
        assert self._gpu_block_pool is not None
        self._gpu_block_pool.free_blocks(
            self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids
        )

    def _request_hash_aliases_for_gpu_block(
        self, gpu_block_id: int
    ) -> tuple[bytes, ...]:
        """Derive group-qualified hashes for a lazily stored GPU block."""
        aliases: list[bytes] = []
        groups = self.cpu_kv_cache_config.kv_cache_groups
        for record in self._residency.values():
            if record.gpu_block_id != gpu_block_id:
                continue
            state = self._reqs_to_store.get(record.request_id)
            request = (
                state.request
                if state is not None
                else self._requests_by_id.get(record.request_id)
            )
            if request is None:
                continue
            source_block_size = (
                groups[record.group_idx].kv_cache_spec.block_size
                * self.cp_world_size
            )
            source_start = record.ordinal * source_block_size
            source_end = source_start + source_block_size
            for group_idx, group in enumerate(groups):
                group_block_size = (
                    group.kv_cache_spec.block_size * self.cp_world_size
                )
                group_hashes = resolve_block_hashes(
                    request.block_hashes,
                    self.hash_block_size,
                    group_block_size,
                )
                first = source_start // group_block_size
                last = min(
                    (source_end + group_block_size - 1) // group_block_size,
                    len(group_hashes),
                )
                for ordinal in range(first, last):
                    aliases.append(
                        make_block_hash_with_group_id(
                            group_hashes[ordinal], group_idx
                        )
                    )
        return tuple(aliases)

    def _release_transfer_refs(self, transfer: TransferMeta) -> None:
        """Release transfer refs without making copied data cacheable."""
        cpu_blocks = [self.cpu_block_pool.blocks[bid] for bid in transfer.cpu_block_ids]
        for cpu_block in cpu_blocks:
            cpu_block.reset_hash()
        self.cpu_block_pool.free_blocks(cpu_blocks)
        assert self._gpu_block_pool is not None
        self._gpu_block_pool.free_blocks(
            self._gpu_block_pool.blocks[bid] for bid in transfer.gpu_block_ids
        )

    def has_pending_stores(self) -> bool:
        """Return True if there are in-flight store transfers."""
        return bool(
            self._store_event_to_blocks or self._abandoned_store_event_to_blocks
        )

    def has_pending_transfers(self) -> bool:
        return bool(
            self._store_event_to_blocks
            or self._reqs_to_load
            or self._load_event_to_reqs
        )

    def has_pending_store_transfers(self) -> bool:
        """Return whether a store is still awaiting scheduler completion.

        Loads and stores use independent worker streams. Lazy store admission
        must therefore not serialize a store behind an unrelated H2D load.
        """
        return bool(self._store_event_to_blocks)

    def _store_transfer_capacity_available(self) -> bool:
        """Bound lazy store admission by the per-direction queue capacity."""
        return len(self._store_event_to_blocks) < self._transfer_queue_depth

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Always returns (False, None). GPU blocks are protected by ref_cnt,
        so the scheduler can free blocks immediately."""
        req_id = request.request_id

        # Release any temp CPU hit pin from get_num_new_matched_tokens()
        # if request is canceled or preempted before update_state_after_alloc()
        pending = self._pending_cpu_hits.pop(req_id, None)
        if pending is not None:
            self._free_pending_cpu_hit(pending)

        # Handle load: defer cleanup if load is in-flight
        load_state = self._reqs_to_load.get(req_id)
        if load_state is not None:
            if load_state.load_event is not None:
                load_state.finished = True  # Defer: load in-flight
            else:
                self._cleanup_load_request(req_id)

        store_state = self._reqs_to_store.get(req_id)
        if store_state is not None:
            if store_state.store_events:
                store_state.finished = True
            elif self._lazy_mode and not any(store_state.block_ids):
                self._cleanup_store_request(req_id)
            else:
                store_state.finished = True

        if store_state is None and block_ids:
            store_state = StoreRequestState(
                request=request,
                block_ids=(list(block_ids),),
                num_stored_blocks=[0],
                finished=True,
            )
            self._reqs_to_store[req_id] = store_state
            self._record_gpu_blocks(req_id, store_state.block_ids, request=request)

        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if block_ids:
            state = self._reqs_to_store.get(request.request_id)
            if state is None:
                self._reqs_to_store[request.request_id] = StoreRequestState(
                    request=request,
                    block_ids=tuple(list(group) for group in block_ids),
                    num_stored_blocks=[0] * len(block_ids),
                )
            else:
                state.block_ids = tuple(list(group) for group in block_ids)
                if len(state.num_stored_blocks) != len(block_ids):
                    state.num_stored_blocks = [0] * len(block_ids)
        return self.request_finished(request, block_ids=[])

    def _free_pending_cpu_hit(self, pending: tuple) -> None:
        """Release the temporary CPU block pin taken in get_num_new_matched_tokens()."""
        cpu_hit_blocks, _ = pending
        blocks_to_free = [
            blk for grp in cpu_hit_blocks for blk in grp if not blk.is_null
        ]
        if blocks_to_free:
            self.cpu_block_pool.free_blocks(blocks_to_free)

    def _cleanup_load_request(self, req_id: str) -> None:
        """Release all load resources for a request.

        Shared between request_finished() and update_connector_output() paths.
        Removes the request from _reqs_to_load, cleans up event mappings,
        and frees CPU/GPU touch refs.
        """
        state = self._reqs_to_load.pop(req_id, None)
        if state is None:
            state = self._abandoned_reqs_to_load.pop(req_id, None)
        if state is None:
            return
        # Remove from load event mapping (only this req, not whole event)
        if state.load_event is not None:
            reqs = self._load_event_to_reqs.get(state.load_event)
            if reqs is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._load_event_to_reqs.pop(state.load_event, None)

            if not self._load_event_to_reqs.get(state.load_event):
                perf = self._load_event_perf.pop(state.load_event, None)
                if perf is not None:
                    elapsed_s = max(time.monotonic() - perf.started_at_s, 1e-9)
                    self._update_estimated_swap_bandwidth(
                        perf.bytes_to_copy, elapsed_s
                    )
                    elapsed_ms = elapsed_s * 1000
                    self._telemetry_swap_in_time_ms += elapsed_ms
                    local_bytes, remote_bytes = self._split_local_remote_bytes(
                        self._load_event_localities.pop(state.load_event, []),
                        self._count_transfer_bytes(1),
                    )
                    self._telemetry_local_swap_in_bytes += local_bytes
                    self._telemetry_remote_swap_in_bytes += remote_bytes
                    local_ms, remote_ms = split_elapsed_ms_by_bytes(
                        elapsed_ms,
                        local_bytes=local_bytes,
                        remote_bytes=remote_bytes,
                    )
                    self._telemetry_local_swap_in_time_ms += local_ms
                    self._telemetry_remote_swap_in_time_ms += remote_ms

        if state.transfer_meta is not None:
            self._mark_residency_for_gpu_blocks(
                state.transfer_meta.gpu_block_ids,
                "synced",
                cpu_block_ids=state.transfer_meta.cpu_block_ids,
            )
            # Free CPU touch refs
            self.cpu_block_pool.free_blocks(
                self.cpu_block_pool.blocks[bid]
                for bid in state.transfer_meta.cpu_block_ids
            )
            # Free GPU touch refs
            assert self._gpu_block_pool is not None
            self._gpu_block_pool.free_blocks(
                self._gpu_block_pool.blocks[bid]
                for bid in state.transfer_meta.gpu_block_ids
            )
        if state.finished:
            self._drop_residency_request(req_id)

    def _cleanup_store_request(self, req_id: str) -> None:
        """Release store metadata for a request.

        Metadata-only cleanup but no block freeing. Job completion handles
        block caching and GPU ref freeing via _process_store_completion().
        """
        state = self._reqs_to_store.pop(req_id, None)
        if state is None:
            return
        for event_idx in list(state.store_events):
            if (reqs := self._store_event_to_reqs.get(event_idx)) is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._store_event_to_reqs.pop(event_idx, None)
        state.store_events.clear()
        self._drop_residency_request(req_id)

    def take_events(self) -> Iterable[KVCacheEvent]:
        return self.cpu_block_pool.take_events()

    def reset(self) -> bool:
        """Abandon pending transfers and reset the CPU cache when safe.

        Worker-side DMA may still be using blocks after reset is requested.
        Keep those block refs pinned until the existing completion path reports
        the transfer finished, then release refs without caching abandoned
        store results.
        """

        self._abandoned_store_event_to_blocks.update(self._store_event_to_blocks)
        self._store_event_to_blocks.clear()
        self._in_flight_store_gpu_blocks.clear()

        # Loads that have not been sent to the worker cannot have running DMA.
        # In-flight loads stay pinned and are cleaned up on completion.
        for req_id in list(self._reqs_to_load):
            state = self._reqs_to_load.pop(req_id)
            if state.load_event is None:
                self._reqs_to_load[req_id] = state
                self._cleanup_load_request(req_id)
            else:
                self._abandoned_reqs_to_load[req_id] = state

        self._reqs_to_store.clear()
        self._residency.clear()
        self._residency_by_gpu_block.clear()
        self._residency_keys_by_request.clear()
        self._gpu_block_hash_aliases.clear()
        self._gpu_block_primary_hashes.clear()
        self._store_event_to_reqs.clear()
        self._store_event_pending_counts = {
            event_idx: count
            for event_idx, count in self._store_event_pending_counts.items()
            if event_idx in self._abandoned_store_event_to_blocks
        }
        self._cursor = None
        # NOTE: _load_event_counter / _store_event_counter are not
        # reset as they are monotonic and must stay ahead of the workers
        # high-water marks to avoid event index collisions

        if self._abandoned_store_event_to_blocks or self._abandoned_reqs_to_load:
            return False

        return self.cpu_block_pool.reset_prefix_cache()
