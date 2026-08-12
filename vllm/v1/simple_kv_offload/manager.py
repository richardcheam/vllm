# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side manager for SimpleCPUOffloadConnector."""

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
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestRotaryState
from vllm.v1.simple_kv_offload.layout import choose_layout_mode
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)
from vllm.v1.simple_kv_offload.topology import (
    Gh200TopologyMapping,
    LocalityPoolPlanner,
    discover_gh200_topology_mapping,
    estimate_remote_penalty_seconds,
    split_elapsed_ms_by_bytes,
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


@dataclass
class TransferPerfState:
    started_at_s: float
    bytes_to_copy: int


class SimpleCPUOffloadScheduler:
    """Scheduler-side manager for CPU offloading."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        lazy_offload: bool = False,
        debug_single_request_swap: bool = False,
        pin_memory_fix: bool = False,
        swapper_block_first: bool = False,
        min_lazy_store_batch_blocks: int = 1,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.enable_kv_cache_events = (
            vllm_config.kv_events_config is not None
            and vllm_config.kv_events_config.enable_kv_cache_events
        )
        # NOTE: We use the same block size for both GPU and CPU.
        self.block_size = vllm_config.cache_config.block_size
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

        logger.info(
            "SimpleCPUOffloadScheduler: Allocating %d CPU blocks (%.2f GB, mode=%s, debug_single_request_swap=%s)",
            self.num_cpu_blocks,
            cpu_capacity_bytes / (1024**3),
            "lazy" if lazy_offload else "eager",
            debug_single_request_swap,
        )

        # TODO (yifan): maybe need to enable kv_cache_events and metrics_collector here.
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
        assert dcp_world_size == 1 and pcp_world_size == 1
        self.cpu_coordinator: KVCacheCoordinator = get_kv_cache_coordinator(
            kv_cache_config=self.cpu_kv_cache_config,
            max_model_len=vllm_config.model_config.max_model_len,
            max_num_batched_tokens=(
                vllm_config.scheduler_config.max_num_batched_tokens
            ),
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=self.block_size,
        )
        self.cpu_block_pool: BlockPool = self.cpu_coordinator.block_pool

        # GPU block pool reference - bound after scheduler builds kv_cache_manager
        self._gpu_block_pool: BlockPool | None = None

        # Load metadata
        self._reqs_to_load: dict[str, LoadRequestState] = {}
        # Inverse map: load_event_idx -> req_ids. Keyed by load_event_idx because
        # the worker reports completions by event index, not request id.
        self._load_event_to_reqs: dict[int, list[str]] = {}

        # Store metadata
        self._lazy_mode = lazy_offload
        self._debug_single_request_swap = debug_single_request_swap
        self._pin_memory_fix = pin_memory_fix
        self._swapper_block_first = swapper_block_first
        extra_config = (
            vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        )
        self._superinfer_high_risk_mode = bool(
            vllm_config.cache_config.superinfer_high_risk_mode
        )
        self._gh200_topology_tuned = bool(
            extra_config.get("gh200_topology_tuned", False)
            or vllm_config.cache_config.gh200_topology_tuned
        )
        self._gpu_rank = int(vllm_config.parallel_config.rank)
        self._local_fraction = float(
            extra_config.get(
                "local_cpu_pool_fraction",
                vllm_config.cache_config.local_cpu_pool_fraction,
            )
        )
        self._local_bw_bytes_per_s = float(
            extra_config.get(
                "local_swap_bandwidth_bytes_per_s",
                vllm_config.cache_config.local_swap_bandwidth_bytes_per_s,
            )
        )
        self._remote_bw_bytes_per_s = float(
            extra_config.get(
                "remote_swap_bandwidth_bytes_per_s",
                vllm_config.cache_config.remote_swap_bandwidth_bytes_per_s,
            )
        )
        self._topology_mapping: Gh200TopologyMapping = discover_gh200_topology_mapping(
            enabled=self._gh200_topology_tuned,
            local_cpu_pool_fraction=self._local_fraction,
            local_bandwidth_bytes_per_s=self._local_bw_bytes_per_s,
            remote_bandwidth_bytes_per_s=self._remote_bw_bytes_per_s,
        )
        for line in self._topology_mapping.to_log_lines():
            if self._topology_mapping.fallback_mode and self._gh200_topology_tuned:
                logger.warning(line)
            else:
                logger.info(line)
        self._min_lazy_store_batch_blocks = max(1, int(min_lazy_store_batch_blocks))
        model_arches = tuple(vllm_config.model_config.architectures or ())
        layout_decision = choose_layout_mode(
            self._swapper_block_first,
            model_arches=model_arches,
            num_kv_cache_groups=len(kv_cache_config.kv_cache_groups),
            has_non_tensor_values=False,
            tensor_parallel_size=vllm_config.parallel_config.tensor_parallel_size,
            aggressive_mode=self._superinfer_high_risk_mode,
        )
        self._block_first_eligible_estimate = layout_decision.block_first_eligible
        self._active_layout_mode = layout_decision.mode
        self._proactive_swap_budget = max(
            0, int(vllm_config.scheduler_config.proactive_swap_budget)
        )
        # Lazy mode: use a cursor to track the last scanned block in the GPU free queue.
        self._cursor: KVCacheBlock | None = None
        estimated_target_free = 0
        if self._lazy_mode:
            estimated_target_free = self._estimate_lazy_target_blocks(
                kv_cache_config,
                vllm_config.scheduler_config.max_num_batched_tokens,
            )
            if self._proactive_swap_budget > 0:
                # Step-6 guarded rollout: allow proactive budget to directly
                # control lazy offload scan depth. This turns the budget into
                # an active scheduler-side movement knob when swap is enabled.
                self._target_free = self._proactive_swap_budget
            else:
                self._target_free = estimated_target_free
        else:
            self._target_free = 0
        logger.info(
            "SimpleCPUOffloadScheduler: lazy_target_free_blocks=%d (estimated=%d, proactive_swap_budget=%d)",
            self._target_free,
            estimated_target_free,
            self._proactive_swap_budget,
        )
        self._store_event_to_blocks: dict[int, TransferMeta] = {}
        # Tracks confirmed full blocks already scheduled/stored to CPU.  In
        # lazy mode this gives us paper-style synced block residency before a
        # later proactive preemption needs to discard GPU blocks.
        self._reqs_to_store: dict[str, StoreRequestState] = {}
        self._store_event_to_reqs: dict[int, list[str]] = {}

        # Event counters
        self._load_event_counter: int = 0
        self._store_event_counter: int = 0

        # For TP/PP: track partial store completions across steps.
        # Events must be reported by all world_size workers before considered complete.
        self._expected_worker_count = vllm_config.parallel_config.world_size
        self._store_event_pending_counts: dict[int, int] = {}

        # Telemetry-only counters.
        self._telemetry_store_events: int = 0
        self._telemetry_load_events: int = 0
        self._telemetry_store_blocks: int = 0
        self._telemetry_load_blocks: int = 0
        self._telemetry_store_bytes: int = 0
        self._telemetry_load_bytes: int = 0
        self._telemetry_store_events_ge_16_blocks: int = 0
        self._telemetry_store_events_lt_4_blocks: int = 0
        self._telemetry_load_events_ge_16_blocks: int = 0
        self._telemetry_load_events_lt_4_blocks: int = 0

        # Rolling estimate used by guarded proactive VLT scoring.
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
        logger.info(
            "SimpleCPUOffloadScheduler locality pools rank=%d sizes=%s",
            self._gpu_rank,
            self._locality_planner.pool_sizes_per_island(),
        )
        self._pending_request_ids_by_store_event: dict[int, list[str]] = {}
        self._pending_request_ids_by_load_event: dict[int, list[str]] = {}

        self._telemetry_local_swap_out_bytes: int = 0
        self._telemetry_local_swap_in_bytes: int = 0
        self._telemetry_remote_swap_out_bytes: int = 0
        self._telemetry_remote_swap_in_bytes: int = 0
        self._telemetry_num_remote_fallbacks: int = 0
        self._telemetry_swap_out_time_ms: float = 0.0
        self._telemetry_swap_in_time_ms: float = 0.0
        self._telemetry_local_swap_out_time_ms: float = 0.0
        self._telemetry_local_swap_in_time_ms: float = 0.0
        self._telemetry_remote_swap_out_time_ms: float = 0.0
        self._telemetry_remote_swap_in_time_ms: float = 0.0

    @property
    def telemetry_stats(self) -> dict[str, int]:
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
            "offload_cpu_free_blocks": self.cpu_block_pool.get_num_free_blocks(),
            "offload_cpu_used_blocks": self.num_cpu_blocks
            - self.cpu_block_pool.get_num_free_blocks(),
            "offload_lazy_target_free_blocks": self._target_free,
            "offload_proactive_swap_budget": self._proactive_swap_budget,
            "offload_gh200_topology_fallback_mode": int(
                self._topology_mapping.fallback_mode
            ),
            "offload_pin_memory_fix": int(self._pin_memory_fix),
            "offload_swapper_block_first": int(self._swapper_block_first),
            "offload_high_risk_mode": int(self._superinfer_high_risk_mode),
            "offload_gh200_topology_tuned": int(self._gh200_topology_tuned),
            "offload_gh200_topology_discovered": int(
                self._topology_mapping.discovered
            ),
            "offload_gh200_gpu_rank": self._gpu_rank,
            "offload_local_cpu_pool_blocks": self._locality_planner.local_total_blocks,
            "offload_remote_cpu_pool_blocks": self._locality_planner.remote_total_blocks,
            "offload_min_lazy_store_batch_blocks": self._min_lazy_store_batch_blocks,
            "offload_block_first_eligible": int(self._block_first_eligible_estimate),
            "offload_block_first_active": int(self._active_layout_mode == "block_first"),
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
            "local_bandwidth_gbps": float(self._local_bw_bytes_per_s * 8 / 1e9),
            "remote_bandwidth_gbps": float(self._remote_bw_bytes_per_s * 8 / 1e9),
            f"local_swap_out_bytes_gpu_{self._gpu_rank}": self._telemetry_local_swap_out_bytes,
            f"local_swap_in_bytes_gpu_{self._gpu_rank}": self._telemetry_local_swap_in_bytes,
            f"remote_swap_out_bytes_gpu_{self._gpu_rank}": self._telemetry_remote_swap_out_bytes,
            f"remote_swap_in_bytes_gpu_{self._gpu_rank}": self._telemetry_remote_swap_in_bytes,
            f"num_remote_fallbacks_gpu_{self._gpu_rank}": self._telemetry_num_remote_fallbacks,
        }

    def take_telemetry_stats(self) -> dict[str, int]:
        stats = self.telemetry_stats
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
        return stats

    def _assign_cpu_block_localities(
        self,
        cpu_block_ids: list[int],
        *,
        source_gpu_rank: int,
        req_id: str,
    ) -> list[str]:
        labels = self._locality_planner.labels_for_block_ids(
            cpu_block_ids,
            source_gpu_rank=source_gpu_rank,
        )
        remote_fallbacks = sum(1 for label in labels if label == "remote")
        if remote_fallbacks:
            self._telemetry_num_remote_fallbacks += remote_fallbacks
            logger.warning(
                "GH200 remote fallback req_id=%s gpu_rank=%d reason=local_pool_exhausted remote_blocks=%d",
                req_id,
                source_gpu_rank,
                remote_fallbacks,
            )
        return labels

    @staticmethod
    def _count_local_remote_bytes(labels: list[str], bytes_per_block: int) -> tuple[int, int]:
        if not labels or bytes_per_block <= 0:
            return 0, 0
        local_blocks = sum(1 for label in labels if label == "local")
        remote_blocks = len(labels) - local_blocks
        return local_blocks * bytes_per_block, remote_blocks * bytes_per_block

    def _estimate_request_remote_penalty_seconds(self, request: "Request") -> float:
        if not self._gh200_topology_tuned:
            return 0.0
        unsynced_blocks = self._estimate_request_unsynced_swap_blocks_for_locality(request)
        if unsynced_blocks <= 0:
            return 0.0
        remote_blocks = self._locality_planner.estimate_remote_blocks_for_gpu(
            source_gpu_rank=self._gpu_rank,
            num_blocks=unsynced_blocks,
        )
        if remote_blocks <= 0:
            return 0.0
        labels = ["local"] * max(unsynced_blocks - remote_blocks, 0)
        labels.extend(["remote"] * remote_blocks)
        _, remote_bytes = self._count_local_remote_bytes(
            labels,
            self._count_transfer_bytes(1),
        )
        return estimate_remote_penalty_seconds(
            remote_bytes,
            local_bandwidth_bytes_per_s=self._local_bw_bytes_per_s,
            remote_bandwidth_bytes_per_s=self._remote_bw_bytes_per_s,
        )

    def estimate_request_remote_penalty_seconds(self, request: "Request") -> float:
        return self._estimate_request_remote_penalty_seconds(request)

    def _estimate_request_unsynced_swap_blocks_for_locality(self, request: "Request") -> int:
        request_id = request.request_id
        if request_id not in self._reqs_to_store:
            return 0
        state = self._reqs_to_store[request_id]
        total_blocks = sum(len(group) for group in state.block_ids)
        resident = self.get_num_cpu_resident_owned_blocks(request)
        return max(total_blocks - resident, 0)

    def _build_cpu_block_island_map(self) -> dict[int, int]:
        islands = self._locality_planner.island_ids
        if not islands:
            return {}
        allocatable_ids = [
            blk.block_id for blk in self.cpu_block_pool.blocks if not blk.is_null
        ]
        if not allocatable_ids:
            return {}
        per_island = len(allocatable_ids) // len(islands)
        extra = len(allocatable_ids) % len(islands)
        mapping: dict[int, int] = {}
        cursor = 0
        for i, island in enumerate(islands):
            count = per_island + (1 if i < extra else 0)
            for block_id in allocatable_ids[cursor : cursor + count]:
                mapping[block_id] = island
            cursor += count
        return mapping

    def _choose_cpu_blocks_for_localities(self, labels: list[str]) -> list[int]:
        if not labels:
            return []

        source_island = self._locality_planner.island_for_gpu_rank(self._gpu_rank)
        free_blocks = self.cpu_block_pool.free_block_queue.get_all_free_blocks()
        local_candidates = [
            blk.block_id
            for blk in free_blocks
            if self._cpu_block_island_map.get(blk.block_id, source_island) == source_island
        ]
        remote_candidates = [
            blk.block_id
            for blk in free_blocks
            if self._cpu_block_island_map.get(blk.block_id, source_island) != source_island
        ]

        selected: list[int] = []
        for label in labels:
            if label == "local":
                if local_candidates:
                    selected.append(local_candidates.pop(0))
                    continue
                if remote_candidates:
                    selected.append(remote_candidates.pop(0))
                    continue
            else:
                if remote_candidates:
                    selected.append(remote_candidates.pop(0))
                    continue
                if local_candidates:
                    selected.append(local_candidates.pop(0))
                    continue
            break

        if len(selected) != len(labels):
            # Should not happen if caller checks free block count. Fallback to deterministic
            # head allocation for safety and keep behavior fail-open.
            needed = len(labels)
            return [blk.block_id for blk in free_blocks[:needed]]

        return selected

    def _allocate_cpu_blocks_locality_aware(
        self,
        *,
        num_blocks: int,
        req_id: str,
    ) -> tuple[list["KVCacheBlock"], list[str], int]:
        if num_blocks <= 0:
            return [], [], 0

        labels = self._locality_planner.plan_labels_for_allocation(
            num_blocks=num_blocks,
            source_gpu_rank=self._gpu_rank,
        )
        selected_ids = self._choose_cpu_blocks_for_localities(labels)
        cpu_blocks = [self.cpu_block_pool.blocks[block_id] for block_id in selected_ids]

        for block in cpu_blocks:
            if block.ref_cnt == 0 and not block.is_null:
                self.cpu_block_pool.free_block_queue.remove(block)
            if self.cpu_block_pool.enable_caching:
                self.cpu_block_pool._maybe_evict_cached_block(block)
            assert block.ref_cnt == 0
            block.ref_cnt += 1
            if self.cpu_block_pool.metrics_collector:
                self.cpu_block_pool.metrics_collector.on_block_allocated(block)

        remote_fallbacks = self._locality_planner.record_block_assignments(
            selected_ids,
            labels=labels,
            source_gpu_rank=self._gpu_rank,
            req_id=req_id,
        )
        if remote_fallbacks:
            self._telemetry_num_remote_fallbacks += remote_fallbacks
            logger.warning(
                "GH200 remote fallback req_id=%s gpu_rank=%d reason=local_pool_exhausted remote_blocks=%d",
                req_id,
                self._gpu_rank,
                remote_fallbacks,
            )
        return cpu_blocks, labels, remote_fallbacks

    def get_estimated_swap_bandwidth_bytes_per_s(self) -> float | None:
        return self._estimated_swap_bandwidth_bytes_per_s

    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_block_pool.get_num_free_blocks()

    def get_num_total_cpu_blocks(self) -> int:
        return self.num_cpu_blocks

    def get_num_cpu_resident_blocks(self, request: "Request") -> int:
        """Return consecutive full request blocks already resident on CPU.

        This is a scheduler-facing estimate for SuperInfer-style synced block
        residency. It intentionally uses the existing CPU prefix-cache lookup so
        DeepSeek-V4 stays layout-opaque and safe under the current gpu-derived
        copy path.
        """
        if not request.block_hashes:
            return 0
        max_hit_len = max(request.num_tokens - 1, 0)
        if max_hit_len <= 0:
            return 0

        cpu_hit_blocks, hit_length = self.cpu_coordinator.find_longest_cache_hit(
            request.block_hashes,
            max_hit_len,
        )
        if hit_length <= 0:
            return 0
        return sum(
            1 for group in cpu_hit_blocks for block in group if not block.is_null
        )

    def get_num_cpu_resident_owned_blocks(self, request: "Request") -> int:
        """Return count of CPU-resident full blocks owned by this request.

        A block is considered owned by the request iff its current GPU block
        refcount is 1. Shared-prefix blocks (refcount > 1) are excluded.
        """
        store_state = self._reqs_to_store.get(request.request_id)
        if store_state is None or self._gpu_block_pool is None:
            return 0

        cached = self.cpu_block_pool.cached_block_hash_to_block
        owned_resident = 0
        for group_ids in store_state.block_ids:
            for gpu_block_id in group_ids:
                gpu_block = self._gpu_block_pool.blocks[gpu_block_id]
                bhash = gpu_block.block_hash
                if gpu_block.is_null or bhash is None:
                    continue
                if gpu_block.ref_cnt > 1:
                    continue
                if cached.get_one_block(bhash) is not None:
                    owned_resident += 1

        return owned_resident

    def _update_estimated_swap_bandwidth(
        self, bytes_to_copy: int, elapsed_s: float
    ) -> None:
        if bytes_to_copy <= 0 or elapsed_s <= 0:
            return

        measured = bytes_to_copy / elapsed_s
        if not math.isfinite(measured) or measured <= 0:
            return

        current = self._estimated_swap_bandwidth_bytes_per_s
        if current is None:
            self._estimated_swap_bandwidth_bytes_per_s = measured
            return

        alpha = self._bandwidth_ema_alpha
        self._estimated_swap_bandwidth_bytes_per_s = (
            (1.0 - alpha) * current + alpha * measured
        )

    def _count_transfer_bytes(self, num_blocks: int) -> int:
        if num_blocks <= 0:
            return 0
        if self.cpu_kv_cache_config.num_blocks <= 0:
            return 0
        total_bytes = sum(t.size for t in self.cpu_kv_cache_config.kv_cache_tensors)
        if total_bytes <= 0:
            return 0
        bytes_per_block = total_bytes // self.cpu_kv_cache_config.num_blocks
        return num_blocks * bytes_per_block

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

        gpu_total_bytes = sum(t.size for t in gpu_config.kv_cache_tensors)
        num_gpu_blocks = gpu_config.num_blocks
        num_cpu_blocks = max(1, num_gpu_blocks * cpu_capacity_bytes // gpu_total_bytes)
        # Create CPU kv_cache_tensors mirroring GPU by scaling size proportionally.
        cpu_tensors = [
            KVCacheTensor(
                size=t.size // num_gpu_blocks * num_cpu_blocks,
                shared_by=list(t.shared_by),
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
        kv_cache_config: "KVCacheConfig", max_num_batched_tokens: int
    ) -> int:
        """GPU blocks to keep available (free/offloaded) per step in lazy mode."""
        WATERMARK_RATIO = 1.0  # Reserve larger space to avoid running out of GPU blocks
        target = 0
        for g in kv_cache_config.kv_cache_groups:
            spec = g.kv_cache_spec
            if isinstance(spec, MambaSpec):
                target += 2
            elif isinstance(spec, SlidingWindowSpec):
                target += cdiv(spec.sliding_window, spec.block_size) + 1
            else:
                target += cdiv(max_num_batched_tokens, spec.block_size)
        return int(target * (1 + WATERMARK_RATIO))

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        """Bind GPU block pool so that we can touch blocks during stores.
        Called by Scheduler after kv_cache_manager is ready."""
        self._gpu_block_pool = gpu_block_pool

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Return (num_new_tokens, is_async) from consecutive CPU cache hits."""
        skipped = num_computed_tokens // self.block_size
        remaining_hashes = request.block_hashes[skipped:]

        if not remaining_hashes:
            return 0, False
        # Must recompute at least the last token, matching the logic in
        # kv_cache_manager.get_computed_blocks().
        max_hit_len = request.num_tokens - 1 - num_computed_tokens
        if max_hit_len <= 0:
            return 0, False

        if request.rotary_state in (
            RequestRotaryState.ROTARY_SWAPPED,
            RequestRotaryState.ROTARY_PENDING_IN,
        ):
            synced_cap = max(int(request.rotary_synced_blocks), 0) * self.block_size
            max_hit_len = min(max_hit_len, synced_cap)

            dirty_tail = max(int(request.rotary_dirty_tail_tokens), 0)
            if dirty_tail > 0:
                clean_limit = max(request.num_tokens - dirty_tail - num_computed_tokens, 0)
                max_hit_len = min(max_hit_len, clean_limit)

            if max_hit_len <= 0:
                return 0, False

        _, hit_length = self.cpu_coordinator.find_longest_cache_hit(
            remaining_hashes, max_hit_len
        )

        if hit_length > 0:
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

        # Store tracking.  In eager mode this is the main store path.  In lazy
        # mode it opportunistically stores confirmed full blocks while the
        # request is still running, so later preemption can reuse CPU-resident
        # synced blocks instead of copying everything at the point of eviction.
        should_track_store = (
            not self._lazy_mode
            or req_id in self._reqs_to_store
            or self._is_lazy_store_pressure_active()
        )
        if req_id not in self._reqs_to_store:
            if should_track_store:
                self._reqs_to_store[req_id] = StoreRequestState(
                    request=request,
                    block_ids=tuple([] for _ in range(num_groups)),
                    num_stored_blocks=[0] * num_groups,
                )

        if num_external_tokens == 0:
            return

        num_blocks_to_load = num_external_tokens // self.block_size
        assert num_blocks_to_load > 0

        skipped = sum(blk.block_hash is not None for blk in blocks.blocks[self.fa_gidx])
        num_computed_tokens = skipped * self.block_size
        hashes_to_load = request.block_hashes[skipped : skipped + num_blocks_to_load]

        # Find CPU cached blocks across all groups.
        max_hit_len = len(hashes_to_load) * self.block_size
        cpu_hit_blocks, hit_length = self.cpu_coordinator.find_longest_cache_hit(
            hashes_to_load, max_hit_len
        )
        assert hit_length == num_external_tokens, (
            f"Expected {num_external_tokens} hit tokens, got {hit_length}"
        )

        # Build transfer pairs across all groups.
        total_computed_tokens = num_computed_tokens + num_external_tokens
        kv_cache_groups = self.cpu_kv_cache_config.kv_cache_groups

        gpu_block_ids: list[int] = []
        cpu_block_ids: list[int] = []
        cpu_blocks_to_touch: list[KVCacheBlock] = []

        for g in range(num_groups):
            cpu_blocks_g = cpu_hit_blocks[g]
            n_ext_g = len(cpu_blocks_g)
            if n_ext_g == 0:
                continue

            # Number of blocks in the computed range for this group.
            g_block_size = kv_cache_groups[g].kv_cache_spec.block_size
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
                source_gpu_rank=self._gpu_rank,
            ),
        )

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
            store_bytes = self._count_transfer_bytes(len(store_gpu))
            store_localities = self._locality_planner.labels_for_block_ids(
                store_cpu,
                source_gpu_rank=self._gpu_rank,
            )
            self._store_event_to_blocks[store_event] = TransferMeta(
                store_gpu,
                store_cpu,
                store_localities,
                source_gpu_rank=self._gpu_rank,
            )
            self._pending_request_ids_by_store_event[store_event] = list(store_req_ids)
            self._store_event_localities[store_event] = store_localities
            self._store_event_perf[store_event] = TransferPerfState(
                started_at_s=time.monotonic(),
                bytes_to_copy=store_bytes,
            )
            self._telemetry_store_events += 1
            self._telemetry_store_blocks += len(store_gpu)
            self._telemetry_store_bytes += store_bytes
            if len(store_gpu) >= 16:
                self._telemetry_store_events_ge_16_blocks += 1
            if 0 < len(store_gpu) < 4:
                self._telemetry_store_events_lt_4_blocks += 1
            if store_req_ids:
                self._store_event_to_reqs[store_event] = store_req_ids
                for req_id in store_req_ids:
                    store_state = self._reqs_to_store.get(req_id)
                    if store_state is not None:
                        store_state.store_events.add(store_event)

        # --- Loads ---
        load_event = -1
        load_gpu: list[int] = []
        load_cpu: list[int] = []
        load_req_ids: list[str] = [
            req_id
            for req_id, load_state in self._reqs_to_load.items()
            if load_state.load_event is None
        ]
        if self._debug_single_request_swap and load_req_ids:
            load_req_ids = load_req_ids[:1]

        for req_id in load_req_ids:
            load_state = self._reqs_to_load[req_id]
            assert load_state.transfer_meta is not None
            load_gpu.extend(load_state.transfer_meta.gpu_block_ids)
            load_cpu.extend(load_state.transfer_meta.cpu_block_ids)

        if load_req_ids:
            load_event = self._load_event_counter
            self._load_event_counter += 1
            load_bytes = self._count_transfer_bytes(len(load_gpu))
            load_localities = self._assign_cpu_block_localities(
                load_cpu,
                source_gpu_rank=self._gpu_rank,
                req_id=",".join(load_req_ids),
            )
            for req_id in load_req_ids:
                self._reqs_to_load[req_id].load_event = load_event
            self._load_event_to_reqs[load_event] = load_req_ids
            self._pending_request_ids_by_load_event[load_event] = list(load_req_ids)
            self._load_event_localities[load_event] = load_localities
            self._load_event_perf[load_event] = TransferPerfState(
                started_at_s=time.monotonic(),
                bytes_to_copy=load_bytes,
            )
            self._telemetry_load_events += 1
            self._telemetry_load_blocks += len(load_gpu)
            self._telemetry_load_bytes += load_bytes
            if len(load_gpu) >= 16:
                self._telemetry_load_events_ge_16_blocks += 1
            if 0 < len(load_gpu) < 4:
                self._telemetry_load_events_lt_4_blocks += 1

        result = SimpleCPUOffloadMetadata(
            load_event=load_event,
            load_gpu_blocks=load_gpu,
            load_cpu_blocks=load_cpu,
            load_event_to_reqs=self._load_event_to_reqs,
            load_cpu_block_localities=self._load_event_localities,
            store_event=store_event,
            store_gpu_blocks=store_gpu,
            store_cpu_blocks=store_cpu,
            store_cpu_block_localities=store_localities if store_gpu else [],
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
            # Throughput-first policy: avoid stacking new D2H work while prior
            # load/store transfers are still in flight.
            if self.has_pending_transfers():
                self._cleanup_finished_store_requests_without_pending_events()
                return [], [], []
            if not self._has_lazy_store_pressure(scheduler_output):
                self._cleanup_finished_store_requests_without_pending_events()
                return [], [], []
            synced_gpu, synced_cpu, synced_req_ids = self._prepare_eager_store_specs(
                scheduler_output
            )
            if self._debug_single_request_swap and synced_gpu:
                return synced_gpu, synced_cpu, synced_req_ids
            gpu_pool = self._gpu_block_pool
            force_small_batch = bool(scheduler_output.preempted_req_ids)
            if gpu_pool is not None and self._target_free > 0:
                force_small_batch = force_small_batch or (
                    gpu_pool.get_num_free_blocks() <= self._target_free
                )
            lazy_gpu, lazy_cpu, _ = self._prepare_lazy_store_specs(
                force_small_batch=force_small_batch
            )
            return synced_gpu + lazy_gpu, synced_cpu + lazy_cpu, synced_req_ids

        return self._prepare_eager_store_specs(scheduler_output)

    def _has_lazy_store_pressure(self, scheduler_output: SchedulerOutput) -> bool:
        """Return whether lazy mode should prepare CPU store work this step."""
        if scheduler_output.preempted_req_ids:
            return True
        gpu_pool = self._gpu_block_pool
        if gpu_pool is None or self._target_free <= 0:
            return False

        # Lazy offload is a pressure valve, not a steady-state tax. Keep a
        # compact pre-store watermark so synced blocks can be prepared shortly
        # before allocator pressure, while avoiding excessive steady D2H traffic.
        prestore_target = self._target_free + 1
        return gpu_pool.get_num_free_blocks() <= prestore_target

    def _is_lazy_store_pressure_active(self) -> bool:
        """Fast local pressure check for lazy-mode bookkeeping decisions."""
        gpu_pool = self._gpu_block_pool
        if gpu_pool is None or self._target_free <= 0:
            return False
        prestore_target = self._target_free + 1
        return gpu_pool.get_num_free_blocks() <= prestore_target

    def _cleanup_finished_store_requests_without_pending_events(self) -> None:
        """Drop finished metadata in lazy mode when no store work is required."""
        for req_id, state in list(self._reqs_to_store.items()):
            if state.finished and not state.store_events:
                self._cleanup_store_request(req_id)

    def _prepare_lazy_store_specs(
        self,
        force_small_batch: bool = False,
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

        # Determine start node.
        if self._cursor is None:
            node = free_queue.fake_free_list_head.next_free_block
        else:
            node = self._cursor.next_free_block

        tail = free_queue.fake_free_list_tail
        gpu_ids: list[int] = []
        block_hashes: list[bytes] = []
        covered = 0
        last_visited = self._cursor

        while (
            node is not None
            and node is not tail
            and covered < self._target_free
            and len(gpu_ids) < num_cpu_free
        ):
            last_visited = node
            bhash = node.block_hash

            if (
                bhash is not None
                and not node.is_null
                and cpu_pool.cached_block_hash_to_block.get_one_block(bhash) is None
            ):
                gpu_ids.append(node.block_id)
                block_hashes.append(bhash)

            covered += 1
            node = node.next_free_block

        self._cursor = last_visited

        min_batch = self._min_lazy_store_batch_blocks
        if self._target_free <= 1:
            # Keep tiny-budget behavior unchanged to avoid starving prestore
            # under very tight headroom.
            min_batch = 1
        if not force_small_batch and len(gpu_ids) < min_batch:
            return [], [], []

        if self._debug_single_request_swap and gpu_ids:
            gpu_ids = gpu_ids[:1]
            block_hashes = block_hashes[:1]

        # Batch-allocate CPU blocks and stamp hashes.
        if gpu_ids:
            cpu_blocks, _, _ = self._allocate_cpu_blocks_locality_aware(
                num_blocks=len(gpu_ids),
                req_id="<lazy-scan>",
            )
            cpu_ids = [blk.block_id for blk in cpu_blocks]
            for cpu_blk, bhash in zip(cpu_blocks, block_hashes):  # type: ignore[assignment]
                cpu_blk._block_hash = bhash  # type: ignore[assignment]
            # Touch GPU blocks to prevent eviction during async copy.
            gpu_pool.touch([gpu_pool.blocks[bid] for bid in gpu_ids])
        else:
            cpu_ids = []

        return gpu_ids, cpu_ids, []

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
        gpu_blocks_this_step: set[int] = set()

        req_new_tokens: dict[str, int] = {}
        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            req_new_tokens[req_id] = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            state = self._reqs_to_store.get(req_id)
            if state is None:
                continue

            # Accumulate new block IDs.
            if preempted:
                state.block_ids = tuple([] for _ in range(num_groups))
                state.num_stored_blocks = [0] * num_groups
            if new_block_id_groups:
                for g in range(min(num_groups, len(new_block_id_groups))):
                    if new_block_id_groups[g] is not None:
                        state.block_ids[g].extend(new_block_id_groups[g])

        for req_id, state in list(self._reqs_to_store.items()):
            if state.finished:
                # Finished requests can still have confirmed full blocks that were
                # not flushed in the same scheduler step they completed.
                pass
            elif req_new_tokens.get(req_id, 0) <= 0:
                continue

            if state.finished and state.store_events:
                continue

            block_ids_by_group = state.block_ids
            if not block_ids_by_group:
                continue

            # --- Phase 1: Scan blocks, classify as cached vs to-store ---
            gpu_block_ids: list[int] = []
            block_hashes_to_store: list[bytes] = []
            advanced_per_group: list[int] = [0] * num_groups
            ready_per_group: list[int] = [0] * num_groups
            out_of_space = False
            # Confirmed tokens: KV data written and visible to all streams.
            req = state.request
            confirmed_tokens = req.num_computed_tokens - req.num_output_placeholders

            for g in range(num_groups):
                # FIXME (yifan): handle CPU cache eviction, where
                # num_stored_blocks can be stale and omit evicted blocks in
                # the middle of the request.
                already_stored_g = state.num_stored_blocks[g]
                group_gpu_ids = block_ids_by_group[g]

                # Cap to blocks with confirmed KV data.
                g_block_size = kv_cache_groups[g].kv_cache_spec.block_size
                ready_blocks_g = confirmed_tokens // g_block_size
                ready_per_group[g] = ready_blocks_g
                scannable = group_gpu_ids[already_stored_g:ready_blocks_g]

                for gpu_block_id in scannable:
                    gpu_block = gpu_block_pool.blocks[gpu_block_id]
                    if gpu_block.is_null:
                        advanced_per_group[g] += 1
                        continue

                    # Shared-prefix block: keep it pinned/shared and avoid
                    # request-local offload ownership assumptions.
                    if (
                        gpu_block.ref_cnt > 1
                        and gpu_block_id not in state.finish_touched_gpu_block_ids
                    ):
                        advanced_per_group[g] += 1
                        continue

                    bhash_with_group = gpu_block.block_hash
                    if bhash_with_group is None:
                        break

                    # Check if this group's data is already scheduled for store
                    # in this step or already cached in CPU.
                    if (
                        gpu_block_id in gpu_blocks_this_step
                        or cpu_block_pool.cached_block_hash_to_block.get_one_block(
                            bhash_with_group
                        )
                        is not None
                    ):
                        advanced_per_group[g] += 1
                        continue

                    if num_free <= 0:
                        out_of_space = True
                        break
                    num_free -= 1

                    gpu_block_ids.append(gpu_block_id)
                    block_hashes_to_store.append(bhash_with_group)
                    advanced_per_group[g] += 1

                if out_of_space:
                    break

            # --- Phase 2: Batch allocate CPU blocks and stamp hashes ---
            n_to_alloc = len(gpu_block_ids)
            if n_to_alloc > 0:
                cpu_blocks_alloc, _, _ = self._allocate_cpu_blocks_locality_aware(
                    num_blocks=n_to_alloc,
                    req_id=req_id,
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
                gpu_blocks_this_step.update(gpu_block_ids)

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

                if self._debug_single_request_swap:
                    break

            # Advance per-group cursors (includes cached hits + newly stored)
            for g in range(num_groups):
                state.num_stored_blocks[g] += advanced_per_group[g]

            if (
                state.finished
                and not state.store_events
                and all(
                    state.num_stored_blocks[g] >= ready_per_group[g]
                    for g in range(num_groups)
                )
            ):
                self._cleanup_store_request(req_id)

        return merged_gpu_block_ids, merged_cpu_block_ids, req_ids

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
        for event_idx, count in meta.completed_store_events.items():
            total = self._store_event_pending_counts.get(event_idx, 0) + count
            if total >= self._expected_worker_count:
                self._store_event_pending_counts.pop(event_idx, None)
                self._process_store_event(event_idx)
            else:
                self._store_event_pending_counts[event_idx] = total

    def _process_store_event(self, event_idx: int) -> None:
        """Process a fully-completed store event."""
        transfer = self._store_event_to_blocks.pop(event_idx)
        self._pending_request_ids_by_store_event.pop(event_idx, [])
        if transfer.cpu_block_ids:
            self._locality_planner.release_block_ids(
                transfer.cpu_block_ids,
                source_gpu_rank=transfer.source_gpu_rank,
            )
        self._process_store_completion(transfer.gpu_block_ids, transfer.cpu_block_ids)
        perf = self._store_event_perf.pop(event_idx, None)
        store_localities = self._store_event_localities.pop(event_idx, [])
        if perf is not None:
            elapsed_s = max(time.monotonic() - perf.started_at_s, 1e-9)
            self._update_estimated_swap_bandwidth(
                perf.bytes_to_copy,
                elapsed_s,
            )
            elapsed_ms = elapsed_s * 1000.0
            self._telemetry_swap_out_time_ms += elapsed_ms
            bytes_per_block = self._count_transfer_bytes(1)
            local_bytes, remote_bytes = self._count_local_remote_bytes(
                store_localities,
                bytes_per_block,
            )
            self._telemetry_local_swap_out_bytes += local_bytes
            self._telemetry_remote_swap_out_bytes += remote_bytes
            local_ms, remote_ms = split_elapsed_ms_by_bytes(
                elapsed_ms,
                local_bytes=local_bytes,
                remote_bytes=remote_bytes,
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
        self, gpu_block_ids: list[int], cpu_block_ids: list[int]
    ) -> None:
        """Cache CPU blocks per-group and release GPU refs.

        Block hashes were stamped on CPU blocks at allocation time (in
        ``_prepare_*_store_specs``).  Here we just register them in the
        cache map so they become discoverable by the load path.
        """
        assert len(cpu_block_ids) == len(gpu_block_ids)

        cpu_blocks = [self.cpu_block_pool.blocks[bid] for bid in cpu_block_ids]

        for cpu_block in cpu_blocks:
            bhash = cpu_block.block_hash
            assert bhash is not None
            self.cpu_block_pool.cached_block_hash_to_block.insert(bhash, cpu_block)

        # Free CPU and GPU blocks' ref counts to turn them into prefix cache
        self.cpu_block_pool.free_blocks(cpu_blocks)
        assert self._gpu_block_pool is not None
        self._gpu_block_pool.free_blocks(
            self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids
        )

    def has_pending_stores(self) -> bool:
        """Return True if there are in-flight store transfers."""
        return bool(self._store_event_to_blocks)

    def has_pending_transfers(self) -> bool:
        """Return True if any load/store transfer is in-flight or pending."""
        if self._store_event_to_blocks:
            return True
        if self._reqs_to_load:
            return True
        if self._load_event_to_reqs:
            return True
        return False

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Always returns (False, None). GPU blocks are protected by ref_cnt,
        so the scheduler can free blocks immediately."""
        req_id = request.request_id

        # Handle load: defer cleanup if load is in-flight
        load_state = self._reqs_to_load.get(req_id)
        if load_state is not None:
            if load_state.load_event is not None:
                load_state.finished = True  # Defer: load in-flight
            else:
                self._cleanup_load_request(req_id)

        # Handle per-request synced store tracking: defer cleanup if stores
        # are still in flight.
        store_state = self._reqs_to_store.get(req_id)
        if store_state is None and block_ids:
            if self._lazy_mode and not self._is_lazy_store_pressure_active():
                return False, None
            store_state = StoreRequestState(
                request=request,
                block_ids=(list(block_ids),),
                num_stored_blocks=[0],
            )
            self._reqs_to_store[req_id] = store_state
        elif store_state is not None and block_ids:
            store_state.block_ids = (list(block_ids),)

        if store_state is not None and self._gpu_block_pool is not None:
            req = store_state.request
            confirmed_tokens = req.num_computed_tokens - req.num_output_placeholders
            gpu_ids_to_touch: list[int] = []
            kv_groups = self.cpu_kv_cache_config.kv_cache_groups
            for g, group_ids in enumerate(store_state.block_ids):
                if g >= len(kv_groups):
                    break
                g_block_size = kv_groups[g].kv_cache_spec.block_size
                ready_blocks_g = min(len(group_ids), confirmed_tokens // g_block_size)
                start = min(store_state.num_stored_blocks[g], ready_blocks_g)
                if start < ready_blocks_g:
                    for bid in group_ids[start:ready_blocks_g]:
                        block = self._gpu_block_pool.blocks[bid]
                        if block.is_null or block.ref_cnt > 1:
                            continue
                        gpu_ids_to_touch.append(bid)
            if gpu_ids_to_touch:
                self._gpu_block_pool.touch(
                    [
                        self._gpu_block_pool.blocks[bid]
                        for bid in gpu_ids_to_touch
                        if not self._gpu_block_pool.blocks[bid].is_null
                    ]
                )
                store_state.finish_touched_gpu_block_ids.update(gpu_ids_to_touch)

        if store_state is not None:
            if store_state.store_events:
                store_state.finished = True
            else:
                store_state.finished = True

        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        req_id = request.request_id
        if req_id not in self._reqs_to_store and block_ids:
            self._reqs_to_store[req_id] = StoreRequestState(
                request=request,
                block_ids=tuple(list(group_ids) for group_ids in block_ids),
                num_stored_blocks=[0] * len(block_ids),
            )
        elif req_id in self._reqs_to_store and block_ids:
            state = self._reqs_to_store[req_id]
            state.block_ids = tuple(list(group_ids) for group_ids in block_ids)
            if len(state.num_stored_blocks) != len(state.block_ids):
                state.num_stored_blocks = [0] * len(state.block_ids)

        return self.request_finished(request, block_ids=[])

    def _cleanup_load_request(self, req_id: str) -> None:
        """Release all load resources for a request.

        Shared between request_finished() and update_connector_output() paths.
        Removes the request from _reqs_to_load, cleans up event mappings,
        and frees CPU/GPU touch refs.
        """
        state = self._reqs_to_load.pop(req_id, None)
        if state is None:
            return
        # Remove from load event mapping (only this req, not whole event)
        completed_event_idx: int | None = None
        if state.load_event is not None:
            reqs = self._load_event_to_reqs.get(state.load_event)
            if reqs is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._load_event_to_reqs.pop(state.load_event, None)
                    completed_event_idx = state.load_event

        if completed_event_idx is not None:
            _ = self._pending_request_ids_by_load_event.pop(completed_event_idx, [])
            perf = self._load_event_perf.pop(completed_event_idx, None)
            load_localities = self._load_event_localities.pop(completed_event_idx, [])
            if perf is not None:
                elapsed_s = max(time.monotonic() - perf.started_at_s, 1e-9)
                self._update_estimated_swap_bandwidth(
                    perf.bytes_to_copy,
                    elapsed_s,
                )
                elapsed_ms = elapsed_s * 1000.0
                self._telemetry_swap_in_time_ms += elapsed_ms
                bytes_per_block = self._count_transfer_bytes(1)
                local_bytes, remote_bytes = self._count_local_remote_bytes(
                    load_localities,
                    bytes_per_block,
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

    def take_events(self) -> Iterable[KVCacheEvent]:
        return self.cpu_block_pool.take_events()
