# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for SimpleCPUOffloadConnector."""

from collections import deque
import os
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.simple_kv_offload.copy_backend import (
    DmaCopyBackend,
    InlineCopyBackend,
    NativeDmaCopyBackend,
)
from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor
from vllm.v1.simple_kv_offload.layout import (
    LayoutModeDecision,
    OffloadLayoutDescriptor,
    build_gpu_cache_views,
    choose_layout_mode,
)
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class SimpleCPUOffloadWorker:
    """Worker-side handler for CPU offloading transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        pin_memory_fix: bool | None = None,
        swapper_block_first: bool | None = None,
        transfer_queue_depth: int = 8,
        cpu_kv_allocation_mode: str = "zero",
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.cpu_capacity_bytes = cpu_capacity_bytes
        if cpu_kv_allocation_mode not in {"zero", "empty"}:
            raise ValueError(
                "cpu_kv_allocation_mode must be 'zero' or 'empty'"
            )
        self.cpu_kv_allocation_mode = cpu_kv_allocation_mode
        self.cpu_kv_allocated_bytes = 0
        self.cpu_kv_pinned_bytes = 0
        self.cpu_kv_allocation_time_ms = 0.0
        cache_config = getattr(vllm_config, "cache_config", None)
        self._pin_memory_fix = (
            bool(getattr(cache_config, "pin_memory_fix", False))
            if pin_memory_fix is None
            else pin_memory_fix
        )
        self._swapper_block_first = (
            bool(getattr(cache_config, "swapper_block_first", False))
            if swapper_block_first is None
            else swapper_block_first
        )
        self._superinfer_high_risk_mode = bool(
            getattr(cache_config, "superinfer_high_risk_mode", False)
        )
        extra_config = getattr(
            getattr(vllm_config, "kv_transfer_config", None),
            "kv_connector_extra_config",
            {},
        ) or {}
        self._native_copy_backend = bool(
            getattr(cache_config, "native_copy_backend", False)
            or extra_config.get("native_copy_backend", False)
            or os.environ.get("NATIVE_COPY_BACKEND") == "1"
        )
        self._reported_static_metadata_mismatch = False

        self.gpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.cpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.layout_descriptor: OffloadLayoutDescriptor | None = None
        self.device: torch.device | None = None
        self.num_cpu_blocks: int = 0
        self._block_first_slab: torch.Tensor | None = None

        # CUDA streams for the async transfers
        self.load_stream: torch.cuda.Stream | None = None
        self.store_stream: torch.cuda.Stream | None = None

        if self._superinfer_high_risk_mode:
            self._backend = InlineCopyBackend()
        elif self._native_copy_backend:
            self._backend = NativeDmaCopyBackend(queue_depth=transfer_queue_depth)
        else:
            self._backend = DmaCopyBackend(queue_depth=transfer_queue_depth)
        logger.info(
            "SimpleCPUOffloadWorker: copy_backend=%s native_requested=%s",
            type(self._backend).__name__,
            self._native_copy_backend,
        )

        # Ordered (event_idx, Event). Events pre-allocated on main thread.
        self._load_events: deque[tuple[int, torch.Event]] = deque()
        self._store_events: deque[tuple[int, torch.Event]] = deque()
        # High-water marks: highest event_idx completed per stream.
        # When the event list is empty, the hwm covers all prior events.
        self._load_hwm: int = -1
        self._store_hwm: int = -1

        # Metadata for the current step
        self._connector_metadata: SimpleCPUOffloadMetadata | None = None

        # Pending event index sets, populated in bind_connector_metadata
        self._pending_load_event_indices: set[int] = set()
        self._pending_store_event_indices: set[int] = set()
        self._submitted_load_event_indices: set[int] = set()
        # Completed store events to report via build_connector_worker_meta
        self._completed_store_events: dict[int, int] = {}

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Register GPU KV caches and allocate pinned CPU tensors.
        The worker will infer the underlying raw storage from the kv_caches.

        Args:
            kv_caches: Per-layer GPU KV caches. Values are either a single
                tensor (attention layers) or a list of tensors (Mamba layers
                in hybrid models). All values are included for offloading
                by resolving to their underlying raw storage.
        """
        if not kv_caches:
            logger.warning("No KV caches to offload.")
            return

        # Resolve each entry to a representative tensor for storage
        # deduplication. For attention layers the value is already a tensor;
        # for Mamba layers it is a list of tensors that all share the same
        # underlying raw storage, so we take the first one.
        def _repr_tensor(v: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
            assert isinstance(v, torch.Tensor | list)
            return v if isinstance(v, torch.Tensor) else v[0]

        any_tensor = _repr_tensor(next(iter(kv_caches.values())))
        self.device = any_tensor.device

        assert self.kv_cache_config is not None
        num_blocks = self.kv_cache_config.num_blocks

        has_non_tensor_values = any(
            not isinstance(value, torch.Tensor) for value in kv_caches.values()
        )
        model_arches = tuple(
            getattr(self.vllm_config.model_config, "architectures", ()) or ()
        )
        num_kv_cache_groups = (
            len(self.kv_cache_config.kv_cache_groups)
            if self.kv_cache_config is not None
            else 0
        )
        layout_decision: LayoutModeDecision = choose_layout_mode(
            self._swapper_block_first,
            model_arches=model_arches,
            num_kv_cache_groups=num_kv_cache_groups,
            has_non_tensor_values=has_non_tensor_values,
            tensor_parallel_size=self.vllm_config.parallel_config.tensor_parallel_size,
            aggressive_mode=self._superinfer_high_risk_mode,
        )
        if self._swapper_block_first and layout_decision.mode != "block_first":
            logger.warning(
                "swapper_block_first requested but using gpu_derived layout (%s)",
                layout_decision.reason,
            )

        repr_kv_caches = {
            name: _repr_tensor(value) for name, value in kv_caches.items()
        }
        unique_gpu_caches, self.layout_descriptor = build_gpu_cache_views(
            repr_kv_caches,
            num_blocks,
            self.device,
            mode=layout_decision.mode,
        )

        # Compute per-tensor bytes_per_block. Tensors may have different
        # page_size_bytes (e.g., UniformTypeKVCacheSpecs with varying head_size).
        per_tensor_bpb = [
            t.stride(0) * t.element_size() for t in unique_gpu_caches.values()
        ]
        total_bytes_per_block = sum(per_tensor_bpb)

        self.num_cpu_blocks = max(1, self.cpu_capacity_bytes // total_bytes_per_block)

        logger.info(
            "SimpleCPUOffloadWorker: %d unique GPU KV tensors, "
            "allocating %d CPU blocks (%.2f GB)",
            len(unique_gpu_caches),
            self.num_cpu_blocks,
            (self.num_cpu_blocks * total_bytes_per_block) / (1024**3),
        )

        pin_memory = is_pin_memory_available()
        if not pin_memory:
            logger.warning(
                "Pinned memory not available. CPU offload performance may be degraded."
            )

        self.gpu_kv_caches = unique_gpu_caches
        self.cpu_kv_caches = {}

        def pin_if_available(tensor: torch.Tensor) -> None:
            if not pin_memory:
                return
            try:
                pin_tensor(tensor)
            except RuntimeError as exc:
                logger.warning(
                    "Unable to pin CPU KV cache memory; continuing unpinned: %s",
                    exc,
                )

        import time

        allocation_started_at = time.monotonic()
        if self.layout_descriptor.mode == "block_first":
            ordered_items = list(unique_gpu_caches.items())
            block_span = sum(
                tensor.stride(0) * tensor.element_size()
                for _, tensor in ordered_items
            )
            allocate = torch.empty if self.cpu_kv_allocation_mode == "empty" else torch.zeros
            slab = allocate(
                (self.num_cpu_blocks, block_span), dtype=torch.int8, device="cpu"
            )
            pin_if_available(slab)
            self.cpu_kv_allocated_bytes += slab.nbytes
            if slab.is_pinned():
                self.cpu_kv_pinned_bytes += slab.nbytes
            self._block_first_slab = slab
            cursor = 0
            for name, gpu_tensor in ordered_items:
                segment_bytes = gpu_tensor.stride(0) * gpu_tensor.element_size()
                self.cpu_kv_caches[name] = slab[:, cursor : cursor + segment_bytes]
                cursor += segment_bytes
            assert cursor == block_span
        else:
            self._block_first_slab = None
            for name, gpu_tensor in unique_gpu_caches.items():
                cpu_shape = (self.num_cpu_blocks,) + gpu_tensor.shape[1:]
                allocate = (
                    torch.empty
                    if self.cpu_kv_allocation_mode == "empty"
                    else torch.zeros
                )
                tensor = allocate(cpu_shape, dtype=gpu_tensor.dtype, device="cpu")
                pin_if_available(tensor)
                self.cpu_kv_allocated_bytes += tensor.nbytes
                if tensor.is_pinned():
                    self.cpu_kv_pinned_bytes += tensor.nbytes
                self.cpu_kv_caches[name] = tensor

        self.cpu_kv_allocation_time_ms = (
            time.monotonic() - allocation_started_at
        ) * 1000
        logger.info(
            "SimpleCPUOffloadWorker: CPU KV allocation mode=%s allocated=%.2f GB "
            "pinned=%.2f GB allocation_time_ms=%.1f",
            self.cpu_kv_allocation_mode,
            self.cpu_kv_allocated_bytes / (1024**3),
            self.cpu_kv_pinned_bytes / (1024**3),
            self.cpu_kv_allocation_time_ms,
        )

        # Use lowest priority so KV cache I/O yields to compute streams.
        low_pri, _ = torch.cuda.Stream.priority_range()
        self.load_stream = torch.cuda.Stream(priority=low_pri)
        self.store_stream = torch.cuda.Stream(priority=low_pri)

        # Initialize copy backend with caches and streams.
        self._backend.init(
            self.gpu_kv_caches,
            self.cpu_kv_caches,
            self.device,
            self.load_stream,
            self.store_stream,
            queue_depth=getattr(self._backend, "_queue_depth", None),
        )

    def bind_connector_metadata(self, metadata: SimpleCPUOffloadMetadata) -> None:
        self._connector_metadata = metadata
        if (
            not self._reported_static_metadata_mismatch
            and (
                metadata.pin_memory_fix != self._pin_memory_fix
                or metadata.swapper_block_first != self._swapper_block_first
            )
        ):
            logger.warning(
                "SimpleCPUOffloadWorker metadata/config mismatch: "
                "metadata(pin_memory_fix=%s, swapper_block_first=%s) != "
                "worker(pin_memory_fix=%s, swapper_block_first=%s)",
                metadata.pin_memory_fix,
                metadata.swapper_block_first,
                self._pin_memory_fix,
                self._swapper_block_first,
            )
            self._reported_static_metadata_mismatch = True
        if metadata.load_event >= 0:
            self._pending_load_event_indices.add(metadata.load_event)
        if metadata.store_event >= 0:
            self._pending_store_event_indices.add(metadata.store_event)

    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None

    def start_load_kv(self) -> None:
        """Submit H2D loads before model execution begins.

        Loads read stable pinned CPU memory and can overlap with the forward
        pass. Stores remain deferred to get_finished() because they must wait
        for the current compute stream to finish writing KV blocks.
        """
        raise_if_failed = getattr(self._backend, "raise_if_failed", None)
        if callable(raise_if_failed):
            raise_if_failed()

        metadata = self._connector_metadata
        if metadata is None or not metadata.load_cpu_blocks:
            return
        event_idx = metadata.load_event
        if event_idx in self._submitted_load_event_indices:
            return
        self._backend.launch_copy(
            metadata.load_cpu_blocks,
            metadata.load_gpu_blocks,
            is_store=False,
            event_idx=event_idx,
            events_list=self._load_events,
        )
        self._submitted_load_event_indices.add(event_idx)

    def wait_for_save(self) -> None:
        pass

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """Submit transfers and report completed events to the scheduler.

        Stores (GPU->CPU) read the live KV cache, which the compute stream may
        still be writing under v1 overlapped execution, so they are ordered
        after a compute-done event recorded on the current stream. Loads
        (CPU->GPU) read stable pinned host memory and launch immediately. See
        #45704 for the bug and #39306 for the srcAccessOrder rationale.

        Returns:
            tuple of (finished_sending, finished_recving).
            - finished_sending: always None (stores use worker metadata).
            - finished_recving: req_ids whose loads have completed.
        """
        raise_if_failed = getattr(self._backend, "raise_if_failed", None)
        if callable(raise_if_failed):
            raise_if_failed()

        # (1) Submit stores. Loads were submitted in start_load_kv() before
        # model execution so their H2D work can overlap with the forward pass.
        metadata = self._connector_metadata
        if metadata is not None:
            if metadata.store_gpu_blocks:
                # Each queued store owns its event. The copy thread may not
                # consume a wait event before the next scheduler step records
                # another event, so reusing one event can order a store behind
                # the wrong compute step.
                store_compute_done = torch.Event()
                store_compute_done.record(torch.cuda.current_stream())
                self._backend.launch_copy(
                    metadata.store_gpu_blocks,
                    metadata.store_cpu_blocks,
                    is_store=True,
                    event_idx=metadata.store_event,
                    events_list=self._store_events,
                    wait_event=store_compute_done,
                )

        # (2) Track completed transfer events
        finished_recving: set[str] = set()

        if self._pending_load_event_indices:
            load_wm = self._poll_stream_events(is_store=False)
            for j in [j for j in self._pending_load_event_indices if j <= load_wm]:
                self._pending_load_event_indices.discard(j)
                req_ids = (
                    metadata.load_event_to_reqs.get(j) if metadata is not None else None
                )
                if req_ids:
                    finished_recving.update(req_ids)
                self._submitted_load_event_indices.discard(j)

        if self._pending_store_event_indices:
            store_wm = self._poll_stream_events(is_store=True)
            for j in [j for j in self._pending_store_event_indices if j <= store_wm]:
                self._pending_store_event_indices.discard(j)
                self._completed_store_events[j] = 1

        if callable(raise_if_failed):
            raise_if_failed()

        return None, finished_recving or None

    def build_connector_worker_meta(self) -> SimpleCPUOffloadWorkerMetadata | None:
        """Return completed stores, queue depths, and native copy telemetry."""
        queue_depths = getattr(self._backend, "queue_depths", lambda: {})()
        queue_stats = getattr(self._backend, "queue_telemetry", lambda: {})()
        load_queue_depth = int(queue_depths.get("load", 0))
        store_queue_depth = int(queue_depths.get("store", 0))
        native = getattr(self._backend, "native_telemetry", lambda: {})()
        copy_stats = getattr(self._backend, "copy_telemetry", lambda: {})()
        if (
            not self._completed_store_events
            and load_queue_depth == 0
            and store_queue_depth == 0
            and not native
            and not copy_stats
            and not queue_stats
        ):
            return None
        meta = SimpleCPUOffloadWorkerMetadata(
            completed_store_events=self._completed_store_events,
            load_queue_depth=load_queue_depth,
            store_queue_depth=store_queue_depth,
            cpu_kv_capacity_bytes=self.cpu_capacity_bytes,
            cpu_kv_allocated_bytes=self.cpu_kv_allocated_bytes,
            cpu_kv_pinned_bytes=self.cpu_kv_pinned_bytes,
            cpu_kv_allocation_time_ms=int(self.cpu_kv_allocation_time_ms),
            cpu_kv_allocation_mode_empty=int(
                self.cpu_kv_allocation_mode == "empty"
            ),
            load_queue_peak_depth=queue_stats.get("load_queue_peak_depth", 0),
            store_queue_peak_depth=queue_stats.get("store_queue_peak_depth", 0),
            load_enqueue_wait_ns=queue_stats.get("load_enqueue_wait_ns", 0),
            store_enqueue_wait_ns=queue_stats.get("store_enqueue_wait_ns", 0),
            load_enqueue_full=queue_stats.get("load_enqueue_full", 0),
            store_enqueue_full=queue_stats.get("store_enqueue_full", 0),
            **{
                f"{direction}_{key}_submitted": copy_stats.get(
                    f"{direction}_{key}", 0
                )
                for direction in ("load", "store")
                for key in (
                    "blocks",
                    "descriptors",
                    "bytes",
                )
            },
            load_coalesced_spans=copy_stats.get("load_coalesced_spans", 0),
            store_coalesced_spans=copy_stats.get("store_coalesced_spans", 0),
            load_workspace_reallocations=copy_stats.get(
                "load_workspace_reallocations", 0
            ),
            store_workspace_reallocations=copy_stats.get(
                "store_workspace_reallocations", 0
            ),
            **{
                f"native_{direction}_{key}": value
                for direction in ("load", "store")
                for key, value in (
                    ("submissions", native.get(f"{direction}_submissions", 0)),
                    ("blocks", native.get(f"{direction}_blocks", 0)),
                    ("descriptors", native.get(f"{direction}_descriptors", 0)),
                    ("bytes", native.get(f"{direction}_bytes", 0)),
                    ("submit_ns", native.get(f"{direction}_submit_ns", 0)),
                    ("errors", native.get(f"{direction}_errors", 0)),
                    (
                        "workspace_reallocations",
                        native.get(f"{direction}_workspace_reallocations", 0),
                    ),
                )
            },
        )
        self._completed_store_events = {}
        return meta

    def handle_preemptions(
        self, kv_connector_metadata: SimpleCPUOffloadMetadata
    ) -> None:
        """Sync all in-flight transfers before preempted blocks are reused."""
        if not kv_connector_metadata.need_flush:
            return
        self._flush_and_sync_all()

    def _flush_and_sync_all(self) -> None:
        """Synchronize all in-flight transfer events."""
        flush = getattr(self._backend, "flush", None)
        if flush is not None:
            flush()
        for event_idx, event in self._load_events:
            event.synchronize()
            self._load_hwm = event_idx
        self._load_events.clear()

        for event_idx, event in self._store_events:
            event.synchronize()
            self._store_hwm = event_idx
        self._store_events.clear()

    def _poll_stream_events(self, is_store: bool) -> int:
        """Non-blocking poll for completed events and return the high-water mark."""
        events = self._store_events if is_store else self._load_events
        hwm = self._store_hwm if is_store else self._load_hwm
        while events:
            event_idx, event = events[0]
            if not event.query():
                break
            hwm = event_idx
            events.popleft()
        if is_store:
            self._store_hwm = hwm
        else:
            self._load_hwm = hwm
        return hwm
