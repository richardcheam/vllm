# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata for SimpleCPUOffloadConnector."""

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

INVALID_JOB_ID = -1


@dataclass
class SimpleCPUOffloadMetadata(KVConnectorMetadata):
    """
    Metadata passed from scheduler to worker for CPU offload operations.

    The worker receives flat block lists keyed by a monotonic event_idx.
    Job->req_id translation is handled by the scheduler-side manager
    (via inverse maps), so the worker never knows about request identities.
    """

    # Load event per step. INVALID_JOB_ID means no blocks to load this step.
    load_event: int = INVALID_JOB_ID
    load_gpu_blocks: list[int] = field(default_factory=list)
    load_cpu_blocks: list[int] = field(default_factory=list)
    # Locality labels are scheduler telemetry hints. The v0.26 worker keeps
    # using its existing DMA backend, which intentionally ignores placement.
    load_cpu_block_localities: dict[int, list[str]] = field(default_factory=dict)
    # Reverse map: load_event->req_ids, for tracking requests with finished load events
    load_event_to_reqs: dict[int, list[str]] = field(default_factory=dict)

    # Store event per step. INVALID_JOB_ID means no blocks to store this step.
    store_event: int = INVALID_JOB_ID
    store_gpu_blocks: list[int] = field(default_factory=list)
    store_cpu_blocks: list[int] = field(default_factory=list)
    store_cpu_block_localities: list[str] = field(default_factory=list)

    # Whether any requests were preempted this step and need flush pending transfers.
    need_flush: bool = False

    # Static compatibility knobs carried for worker-side validation by newer
    # workers; harmless for the v0.26 worker.
    pin_memory_fix: bool = False
    swapper_block_first: bool = False


@dataclass
class SimpleCPUOffloadWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> Scheduler metadata for completed store events.

    Each worker reports {event_idx: 1} for newly completed stores.
    ``aggregate()`` sums counts across workers within a step.
    The scheduler-side manager accumulates across steps and processes
    a store completion only when count reaches ``world_size``.
    """

    completed_store_events: dict[int, int]
    cpu_kv_capacity_bytes: int = 0
    cpu_kv_allocated_bytes: int = 0
    cpu_kv_pinned_bytes: int = 0
    cpu_kv_allocation_time_ms: int = 0
    cpu_kv_allocation_mode_empty: int = 0
    load_queue_depth: int = 0
    store_queue_depth: int = 0
    load_queue_peak_depth: int = 0
    store_queue_peak_depth: int = 0
    load_enqueue_wait_ns: int = 0
    store_enqueue_wait_ns: int = 0
    load_enqueue_full: int = 0
    store_enqueue_full: int = 0
    load_blocks_submitted: int = 0
    store_blocks_submitted: int = 0
    load_descriptors_submitted: int = 0
    store_descriptors_submitted: int = 0
    load_bytes_submitted: int = 0
    store_bytes_submitted: int = 0
    load_coalesced_spans: int = 0
    store_coalesced_spans: int = 0
    load_workspace_reallocations: int = 0
    store_workspace_reallocations: int = 0
    native_load_submissions: int = 0
    native_store_submissions: int = 0
    native_load_blocks: int = 0
    native_store_blocks: int = 0
    native_load_descriptors: int = 0
    native_store_descriptors: int = 0
    native_load_bytes: int = 0
    native_store_bytes: int = 0
    native_load_submit_ns: int = 0
    native_store_submit_ns: int = 0
    native_load_errors: int = 0
    native_store_errors: int = 0
    native_load_workspace_reallocations: int = 0
    native_store_workspace_reallocations: int = 0

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, SimpleCPUOffloadWorkerMetadata)
        merged = dict(self.completed_store_events)
        for k, v in other.completed_store_events.items():
            merged[k] = merged.get(k, 0) + v
        return SimpleCPUOffloadWorkerMetadata(
            completed_store_events=merged,
            cpu_kv_capacity_bytes=max(
                self.cpu_kv_capacity_bytes, other.cpu_kv_capacity_bytes
            ),
            cpu_kv_allocated_bytes=max(
                self.cpu_kv_allocated_bytes, other.cpu_kv_allocated_bytes
            ),
            cpu_kv_pinned_bytes=max(
                self.cpu_kv_pinned_bytes, other.cpu_kv_pinned_bytes
            ),
            cpu_kv_allocation_time_ms=max(
                self.cpu_kv_allocation_time_ms, other.cpu_kv_allocation_time_ms
            ),
            cpu_kv_allocation_mode_empty=max(
                self.cpu_kv_allocation_mode_empty,
                other.cpu_kv_allocation_mode_empty,
            ),
            load_queue_depth=max(self.load_queue_depth, other.load_queue_depth),
            store_queue_depth=max(self.store_queue_depth, other.store_queue_depth),
            load_queue_peak_depth=max(
                self.load_queue_peak_depth, other.load_queue_peak_depth
            ),
            store_queue_peak_depth=max(
                self.store_queue_peak_depth, other.store_queue_peak_depth
            ),
            load_enqueue_wait_ns=self.load_enqueue_wait_ns + other.load_enqueue_wait_ns,
            store_enqueue_wait_ns=self.store_enqueue_wait_ns + other.store_enqueue_wait_ns,
            load_enqueue_full=self.load_enqueue_full + other.load_enqueue_full,
            store_enqueue_full=self.store_enqueue_full + other.store_enqueue_full,
            load_blocks_submitted=(
                self.load_blocks_submitted + other.load_blocks_submitted
            ),
            store_blocks_submitted=(
                self.store_blocks_submitted + other.store_blocks_submitted
            ),
            load_descriptors_submitted=(
                self.load_descriptors_submitted + other.load_descriptors_submitted
            ),
            store_descriptors_submitted=(
                self.store_descriptors_submitted + other.store_descriptors_submitted
            ),
            load_bytes_submitted=self.load_bytes_submitted + other.load_bytes_submitted,
            store_bytes_submitted=self.store_bytes_submitted + other.store_bytes_submitted,
            load_coalesced_spans=self.load_coalesced_spans + other.load_coalesced_spans,
            store_coalesced_spans=self.store_coalesced_spans + other.store_coalesced_spans,
            load_workspace_reallocations=(
                self.load_workspace_reallocations
                + other.load_workspace_reallocations
            ),
            store_workspace_reallocations=(
                self.store_workspace_reallocations
                + other.store_workspace_reallocations
            ),
            native_load_submissions=(
                self.native_load_submissions + other.native_load_submissions
            ),
            native_store_submissions=(
                self.native_store_submissions + other.native_store_submissions
            ),
            native_load_blocks=self.native_load_blocks + other.native_load_blocks,
            native_store_blocks=self.native_store_blocks + other.native_store_blocks,
            native_load_descriptors=(
                self.native_load_descriptors + other.native_load_descriptors
            ),
            native_store_descriptors=(
                self.native_store_descriptors + other.native_store_descriptors
            ),
            native_load_bytes=self.native_load_bytes + other.native_load_bytes,
            native_store_bytes=self.native_store_bytes + other.native_store_bytes,
            native_load_submit_ns=(
                self.native_load_submit_ns + other.native_load_submit_ns
            ),
            native_store_submit_ns=(
                self.native_store_submit_ns + other.native_store_submit_ns
            ),
            native_load_errors=self.native_load_errors + other.native_load_errors,
            native_store_errors=self.native_store_errors + other.native_store_errors,
            native_load_workspace_reallocations=(
                self.native_load_workspace_reallocations
                + other.native_load_workspace_reallocations
            ),
            native_store_workspace_reallocations=(
                self.native_store_workspace_reallocations
                + other.native_store_workspace_reallocations
            ),
        )
