# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DMA copy backend for GPU<->CPU block transfers."""

from __future__ import annotations

import queue
import threading
import time

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    BatchCopyStats,
    BatchCopyWorkspace,
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    BatchMemcpyParams,
    build_params,
    copy_blocks,
)

logger = init_logger(__name__)


class CopyBackendError(RuntimeError):
    """Raised when an asynchronous KV copy cannot be submitted or completed."""


def contiguous_block_runs(
    src_blocks: list[int], dst_blocks: list[int]
) -> list[tuple[int, int, int]]:
    """Return maximal runs with contiguous source and destination IDs."""
    if len(src_blocks) != len(dst_blocks) or not src_blocks:
        return []
    runs: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(src_blocks) + 1):
        if index < len(src_blocks):
            if (
                src_blocks[index] == src_blocks[index - 1] + 1
                and dst_blocks[index] == dst_blocks[index - 1] + 1
            ):
                continue
        runs.append((src_blocks[start], dst_blocks[start], index - start))
        start = index
    return runs


class DmaCopyBackend:
    """cuMemcpyBatchAsync copy backend with independent direction workers."""

    def __init__(self, queue_depth: int = 8) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._load_queue: queue.Queue | None = None
        self._store_queue: queue.Queue | None = None
        self._load_thread: threading.Thread | None = None
        self._store_thread: threading.Thread | None = None
        self._load_src: tuple[torch.Tensor, ...] | None = None
        self._load_dst: tuple[torch.Tensor, ...] | None = None
        self._store_src: tuple[torch.Tensor, ...] | None = None
        self._store_dst: tuple[torch.Tensor, ...] | None = None
        self._shutdown = False
        self._queue_depth = max(1, int(queue_depth))
        self._error_lock = threading.Lock()
        self._error: BaseException | None = None
        self._native_submit = None
        self._native_plan_load = None
        self._native_plan_store = None
        self._native_stats_lock = threading.Lock()
        self._native_stats_last: dict[str, int] = {}
        self._copy_stats_lock = threading.Lock()
        self._copy_stats: dict[str, int] = {}
        self._load_workspace = BatchCopyWorkspace()
        self._store_workspace = BatchCopyWorkspace()
        self._queue_stats_lock = threading.Lock()
        self._queue_peak = {"load": 0, "store": 0}
        self._enqueue_wait_ns = {"load": 0, "store": 0}
        self._enqueue_full = {"load": 0, "store": 0}

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
        queue_depth: int | None = None,
    ) -> None:
        if queue_depth is not None:
            self._queue_depth = max(1, int(queue_depth))
        self._load_stream = load_stream
        self._store_stream = store_stream
        self._load_src = tuple(cpu_caches.values())
        self._load_dst = tuple(gpu_caches.values())
        self._store_src = tuple(gpu_caches.values())
        self._store_dst = tuple(cpu_caches.values())

        # Stores read the live KV cache, so STREAM is required together with
        # the compute-done wait in the worker. Loads read stable host memory.
        self._store_params = self._try_build_params(
            gpu_caches,
            cpu_caches,
            store_stream,
            CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        )
        self._load_params = self._try_build_params(
            cpu_caches,
            gpu_caches,
            load_stream,
            CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
        )
        self._native_plan_load = self._build_native_plan(self._load_params)
        self._native_plan_store = self._build_native_plan(self._store_params)

        self._load_queue = queue.Queue(maxsize=self._queue_depth)
        self._store_queue = queue.Queue(maxsize=self._queue_depth)
        self._load_thread = threading.Thread(
            target=self._copy_loop,
            args=(
                self._load_queue,
                device,
                load_stream,
                self._load_src,
                self._load_dst,
                self._load_params,
                "load",
            ),
            daemon=True,
        )
        self._store_thread = threading.Thread(
            target=self._copy_loop,
            args=(
                self._store_queue,
                device,
                store_stream,
                self._store_src,
                self._store_dst,
                self._store_params,
                "store",
            ),
            daemon=True,
        )
        self._load_thread.start()
        self._store_thread.start()
        logger.info("DmaCopyBackend: dual-thread async copy backend enabled")

    @staticmethod
    def _try_build_params(
        src_caches: dict[str, torch.Tensor],
        dst_caches: dict[str, torch.Tensor],
        stream: torch.cuda.Stream,
        src_access_order: int,
    ) -> BatchMemcpyParams | None:
        try:
            for src, dst in zip(src_caches.values(), dst_caches.values()):
                src_payload = (src.numel() // src.shape[0]) * src.element_size()
                dst_payload = (dst.numel() // dst.shape[0]) * dst.element_size()
                if src_payload != dst_payload:
                    return None
        except AttributeError:
            # Lightweight test doubles do not expose tensor stride metadata.
            pass
        try:
            return build_params(
                src_caches,
                dst_caches,
                stream,
                src_access_order=src_access_order,
            )
        except TypeError:
            # Keep lightweight test doubles and older local helpers callable;
            # the production helper accepts src_access_order.
            return build_params(src_caches, dst_caches, stream)
        except AssertionError:
            # Block-first CPU slabs have a larger row stride than each GPU
            # segment. They use the ordered row-copy fallback below.
            return None

    def raise_if_failed(self) -> None:
        with self._error_lock:
            error = self._error
        if error is not None:
            raise CopyBackendError("asynchronous KV copy failed") from error

    def queue_depths(self) -> dict[str, int]:
        return {
            "load": self._load_queue.qsize() if self._load_queue is not None else 0,
            "store": self._store_queue.qsize()
            if self._store_queue is not None
            else 0,
        }

    def queue_telemetry(self) -> dict[str, int]:
        with self._queue_stats_lock:
            result = {
                f"{direction}_queue_peak_depth": self._queue_peak[direction]
                for direction in ("load", "store")
            }
            result.update(
                {
                    f"{direction}_enqueue_wait_ns": self._enqueue_wait_ns[direction]
                    for direction in ("load", "store")
                }
            )
            result.update(
                {
                    f"{direction}_enqueue_full": self._enqueue_full[direction]
                    for direction in ("load", "store")
                }
            )
            self._enqueue_wait_ns = {"load": 0, "store": 0}
            self._enqueue_full = {"load": 0, "store": 0}
        return result

    @staticmethod
    def _build_native_plan(params: BatchMemcpyParams | None):
        del params
        return None

    def native_telemetry(self) -> dict[str, int]:
        return {}

    def _record_copy_stats(self, direction: str, stats: BatchCopyStats) -> None:
        with self._copy_stats_lock:
            for name, value in (
                ("blocks", stats.blocks),
                ("descriptors", stats.descriptors),
                ("coalesced_spans", stats.coalesced_spans),
                ("bytes", stats.bytes),
                ("workspace_reallocations", stats.workspace_reallocations),
            ):
                key = f"{direction}_{name}"
                self._copy_stats[key] = self._copy_stats.get(key, 0) + value

    def copy_telemetry(self) -> dict[str, int]:
        with self._copy_stats_lock:
            current = dict(self._copy_stats)
            self._copy_stats.clear()
        return current

    def _set_error(self, error: BaseException, direction: str) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = error
                logger.exception("%s KV copy thread failed", direction)

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        wait_event: torch.Event | None = None,
        localities: list[str] | None = None,
    ) -> None:
        self.raise_if_failed()
        q = self._store_queue if is_store else self._load_queue
        params = self._store_params if is_store else self._load_params
        src_caches = self._store_src if is_store else self._load_src
        dst_caches = self._store_dst if is_store else self._load_dst
        assert (
            q is not None
            and src_caches is not None
            and dst_caches is not None
        )
        direction = "store" if is_store else "load"
        started_ns = time.monotonic_ns()
        try:
            q.put(
                (
                    src_blocks,
                    dst_blocks,
                    params,
                    src_caches,
                    dst_caches,
                    event_idx,
                    events_list,
                    wait_event,
                    localities,
                ),
                timeout=1.0,
            )
        except queue.Full as exc:
            with self._queue_stats_lock:
                self._enqueue_full[direction] += 1
            raise CopyBackendError(
                f"{('store' if is_store else 'load')} KV copy queue is full"
            ) from exc
        finally:
            elapsed_ns = time.monotonic_ns() - started_ns
            with self._queue_stats_lock:
                self._enqueue_wait_ns[direction] += elapsed_ns
                self._queue_peak[direction] = max(
                    self._queue_peak[direction], q.qsize()
                )

    def flush(self) -> None:
        """Wait until both queues have submitted all work enqueued so far."""
        self.raise_if_failed()
        barriers: list[threading.Event] = []
        for q in (self._load_queue, self._store_queue):
            if q is None:
                continue
            barrier = threading.Event()
            barriers.append(barrier)
            try:
                q.put((barrier,), timeout=1.0)
            except queue.Full as exc:
                raise CopyBackendError("KV copy queue did not drain") from exc
        for barrier in barriers:
            if not barrier.wait(timeout=10.0):
                self.raise_if_failed()
                raise CopyBackendError("KV copy queue barrier timed out")
        self.raise_if_failed()

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        for q, thread in (
            (self._load_queue, self._load_thread),
            (self._store_queue, self._store_thread),
        ):
            if q is not None and thread is not None and thread.is_alive():
                q.put(None, timeout=5.0)
        if self._load_thread is not None:
            self._load_thread.join(timeout=5.0)
        if self._store_thread is not None:
            self._store_thread.join(timeout=5.0)

    def _copy_loop(
        self,
        q: queue.Queue,
        device: torch.device,
        stream: torch.cuda.Stream,
        src_caches: tuple[torch.Tensor, ...],
        dst_caches: tuple[torch.Tensor, ...],
        params: BatchMemcpyParams | None,
        direction: str,
    ) -> None:
        try:
            current_platform.set_device(device)
            while True:
                item = q.get()
                if item is None:
                    return
                if len(item) == 1:
                    item[0].set()
                    continue
                (
                    src_blocks,
                    dst_blocks,
                    _params,
                    _src_caches,
                    _dst_caches,
                    event_idx,
                    events_list,
                    wait_event,
                    _localities,
                ) = item
                del _params, _src_caches, _dst_caches
                if wait_event is not None:
                    stream.wait_event(wait_event)
                native_plan = (
                    self._native_plan_store if direction == "store" else self._native_plan_load
                )
                if params is not None and self._native_submit is not None:
                    self._native_submit(native_plan, src_blocks, dst_blocks)
                elif params is not None:
                    self._record_copy_stats(
                        direction,
                        copy_blocks(
                            src_blocks,
                            dst_blocks,
                            params,
                            self._store_workspace
                            if direction == "store"
                            else self._load_workspace,
                        )
                    )
                else:
                    with torch.cuda.stream(stream):
                        for src_start, dst_start, run_length in contiguous_block_runs(
                            src_blocks, dst_blocks
                        ):
                            for src, dst in zip(src_caches, dst_caches):
                                dst[dst_start : dst_start + run_length].copy_(
                                    src[src_start : src_start + run_length],
                                    non_blocking=True,
                                )
                event = torch.Event()
                event.record(stream)
                events_list.append((event_idx, event))
        except BaseException as exc:
            self._set_error(exc, direction)


class NativeDmaCopyBackend(DmaCopyBackend):
    """DmaCopyBackend using a native CUDA batch-submission bridge.

    The Python queue, CUDA streams, wait events, and completion polling remain
    unchanged. Only the batch API call crosses into C++/CUDA, preserving the
    heterogeneous descriptor path used by DeepSeek-V4.
    """

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
        queue_depth: int | None = None,
    ) -> None:
        from vllm.v1.simple_kv_offload.native_copy import submit_native_copy

        self._native_submit = submit_native_copy
        super().init(
            gpu_caches,
            cpu_caches,
            device,
            load_stream,
            store_stream,
            queue_depth=queue_depth,
        )
        logger.info("NativeDmaCopyBackend: persistent Python queues + native DMA")

    @staticmethod
    def _build_native_plan(params: BatchMemcpyParams | None):
        if params is None:
            return None
        from vllm.v1.simple_kv_offload.native_copy import build_native_plan

        return build_native_plan(params)

    def native_telemetry(self) -> dict[str, int]:
        plans = {
            "load": self._native_plan_load,
            "store": self._native_plan_store,
        }
        current: dict[str, int] = {}
        for direction, plan in plans.items():
            if plan is None:
                continue
            for key, value in plan.stats().items():
                current[f"{direction}_{key}"] = int(value)
        with self._native_stats_lock:
            delta = {
                key: value - self._native_stats_last.get(key, 0)
                for key, value in current.items()
            }
            self._native_stats_last = current
        return delta

    def copy_telemetry(self) -> dict[str, int]:
        # Native plans report cumulative counters; expose only the interval
        # delta so worker metadata remains additive across steps.
        deltas = super().copy_telemetry()
        current: dict[str, int] = {}
        for direction, plan in (
            ("load", self._native_plan_load),
            ("store", self._native_plan_store),
        ):
            if plan is None:
                continue
            stats = plan.stats()
            for key in (
                "blocks",
                "descriptors",
                "coalesced_spans",
                "bytes",
                "workspace_reallocations",
            ):
                current[f"{direction}_{key}"] = int(stats.get(key, 0))
        with self._native_stats_lock:
            previous = getattr(self, "_native_copy_stats_last", {})
            for key, value in current.items():
                deltas[key] = value - previous.get(key, 0)
            self._native_copy_stats_last = current
        return deltas


class InlineCopyBackend:
    """Direct submission backend that preserves store ordering semantics."""

    def __init__(self, queue_depth: int = 8) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._load_src: tuple[torch.Tensor, ...] | None = None
        self._load_dst: tuple[torch.Tensor, ...] | None = None
        self._store_src: tuple[torch.Tensor, ...] | None = None
        self._store_dst: tuple[torch.Tensor, ...] | None = None
        self._queue_depth = max(1, int(queue_depth))

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
        queue_depth: int | None = None,
    ) -> None:
        del device
        if queue_depth is not None:
            self._queue_depth = max(1, int(queue_depth))
        self._load_stream = load_stream
        self._store_stream = store_stream
        self._load_src = tuple(cpu_caches.values())
        self._load_dst = tuple(gpu_caches.values())
        self._store_src = tuple(gpu_caches.values())
        self._store_dst = tuple(cpu_caches.values())
        self._store_params = DmaCopyBackend._try_build_params(
            gpu_caches,
            cpu_caches,
            store_stream,
            CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        )
        self._load_params = DmaCopyBackend._try_build_params(
            cpu_caches,
            gpu_caches,
            load_stream,
            CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
        )

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        wait_event: torch.Event | None = None,
        localities: list[str] | None = None,
    ) -> None:
        self.raise_if_failed()
        del localities
        params = self._store_params if is_store else self._load_params
        stream = self._store_stream if is_store else self._load_stream
        src_caches = self._store_src if is_store else self._load_src
        dst_caches = self._store_dst if is_store else self._load_dst
        assert (
            stream is not None
            and src_caches is not None
            and dst_caches is not None
        )
        if wait_event is not None:
            stream.wait_event(wait_event)
        if params is not None:
            copy_blocks(src_blocks, dst_blocks, params)
        else:
            with torch.cuda.stream(stream):
                for src_start, dst_start, run_length in contiguous_block_runs(
                    src_blocks, dst_blocks
                ):
                    for src, dst in zip(src_caches, dst_caches):
                        dst[dst_start : dst_start + run_length].copy_(
                            src[src_start : src_start + run_length],
                            non_blocking=True,
                        )
        event = torch.Event()
        event.record(stream)
        events_list.append((event_idx, event))

    def flush(self) -> None:
        return

    def shutdown(self) -> None:
        return

    def raise_if_failed(self) -> None:
        return

    def queue_depths(self) -> dict[str, int]:
        return {"load": 0, "store": 0}
