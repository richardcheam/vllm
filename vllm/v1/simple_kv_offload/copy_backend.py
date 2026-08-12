# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DMA copy backend for GPU<->CPU block transfers."""

from __future__ import annotations

import queue
import threading

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    BatchMemcpyParams,
    build_params,
    copy_blocks,
)

logger = init_logger(__name__)


class DmaCopyBackend:
    """cuMemcpyBatchAsync copy backend (background thread)."""

    def __init__(self) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._load_queue: queue.SimpleQueue | None = None
        self._store_queue: queue.SimpleQueue | None = None
        self._load_thread: threading.Thread | None = None
        self._store_thread: threading.Thread | None = None
        self._shutdown: bool = False

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        self._load_stream = load_stream
        self._store_stream = store_stream

        self._store_params = build_params(gpu_caches, cpu_caches, store_stream)
        self._load_params = build_params(cpu_caches, gpu_caches, load_stream)

        self._load_queue = queue.SimpleQueue()
        self._store_queue = queue.SimpleQueue()
        self._load_thread = threading.Thread(
            target=self._copy_loop,
            args=(self._load_queue, device, load_stream),
            daemon=True,
        )
        self._store_thread = threading.Thread(
            target=self._copy_loop,
            args=(self._store_queue, device, store_stream),
            daemon=True,
        )
        self._load_thread.start()
        self._store_thread.start()
        logger.info("DmaCopyBackend: dual-thread batched async copy backend enabled")

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        localities: list[str] | None = None,
    ) -> None:
        params = self._store_params if is_store else self._load_params
        q = self._store_queue if is_store else self._load_queue
        assert params is not None and q is not None
        q.put((src_blocks, dst_blocks, params, event_idx, events_list, localities or []))

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._load_queue is not None:
            self._load_queue.put(None)
        if self._store_queue is not None:
            self._store_queue.put(None)
        if self._load_thread is not None:
            self._load_thread.join(timeout=5.0)
        if self._store_thread is not None:
            self._store_thread.join(timeout=5.0)

    @staticmethod
    def _copy_loop(
        q: queue.SimpleQueue,
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> None:
        current_platform.set_device(device)
        while True:
            item = q.get()
            if item is None:
                return
            src_blocks, dst_blocks, params, event_idx, events_list, localities = item
            _ = localities
            copy_blocks(src_blocks, dst_blocks, params)
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))


class InlineCopyBackend:
    """Direct submission backend without background worker threads."""

    def __init__(self) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        del device
        self._load_stream = load_stream
        self._store_stream = store_stream
        self._store_params = build_params(gpu_caches, cpu_caches, store_stream)
        self._load_params = build_params(cpu_caches, gpu_caches, load_stream)
        logger.info("InlineCopyBackend: direct transfer submission enabled")

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        localities: list[str] | None = None,
    ) -> None:
        _ = localities
        params = self._store_params if is_store else self._load_params
        stream = self._store_stream if is_store else self._load_stream
        assert params is not None and stream is not None
        copy_blocks(src_blocks, dst_blocks, params)
        event = torch.Event()
        event.record(stream)
        events_list.append((event_idx, event))

    def shutdown(self) -> None:
        return
