# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for copy backend admission and failure contracts."""

import queue
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.v1.simple_kv_offload.cuda_mem_ops import build_params, copy_blocks
from vllm.v1.simple_kv_offload.copy_backend import (
    CopyBackendError,
    DmaCopyBackend,
    NativeDmaCopyBackend,
    contiguous_block_runs,
)
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker


def test_contiguous_block_runs_coalesces_only_matching_runs():
    assert contiguous_block_runs([4, 5, 6, 9], [10, 11, 12, 20]) == [
        (4, 10, 3),
        (9, 20, 1),
    ]
    assert contiguous_block_runs([4, 6], [10, 11]) == [(4, 10, 1), (6, 11, 1)]


def test_batch_params_allow_different_source_and_destination_strides():
    src = {"segment": torch.zeros((4, 8), dtype=torch.int8)}
    dst = {"segment": torch.zeros((4, 16), dtype=torch.int8)[:, :8]}
    stream = SimpleNamespace(cuda_stream=1)
    with patch(
        "vllm.v1.simple_kv_offload.cuda_mem_ops._batch_memcpy_fn",
        new=lambda *args: 0,
    ):
        params = build_params(src, dst, stream)

    assert params.src_bpb[0] == 8
    assert params.dst_bpb[0] == 16
    assert params.copy_bpb[0] == 8


def test_batch_copy_coalesces_contiguous_payload_spans():
    src = {"segment": torch.zeros((8, 8), dtype=torch.int8)}
    dst = {"segment": torch.zeros((8, 8), dtype=torch.int8)}
    stream = SimpleNamespace(cuda_stream=1)
    with patch(
        "vllm.v1.simple_kv_offload.cuda_mem_ops._batch_memcpy_fn",
        new=lambda *args: 0,
    ):
        params = build_params(src, dst, stream)
        stats = copy_blocks([0, 1, 2], [4, 5, 6], params)

    assert stats.blocks == 3
    assert stats.descriptors == 1
    assert stats.coalesced_spans == 1
    assert stats.bytes == 3 * 8


def test_batch_copy_keeps_gapped_spans_separate():
    src = {"segment": torch.zeros((8, 8), dtype=torch.int8)}
    dst = {"segment": torch.zeros((8, 8), dtype=torch.int8)}
    stream = SimpleNamespace(cuda_stream=1)
    with patch(
        "vllm.v1.simple_kv_offload.cuda_mem_ops._batch_memcpy_fn",
        new=lambda *args: 0,
    ):
        params = build_params(src, dst, stream)
        stats = copy_blocks([0, 2], [4, 5], params)

    assert stats.blocks == 2
    assert stats.descriptors == 2
    assert stats.coalesced_spans == 0
    assert stats.bytes == 2 * 8


def test_dma_backend_rejects_full_direction_queue():
    backend = DmaCopyBackend(queue_depth=1)
    backend._store_queue = queue.Queue(maxsize=1)
    backend._store_queue.put((None,))
    backend._store_src = (object(),)
    backend._store_dst = (object(),)

    with pytest.raises(CopyBackendError, match="store KV copy queue is full"):
        backend.launch_copy([0], [0], True, 1, [])


def test_dma_backend_reports_direction_queue_depth():
    backend = DmaCopyBackend(queue_depth=2)
    backend._load_queue = queue.Queue(maxsize=2)
    backend._store_queue = queue.Queue(maxsize=2)
    backend._load_queue.put((None,))

    assert backend.queue_depths() == {"load": 1, "store": 0}


def test_native_backend_preserves_dma_interface():
    """The optional native backend keeps the existing queue contract."""
    backend = NativeDmaCopyBackend(queue_depth=2)
    assert isinstance(backend, DmaCopyBackend)
    assert backend._queue_depth == 2


def test_worker_builds_copy_telemetry_metadata_once():
    """Worker metadata accepts generic and native telemetry without duplicates."""

    class Backend:
        def queue_depths(self):
            return {"load": 1, "store": 2}

        def queue_telemetry(self):
            return {
                "load_queue_peak_depth": 3,
                "store_queue_peak_depth": 4,
                "load_enqueue_wait_ns": 5,
                "store_enqueue_wait_ns": 6,
                "load_enqueue_full": 0,
                "store_enqueue_full": 1,
            }

        def native_telemetry(self):
            return {
                "load_submissions": 7,
                "store_submissions": 8,
                "load_blocks": 9,
                "store_blocks": 10,
                "load_descriptors": 11,
                "store_descriptors": 12,
                "load_bytes": 13,
                "store_bytes": 14,
                "load_submit_ns": 15,
                "store_submit_ns": 16,
                "load_errors": 0,
                "store_errors": 0,
                "load_workspace_reallocations": 1,
                "store_workspace_reallocations": 2,
            }

        def copy_telemetry(self):
            return {
                "load_blocks": 17,
                "store_blocks": 18,
                "load_descriptors": 19,
                "store_descriptors": 20,
                "load_bytes": 21,
                "store_bytes": 22,
                "load_coalesced_spans": 23,
                "store_coalesced_spans": 24,
                "load_workspace_reallocations": 3,
                "store_workspace_reallocations": 4,
            }

    worker = SimpleCPUOffloadWorker(
        vllm_config=None, kv_cache_config=None, cpu_capacity_bytes=0
    )
    worker._backend = Backend()
    worker._completed_store_events = {3: 1}

    metadata = worker.build_connector_worker_meta()

    assert metadata is not None
    assert metadata.completed_store_events == {3: 1}
    assert metadata.load_blocks_submitted == 17
    assert metadata.store_descriptors_submitted == 20
    assert metadata.load_coalesced_spans == 23
    assert metadata.store_coalesced_spans == 24
    assert metadata.native_load_submissions == 7
    assert metadata.native_store_workspace_reallocations == 2


def test_cpu_kv_allocation_mode_is_explicit_and_validated():
    zero = SimpleCPUOffloadWorker(
        vllm_config=None,
        kv_cache_config=None,
        cpu_capacity_bytes=1024,
        cpu_kv_allocation_mode="zero",
    )
    empty = SimpleCPUOffloadWorker(
        vllm_config=None,
        kv_cache_config=None,
        cpu_capacity_bytes=1024,
        cpu_kv_allocation_mode="empty",
    )

    assert zero.cpu_kv_allocation_mode == "zero"
    assert empty.cpu_kv_allocation_mode == "empty"
    with pytest.raises(ValueError, match="cpu_kv_allocation_mode"):
        SimpleCPUOffloadWorker(
            vllm_config=None,
            kv_cache_config=None,
            cpu_capacity_bytes=1024,
            cpu_kv_allocation_mode="lazy",
        )
