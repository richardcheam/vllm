# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-level CUDA/HIP memory helpers: pinning and batch DMA transfers."""

import ctypes
from typing import Any, NamedTuple

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# CUmemcpySrcAccessOrder values (CUDA driver API). STREAM(1): source read in
# stream order, safe when the source may still be written. ANY(3): source may
# be read early, only safe for a stable source (e.g. pinned host memory).
CU_MEMCPY_SRC_ACCESS_ORDER_STREAM = 1
CU_MEMCPY_SRC_ACCESS_ORDER_ANY = 3


def pin_tensor(tensor: torch.Tensor) -> None:
    """Pin a CPU tensor via cudaHostRegister.

    This bypasses PyTorch's CUDACachingHostAllocator which rounds
    every ``pin_memory=True`` allocation up to the next power of 2
    (e.g. 100 GB becomes 128 GB).
    """
    err = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), tensor.nbytes, 0)
    if err.value != 0:
        raise RuntimeError(f"cudaHostRegister failed: {err}")


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint), ("id", ctypes.c_int)]


class _CUmemcpyAttributes(ctypes.Structure):
    _fields_ = [
        ("srcAccessOrder", ctypes.c_uint),
        ("srcLocHint", _CUmemLocation),
        ("dstLocHint", _CUmemLocation),
        ("flags", ctypes.c_uint),
    ]


_BATCH_MEMCPY_FUNC_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_uint,  # CUresult / hipError_t
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
)

# Resolved lazily on first use.
_batch_memcpy_fn: Any = None


def _resolve_batch_memcpy():
    """Resolve the platform batch-memcpy entry point (one-time).

    * CUDA: ``cuMemcpyBatchAsync`` via ``cuGetProcAddress`` (uses
      srcAccessOrder=STREAM via one attributes entry).
    * ROCm: ``hipMemcpyBatchAsync`` from libamdhip64 (ROCm 7.1+). ROCm
      7.2.1 or 7.2.2 rejects any call with ``numAttrs > 0``
      (see ROCm/clr @ rocm-7.2.1 hipamd/src/hip_memory.cpp:2819-2822), so
      we call with ``numAttrs=0``.

    Raises ``RuntimeError`` if the symbol is unavailable (older CUDA
    driver, ROCm < 7.1, unusual install). The connector requires the
    batch API.
    """
    if current_platform.is_rocm():
        try:
            lib = ctypes.CDLL("libamdhip64.so", mode=ctypes.RTLD_GLOBAL)
            fn = lib.hipMemcpyBatchAsync
        except (OSError, AttributeError) as e:
            raise RuntimeError(
                "hipMemcpyBatchAsync is unavailable in this ROCm install; "
                "SimpleCPUOffloadConnector requires ROCm 7.1+."
            ) from e
        fn.restype = ctypes.c_uint
        fn.argtypes = [
            ctypes.c_void_p,  # dsts
            ctypes.c_void_p,  # srcs
            ctypes.c_void_p,  # sizes
            ctypes.c_size_t,  # count
            ctypes.c_void_p,  # attrs
            ctypes.c_void_p,  # attrIdxs
            ctypes.c_size_t,  # numAttrs
            ctypes.c_void_p,  # failIdx
            ctypes.c_void_p,  # stream
        ]
        return fn

    from cuda.bindings import driver as drv

    err, ptr, _ = drv.cuGetProcAddress(b"cuMemcpyBatchAsync", 12080, 0)
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuGetProcAddress(cuMemcpyBatchAsync) failed: {err}")
    return _BATCH_MEMCPY_FUNC_TYPE(ptr)


class BatchMemcpyParams(NamedTuple):
    src_bases: np.ndarray  # [num_layers] uint64 — data_ptr per layer
    dst_bases: np.ndarray  # [num_layers] uint64
    src_bpb: np.ndarray  # [num_layers] uint64 — source stride per block
    dst_bpb: np.ndarray  # [num_layers] uint64 — destination stride per block
    copy_bpb: np.ndarray  # [num_layers] uint64 — payload bytes per block
    num_layers: int
    # CUDA only: one attributes entry carrying srcAccessOrder. Unused on ROCm
    # (7.2.1 or 7.2.2) because the current runtime rejects numAttrs > 0.
    attrs: _CUmemcpyAttributes
    attrs_idx: ctypes.c_size_t
    # NOTE: cuMemcpyBatchAsync_v2() removed fail_idx field, but we use
    # cuMemcpyBatchAsync() with fail_idx for backward compatibility
    fail_idx: ctypes.c_size_t
    stream_handle: int  # raw cudaStream_t / CUstream

    @property
    def bpb(self) -> np.ndarray:
        """Compatibility alias for payload bytes per block."""
        return self.copy_bpb


class BatchCopyStats(NamedTuple):
    """Host-side statistics for one submitted batch."""

    blocks: int
    descriptors: int
    coalesced_spans: int
    bytes: int
    workspace_reallocations: int


class BatchCopyWorkspace:
    """Reusable host arrays for one direction's batch submissions."""

    def __init__(self) -> None:
        self.src = np.empty(0, dtype=np.uint64)
        self.dst = np.empty(0, dtype=np.uint64)
        self.sizes = np.empty(0, dtype=np.uint64)
        self.capacity = 0
        self.reallocations = 0

    def ensure(self, count: int) -> None:
        if count <= self.capacity:
            return
        capacity = max(count, max(self.capacity * 2, 16))
        self.src = np.empty(capacity, dtype=np.uint64)
        self.dst = np.empty(capacity, dtype=np.uint64)
        self.sizes = np.empty(capacity, dtype=np.uint64)
        self.capacity = capacity
        self.reallocations += 1


def _contiguous_runs(
    src_block_ids: list[int], dst_block_ids: list[int]
) -> list[tuple[int, int, int]]:
    if len(src_block_ids) != len(dst_block_ids) or not src_block_ids:
        return []
    runs: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(src_block_ids) + 1):
        if index < len(src_block_ids):
            if (
                src_block_ids[index] == src_block_ids[index - 1] + 1
                and dst_block_ids[index] == dst_block_ids[index - 1] + 1
            ):
                continue
        runs.append((src_block_ids[start], dst_block_ids[start], index - start))
        start = index
    return runs


def build_params(
    src_caches: dict[str, torch.Tensor],
    dst_caches: dict[str, torch.Tensor],
    stream: torch.cuda.Stream,
    src_access_order: int = CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
) -> BatchMemcpyParams:
    global _batch_memcpy_fn
    if _batch_memcpy_fn is None:
        _batch_memcpy_fn = _resolve_batch_memcpy()

    assert list(src_caches.keys()) == list(dst_caches.keys())
    src_tensors = list(src_caches.values())
    dst_tensors = list(dst_caches.values())

    src_bases, dst_bases = [], []
    src_bpb, dst_bpb, copy_bpb = [], [], []
    for s, d in zip(src_tensors, dst_tensors):
        s_stride = s.stride(0) * s.element_size()
        d_stride = d.stride(0) * d.element_size()
        s_payload = (s.numel() // s.shape[0]) * s.element_size()
        d_payload = (d.numel() // d.shape[0]) * d.element_size()
        assert s_payload == d_payload
        src_bases.append(s.data_ptr())
        dst_bases.append(d.data_ptr())
        src_bpb.append(s_stride)
        dst_bpb.append(d_stride)
        copy_bpb.append(s_payload)

    attrs = _CUmemcpyAttributes(srcAccessOrder=src_access_order)

    return BatchMemcpyParams(
        src_bases=np.array(src_bases, dtype=np.uint64),
        dst_bases=np.array(dst_bases, dtype=np.uint64),
        src_bpb=np.array(src_bpb, dtype=np.uint64),
        dst_bpb=np.array(dst_bpb, dtype=np.uint64),
        copy_bpb=np.array(copy_bpb, dtype=np.uint64),
        num_layers=len(src_tensors),
        attrs=attrs,
        attrs_idx=ctypes.c_size_t(0),
        fail_idx=ctypes.c_size_t(0),
        stream_handle=stream.cuda_stream,
    )


def copy_blocks(
    src_block_ids: list[int],
    dst_block_ids: list[int],
    params: BatchMemcpyParams,
    workspace: BatchCopyWorkspace | None = None,
) -> BatchCopyStats:
    """Copy blocks via cuMemcpyBatchAsync / hipMemcpyBatchAsync."""
    n = len(src_block_ids)
    if n == 0:
        return BatchCopyStats(0, 0, 0, 0, 0)

    src_all: list[int] = []
    dst_all: list[int] = []
    sz_all: list[int] = []
    descriptors = 0
    coalesced_spans = 0
    total_bytes = 0
    for layer in range(params.num_layers):
        can_coalesce = (
            params.src_bpb[layer] == params.copy_bpb[layer]
            and params.dst_bpb[layer] == params.copy_bpb[layer]
        )
        runs = (
            _contiguous_runs(src_block_ids, dst_block_ids)
            if can_coalesce
            else [(src_id, dst_id, 1) for src_id, dst_id in zip(src_block_ids, dst_block_ids)]
        )
        for src_start, dst_start, run_length in runs:
            src_all.append(
                int(params.src_bases[layer] + src_start * params.src_bpb[layer])
            )
            dst_all.append(
                int(params.dst_bases[layer] + dst_start * params.dst_bpb[layer])
            )
            size = int(params.copy_bpb[layer]) * run_length
            sz_all.append(size)
            descriptors += 1
            coalesced_spans += int(run_length > 1)
            total_bytes += size

    total = descriptors
    if workspace is None:
        workspace = BatchCopyWorkspace()
    previous_reallocations = workspace.reallocations
    workspace.ensure(total)
    workspace.src[:total] = src_all
    workspace.dst[:total] = dst_all
    workspace.sizes[:total] = sz_all

    # ROCm 7.2.1/7.2.2 rejects any call with numAttrs>0 (hipMemcpyBatchAsync
    # hipamd/src/hip_memory.cpp:2819-2822); CUDA uses one attrs entry so
    # srcAccessOrder is honored. attrs / attrsIdxs are ignored when
    # numAttrs==0, so we pass the same values from both paths.
    num_attrs = 0 if current_platform.is_rocm() else 1
    err = _batch_memcpy_fn(
        workspace.dst.ctypes.data,
        workspace.src.ctypes.data,
        workspace.sizes.ctypes.data,
        total,
        ctypes.addressof(params.attrs),
        ctypes.byref(params.attrs_idx),
        num_attrs,
        ctypes.byref(params.fail_idx),
        params.stream_handle,
    )
    if err != 0:
        raise RuntimeError(
            f"batch memcpy failed: err={err} failIdx={params.fail_idx.value}"
        )
    return BatchCopyStats(
        n,
        descriptors,
        coalesced_spans,
        total_bytes,
        workspace.reallocations - previous_reallocations,
    )
