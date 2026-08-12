# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes

import numpy as np
import pytest

from vllm.v1.simple_kv_offload import cuda_mem_ops


def _make_params() -> cuda_mem_ops.BatchMemcpyParams:
    return cuda_mem_ops.BatchMemcpyParams(
        src_bases=np.array([1000, 2000], dtype=np.uint64),
        dst_bases=np.array([3000, 4000], dtype=np.uint64),
        src_bpb=np.array([10, 100], dtype=np.uint64),
        dst_bpb=np.array([10, 100], dtype=np.uint64),
        copy_bpb=np.array([10, 100], dtype=np.uint64),
        num_layers=2,
        attrs=cuda_mem_ops._CUmemcpyAttributes(srcAccessOrder=3),
        attrs_idx=ctypes.c_size_t(0),
        fail_idx=ctypes.c_size_t(0),
        stream_handle=0,
    )


def test_prepare_batch_copy_arrays_computes_expected_addresses() -> None:
    params = _make_params()
    arrays = cuda_mem_ops.prepare_batch_copy_arrays([1, 3], [2, 4], params)
    assert arrays is not None

    assert arrays.total == 4
    assert arrays.src_all.tolist() == [1010, 1030, 2100, 2300]
    assert arrays.dst_all.tolist() == [3020, 3040, 4200, 4400]
    assert arrays.sz_all.tolist() == [10, 10, 100, 100]


def test_prepare_batch_copy_arrays_validates_block_id_lengths() -> None:
    params = _make_params()

    with pytest.raises(ValueError, match="equal length"):
        _ = cuda_mem_ops.prepare_batch_copy_arrays([1], [2, 3], params)


def test_prepare_batch_copy_arrays_supports_different_src_dst_strides() -> None:
    params = cuda_mem_ops.BatchMemcpyParams(
        src_bases=np.array([1000], dtype=np.uint64),
        dst_bases=np.array([2000], dtype=np.uint64),
        src_bpb=np.array([32], dtype=np.uint64),
        dst_bpb=np.array([64], dtype=np.uint64),
        copy_bpb=np.array([32], dtype=np.uint64),
        num_layers=1,
        attrs=cuda_mem_ops._CUmemcpyAttributes(srcAccessOrder=3),
        attrs_idx=ctypes.c_size_t(0),
        fail_idx=ctypes.c_size_t(0),
        stream_handle=0,
    )
    arrays = cuda_mem_ops.prepare_batch_copy_arrays([1], [2], params)
    assert arrays is not None
    assert arrays.src_all.tolist() == [1032]
    assert arrays.dst_all.tolist() == [2128]
    assert arrays.sz_all.tolist() == [32]


def test_prepare_batch_copy_arrays_uses_copy_size_for_layer() -> None:
    params = cuda_mem_ops.BatchMemcpyParams(
        src_bases=np.array([1000], dtype=np.uint64),
        dst_bases=np.array([2000], dtype=np.uint64),
        src_bpb=np.array([96], dtype=np.uint64),
        dst_bpb=np.array([64], dtype=np.uint64),
        copy_bpb=np.array([64], dtype=np.uint64),
        num_layers=1,
        attrs=cuda_mem_ops._CUmemcpyAttributes(srcAccessOrder=3),
        attrs_idx=ctypes.c_size_t(0),
        fail_idx=ctypes.c_size_t(0),
        stream_handle=0,
    )
    arrays = cuda_mem_ops.prepare_batch_copy_arrays([1, 2], [3, 4], params)
    assert arrays is not None

    assert arrays.src_all.tolist() == [1096, 1192]
    assert arrays.dst_all.tolist() == [2192, 2256]
    assert arrays.sz_all.tolist() == [64, 64]


def test_prepare_batch_copy_arrays_interleaves_layers_per_block() -> None:
    params = cuda_mem_ops.BatchMemcpyParams(
        src_bases=np.array([100, 1000], dtype=np.uint64),
        dst_bases=np.array([200, 2000], dtype=np.uint64),
        src_bpb=np.array([8, 80], dtype=np.uint64),
        dst_bpb=np.array([16, 160], dtype=np.uint64),
        copy_bpb=np.array([8, 80], dtype=np.uint64),
        num_layers=2,
        attrs=cuda_mem_ops._CUmemcpyAttributes(srcAccessOrder=3),
        attrs_idx=ctypes.c_size_t(0),
        fail_idx=ctypes.c_size_t(0),
        stream_handle=0,
    )
    arrays = cuda_mem_ops.prepare_batch_copy_arrays([0, 1], [2, 3], params)
    assert arrays is not None

    assert arrays.src_all.tolist() == [100, 108, 1000, 1080]
    assert arrays.dst_all.tolist() == [232, 248, 2320, 2480]
    assert arrays.sz_all.tolist() == [8, 8, 80, 80]
