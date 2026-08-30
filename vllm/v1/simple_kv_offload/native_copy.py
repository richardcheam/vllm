# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional native batch-copy submission for SimpleCPUOffloadConnector."""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.simple_kv_offload.cuda_mem_ops import BatchMemcpyParams

logger = init_logger(__name__)

_NATIVE_SOURCE = r"""
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class BatchPlan {
 public:
  BatchPlan(
      torch::Tensor src_bases,
      torch::Tensor dst_bases,
      torch::Tensor src_strides,
      torch::Tensor dst_strides,
      torch::Tensor copy_sizes,
      std::uintptr_t stream_handle,
      std::uint32_t src_access_order)
      : stream_(reinterpret_cast<cudaStream_t>(stream_handle)) {
    check_descriptor(src_bases, "src_bases");
    check_descriptor(dst_bases, "dst_bases");
    check_descriptor(src_strides, "src_strides");
    check_descriptor(dst_strides, "dst_strides");
    check_descriptor(copy_sizes, "copy_sizes");
    auto count = src_bases.numel();
    TORCH_CHECK(dst_bases.numel() == count, "base count mismatch");
    TORCH_CHECK(src_strides.numel() == count, "source stride count mismatch");
    TORCH_CHECK(dst_strides.numel() == count, "destination stride count mismatch");
    TORCH_CHECK(copy_sizes.numel() == count, "copy size count mismatch");
    copy(src_bases, &src_bases_);
    copy(dst_bases, &dst_bases_);
    copy(src_strides, &src_strides_);
    copy(dst_strides, &dst_strides_);
    copy(copy_sizes, &copy_sizes_);
    attrs_.srcAccessOrder = static_cast<cudaMemcpySrcAccessOrder>(src_access_order);
  }

  void submit(
      const std::vector<std::int64_t>& src_blocks,
      const std::vector<std::int64_t>& dst_blocks) {
    TORCH_CHECK(src_blocks.size() == dst_blocks.size(), "block count mismatch");
    if (src_blocks.empty()) {
      return;
    }

    auto started = std::chrono::steady_clock::now();
    const auto num_layers = src_bases_.size();
    const auto num_blocks = src_blocks.size();
    const auto count = num_layers * num_blocks;
    ensure_workspace(count);
    std::uint64_t bytes = 0;
    std::uint64_t descriptors = 0;
    std::uint64_t coalesced_spans = 0;
    for (std::size_t layer = 0; layer < num_layers; ++layer) {
      const bool can_coalesce =
          src_strides_[layer] == copy_sizes_[layer] &&
          dst_strides_[layer] == copy_sizes_[layer];
      std::size_t block = 0;
      while (block < num_blocks) {
        std::size_t run_length = 1;
        if (can_coalesce) {
          while (block + run_length < num_blocks &&
                 src_blocks[block + run_length] ==
                     src_blocks[block] + run_length &&
                 dst_blocks[block + run_length] ==
                     dst_blocks[block] + run_length) {
            ++run_length;
          }
        }
        TORCH_CHECK(src_blocks[block] >= 0, "negative source block id");
        TORCH_CHECK(dst_blocks[block] >= 0, "negative destination block id");
        auto src_offset = static_cast<std::uintptr_t>(src_blocks[block]) *
            src_strides_[layer];
        auto dst_offset = static_cast<std::uintptr_t>(dst_blocks[block]) *
            dst_strides_[layer];
        src_ptrs_[descriptors] =
            reinterpret_cast<const void*>(src_bases_[layer] + src_offset);
        dst_ptrs_[descriptors] =
            reinterpret_cast<void*>(dst_bases_[layer] + dst_offset);
        sizes_[descriptors] = copy_sizes_[layer] * run_length;
        bytes += copy_sizes_[layer] * run_length;
        ++descriptors;
        coalesced_spans += run_length > 1 ? 1 : 0;
        block += run_length;
      }
    }
    auto err = cudaMemcpyBatchAsync(
        dst_ptrs_.data(),
        src_ptrs_.data(),
        sizes_.data(),
        descriptors,
        &attrs_,
        attrs_idx_.data(),
        1,
        stream_);
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();
    submissions_.fetch_add(1, std::memory_order_relaxed);
    blocks_.fetch_add(num_blocks, std::memory_order_relaxed);
    descriptors_.fetch_add(descriptors, std::memory_order_relaxed);
    coalesced_spans_.fetch_add(coalesced_spans, std::memory_order_relaxed);
    bytes_.fetch_add(bytes, std::memory_order_relaxed);
    submit_ns_.fetch_add(static_cast<std::uint64_t>(elapsed), std::memory_order_relaxed);
    if (err != cudaSuccess) {
      errors_.fetch_add(1, std::memory_order_relaxed);
      TORCH_CHECK(
          false,
          "cudaMemcpyBatchAsync failed: ",
          cudaGetErrorString(err));
    }
  }

  py::dict stats() const {
    py::dict result;
    result["submissions"] = submissions_.load(std::memory_order_relaxed);
    result["blocks"] = blocks_.load(std::memory_order_relaxed);
    result["descriptors"] = descriptors_.load(std::memory_order_relaxed);
    result["coalesced_spans"] = coalesced_spans_.load(std::memory_order_relaxed);
    result["bytes"] = bytes_.load(std::memory_order_relaxed);
    result["submit_ns"] = submit_ns_.load(std::memory_order_relaxed);
    result["errors"] = errors_.load(std::memory_order_relaxed);
    result["workspace_reallocations"] =
        workspace_reallocations_.load(std::memory_order_relaxed);
    return result;
  }

 private:
  void ensure_workspace(std::size_t count) {
    if (count <= workspace_capacity_) {
      return;
    }
    workspace_capacity_ = std::max(count, std::max(workspace_capacity_ * 2, std::size_t(16)));
    dst_ptrs_.resize(workspace_capacity_);
    src_ptrs_.resize(workspace_capacity_);
    sizes_.resize(workspace_capacity_);
    attrs_idx_.resize(workspace_capacity_, 0);
    workspace_reallocations_.fetch_add(1, std::memory_order_relaxed);
  }

  static void check_descriptor(torch::Tensor tensor, const char* name) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be on CPU");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt64, name, " must be int64");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  }

  static void copy(torch::Tensor tensor, std::vector<std::uint64_t>* result) {
    auto* data = tensor.data_ptr<int64_t>();
    result->assign(data, data + tensor.numel());
  }

  std::vector<std::uint64_t> src_bases_;
  std::vector<std::uint64_t> dst_bases_;
  std::vector<std::uint64_t> src_strides_;
  std::vector<std::uint64_t> dst_strides_;
  std::vector<std::uint64_t> copy_sizes_;
  std::vector<void*> dst_ptrs_;
  std::vector<const void*> src_ptrs_;
  std::vector<std::size_t> sizes_;
  std::vector<std::size_t> attrs_idx_;
  std::size_t workspace_capacity_{0};
  cudaMemcpyAttributes attrs_{};
  cudaStream_t stream_;
  std::atomic<std::uint64_t> submissions_{0};
  std::atomic<std::uint64_t> blocks_{0};
  std::atomic<std::uint64_t> descriptors_{0};
  std::atomic<std::uint64_t> coalesced_spans_{0};
  std::atomic<std::uint64_t> bytes_{0};
  std::atomic<std::uint64_t> submit_ns_{0};
  std::atomic<std::uint64_t> errors_{0};
  std::atomic<std::uint64_t> workspace_reallocations_{0};
};

std::shared_ptr<BatchPlan> make_plan(
    torch::Tensor src_bases,
    torch::Tensor dst_bases,
    torch::Tensor src_strides,
    torch::Tensor dst_strides,
    torch::Tensor copy_sizes,
    std::uintptr_t stream_handle,
    std::uint32_t src_access_order) {
  return std::make_shared<BatchPlan>(
      src_bases,
      dst_bases,
      src_strides,
      dst_strides,
      copy_sizes,
      stream_handle,
      src_access_order);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<BatchPlan, std::shared_ptr<BatchPlan>>(m, "BatchPlan")
      .def("submit", &BatchPlan::submit, py::call_guard<py::gil_scoped_release>())
      .def("stats", &BatchPlan::stats);
  m.def("make_plan", &make_plan);
}
"""

_native_module: Any | None = None
_native_error: BaseException | None = None


def _load_native_module() -> Any:
    global _native_module, _native_error
    if _native_module is not None:
        return _native_module
    if _native_error is not None:
        raise RuntimeError("native KV copy backend is unavailable") from _native_error

    try:
        import fcntl
        import tempfile
        from torch.utils.cpp_extension import load_inline

        lock_path = os.path.join(
            tempfile.gettempdir(), "vllm_superinfer_native_copy.lock"
        )
        with open(lock_path, "w", encoding="ascii") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            _native_module = load_inline(
                name="vllm_superinfer_native_copy",
                cpp_sources=_NATIVE_SOURCE,
                with_cuda=True,
                extra_cflags=["-O3", "-std=c++17"],
                verbose=os.environ.get("VLLM_LOGGING_LEVEL") == "DEBUG",
            )
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        logger.info("NativeDmaCopyBackend: native batch submission enabled")
        return _native_module
    except BaseException as exc:
        _native_error = exc
        raise RuntimeError("failed to compile native KV copy backend") from exc


def _params_tensors(params: BatchMemcpyParams) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.from_numpy(array.astype("int64", copy=False))
        for array in (
            params.src_bases,
            params.dst_bases,
            params.src_bpb,
            params.dst_bpb,
            params.copy_bpb,
        )
    )


def build_native_plan(params: BatchMemcpyParams) -> Any:
    """Build an immutable native descriptor plan for one transfer direction."""
    src_bases, dst_bases, src_strides, dst_strides, copy_sizes = _params_tensors(
        params
    )
    return _load_native_module().make_plan(
        src_bases,
        dst_bases,
        src_strides,
        dst_strides,
        copy_sizes,
        int(params.stream_handle),
        int(params.attrs.srcAccessOrder),
    )


def submit_native_copy(
    plan: Any,
    src_blocks: list[int],
    dst_blocks: list[int],
) -> None:
    """Submit one heterogeneous batch through a persistent native plan."""
    if len(src_blocks) != len(dst_blocks):
        raise ValueError("source and destination block counts differ")
    plan.submit(src_blocks, dst_blocks)
