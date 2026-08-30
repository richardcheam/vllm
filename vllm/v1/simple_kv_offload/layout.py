# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV offload layout helpers for worker-side cache registration."""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Keep block-first activation narrow until broader model families are
# stress-validated in GPU runs.
_BLOCK_FIRST_ARCH_ALLOWLIST = frozenset({"OPTForCausalLM"})


@dataclass(frozen=True)
class OffloadSegmentLayout:
    name: str
    bytes_per_block: int
    storage_nbytes: int


@dataclass(frozen=True)
class OffloadLayoutDescriptor:
    mode: str
    num_blocks: int
    segments: tuple[OffloadSegmentLayout, ...]


@dataclass(frozen=True)
class LayoutModeDecision:
    mode: str
    reason: str | None = None
    block_first_eligible: bool = False


def choose_layout_mode(
    swapper_block_first: bool,
    *,
    model_arches: tuple[str, ...],
    num_kv_cache_groups: int,
    has_non_tensor_values: bool,
    tensor_parallel_size: int,
    aggressive_mode: bool = False,
) -> LayoutModeDecision:
    """Select active offload layout mode under strict safety gates."""
    if not swapper_block_first:
        return LayoutModeDecision(
            mode="gpu_derived", reason="flag disabled", block_first_eligible=False
        )

    if aggressive_mode:
        return LayoutModeDecision(
            mode="block_first",
            reason="high-risk override enabled",
            block_first_eligible=True,
        )

    if any("DeepseekV4" in arch for arch in model_arches):
        return LayoutModeDecision(
            mode="gpu_derived",
            reason="DeepSeek-V4 forced fallback",
            block_first_eligible=False,
        )

    if not model_arches:
        return LayoutModeDecision(
            mode="gpu_derived",
            reason="unknown model architecture",
            block_first_eligible=False,
        )

    if any(arch not in _BLOCK_FIRST_ARCH_ALLOWLIST for arch in model_arches):
        return LayoutModeDecision(
            mode="gpu_derived",
            reason="model architecture not in block-first allowlist",
            block_first_eligible=False,
        )

    if tensor_parallel_size != 1:
        return LayoutModeDecision(
            mode="gpu_derived", reason="tp_size != 1", block_first_eligible=False
        )

    if num_kv_cache_groups != 1:
        return LayoutModeDecision(
            mode="gpu_derived",
            reason="num_kv_cache_groups != 1",
            block_first_eligible=False,
        )

    if has_non_tensor_values:
        return LayoutModeDecision(
            mode="gpu_derived",
            reason="hybrid/mamba cache values detected",
            block_first_eligible=False,
        )

    return LayoutModeDecision(mode="block_first", reason=None, block_first_eligible=True)


def build_gpu_cache_views(
    kv_caches: dict[str, torch.Tensor],
    num_blocks: int,
    device: torch.device,
    mode: str = "gpu_derived",
) -> tuple[dict[str, torch.Tensor], OffloadLayoutDescriptor]:
    """Build per-storage [num_blocks, block_bytes] int8 views and descriptor.

    The returned mapping is a behavior-preserving view of existing KV cache
    storage used by simple CPU offload copy paths.
    """
    seen_ptrs: dict[int, tuple[str, torch.Tensor]] = {}
    for name, tensor in kv_caches.items():
        ptr = tensor.untyped_storage().data_ptr()
        if ptr not in seen_ptrs:
            seen_ptrs[ptr] = (name, tensor)

    unique_gpu_caches: dict[str, torch.Tensor] = {}
    segments: list[OffloadSegmentLayout] = []
    for name, tensor in seen_ptrs.values():
        storage = tensor.untyped_storage()
        raw = torch.empty(0, dtype=torch.int8, device=device).set_(
            storage, 0, (storage.nbytes(),)
        )
        el = tensor.element_size()
        page_size_bytes = storage.nbytes() // num_blocks
        outer_dims = [
            d for d in range(tensor.ndim) if tensor.stride(d) * el > page_size_bytes
        ]
        if not outer_dims:
            view = raw.view(num_blocks, -1)
            unique_gpu_caches[name] = view
            segments.append(
                OffloadSegmentLayout(
                    name=name,
                    bytes_per_block=view.stride(0) * view.element_size(),
                    storage_nbytes=storage.nbytes(),
                )
            )
        else:
            seg_stride = tensor.stride(outer_dims[0]) * el
            for idx in range(tensor.shape[outer_dims[0]]):
                seg_name = f"{name}.{idx}"
                offset = idx * seg_stride
                chunk = raw[offset : offset + seg_stride]
                view = chunk.view(num_blocks, -1)
                unique_gpu_caches[seg_name] = view
                segments.append(
                    OffloadSegmentLayout(
                        name=seg_name,
                        bytes_per_block=view.stride(0) * view.element_size(),
                        storage_nbytes=seg_stride,
                    )
                )

    return unique_gpu_caches, OffloadLayoutDescriptor(
        mode=mode,
        num_blocks=num_blocks,
        segments=tuple(segments),
    )
