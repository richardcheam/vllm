# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.simple_kv_offload.layout import (
    build_gpu_cache_views,
    choose_layout_mode,
)


def test_build_gpu_cache_views_deduplicates_shared_storage() -> None:
    num_blocks = 4
    base = torch.zeros((num_blocks, 8), dtype=torch.float16)
    kv_caches = {
        "layer_a": base,
        "layer_b": base,
    }

    views, descriptor = build_gpu_cache_views(
        kv_caches,
        num_blocks=num_blocks,
        device=torch.device("cpu"),
    )

    assert list(views.keys()) == ["layer_a"]
    assert descriptor.mode == "gpu_derived"
    assert descriptor.num_blocks == num_blocks
    assert len(descriptor.segments) == 1
    assert descriptor.segments[0].name == "layer_a"
    assert descriptor.segments[0].bytes_per_block == views["layer_a"].stride(0)


def test_build_gpu_cache_views_splits_outer_segment_dimension() -> None:
    num_blocks = 4
    # Shape with an outer segment dim (K/V), similar to FlashAttn layout.
    kv = torch.zeros((2, num_blocks, 4), dtype=torch.float16)

    views, descriptor = build_gpu_cache_views(
        {"attn": kv},
        num_blocks=num_blocks,
        device=torch.device("cpu"),
    )

    assert sorted(views.keys()) == ["attn.0", "attn.1"]
    assert len(descriptor.segments) == 2
    assert descriptor.segments[0].name == "attn.0"
    assert descriptor.segments[1].name == "attn.1"


def test_build_gpu_cache_views_descriptor_tracks_segment_sizes() -> None:
    num_blocks = 3
    dense = torch.zeros((num_blocks, 5), dtype=torch.float16)
    split = torch.zeros((2, num_blocks, 4), dtype=torch.float16)

    views, descriptor = build_gpu_cache_views(
        {"dense": dense, "attn": split},
        num_blocks=num_blocks,
        device=torch.device("cpu"),
        mode="block_first",
    )

    assert descriptor.mode == "block_first"
    assert tuple(views.keys()) == tuple(seg.name for seg in descriptor.segments)
    for seg in descriptor.segments:
        assert seg.storage_nbytes == seg.bytes_per_block * num_blocks


def test_choose_layout_mode_requires_strict_safety_gates() -> None:
    assert choose_layout_mode(
        True,
        model_arches=("DeepseekV4ForCausalLM",),
        num_kv_cache_groups=1,
        has_non_tensor_values=False,
        tensor_parallel_size=1,
    ).block_first_eligible is False

    assert choose_layout_mode(
        True,
        model_arches=("OPTForCausalLM",),
        num_kv_cache_groups=2,
        has_non_tensor_values=False,
        tensor_parallel_size=1,
    ).reason == "num_kv_cache_groups != 1"

    decision = choose_layout_mode(
        True,
        model_arches=("OPTForCausalLM",),
        num_kv_cache_groups=1,
        has_non_tensor_values=False,
        tensor_parallel_size=1,
    )
    assert decision.mode == "block_first"
    assert decision.block_first_eligible is True
    assert decision.reason is None


def test_choose_layout_mode_reports_fallback_reasons() -> None:
    assert choose_layout_mode(
        False,
        model_arches=("OPTForCausalLM",),
        num_kv_cache_groups=1,
        has_non_tensor_values=False,
        tensor_parallel_size=1,
    ).reason == "flag disabled"

    assert choose_layout_mode(
        True,
        model_arches=(),
        num_kv_cache_groups=1,
        has_non_tensor_values=False,
        tensor_parallel_size=1,
    ).reason == "unknown model architecture"

    assert choose_layout_mode(
        True,
        model_arches=("LlamaForCausalLM",),
        num_kv_cache_groups=1,
        has_non_tensor_values=False,
        tensor_parallel_size=1,
    ).reason == "model architecture not in block-first allowlist"

    assert choose_layout_mode(
        True,
        model_arches=("OPTForCausalLM",),
        num_kv_cache_groups=1,
        has_non_tensor_values=True,
        tensor_parallel_size=1,
    ).reason == "hybrid/mamba cache values detected"

    assert choose_layout_mode(
        True,
        model_arches=("OPTForCausalLM",),
        num_kv_cache_groups=1,
        has_non_tensor_values=False,
        tensor_parallel_size=2,
    ).reason == "tp_size != 1"


def test_choose_layout_mode_aggressive_mode_overrides_safety_gates() -> None:
    decision = choose_layout_mode(
        True,
        model_arches=("DeepseekV4ForCausalLM",),
        num_kv_cache_groups=8,
        has_non_tensor_values=True,
        tensor_parallel_size=2,
        aggressive_mode=True,
    )

    assert decision.mode == "block_first"
    assert decision.block_first_eligible is True
    assert decision.reason == "high-risk override enabled"
