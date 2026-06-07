# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VLTInputs:
    predicted_ttft_s: float | None = None
    predicted_tbt_s: float | None = None
    predicted_future_delay_s: float = 0.0
    swap_bytes: int = 0
    swap_bandwidth_bytes_per_s: float | None = None


@dataclass(frozen=True, slots=True)
class VLTScoredCandidate:
    request_id: str
    score: float


def estimate_swap_time_s(swap_bytes: int, swap_bandwidth_bytes_per_s: float | None) -> float:
    if swap_bytes <= 0:
        return 0.0
    if swap_bandwidth_bytes_per_s is None or swap_bandwidth_bytes_per_s <= 0:
        return float("inf")
    return swap_bytes / swap_bandwidth_bytes_per_s


def compute_vlt_score(
    inputs: VLTInputs,
    *,
    alpha: float,
    beta_bandwidth: float,
    beta_future: float,
    slo_ttft: float | None,
    slo_tbt: float | None,
) -> float:
    if alpha < 0 or beta_bandwidth < 0 or beta_future < 0:
        raise ValueError("VLT weights must be non-negative")

    ttft_penalty = _normalized_slo_penalty(inputs.predicted_ttft_s, slo_ttft)
    tbt_penalty = _normalized_slo_penalty(inputs.predicted_tbt_s, slo_tbt)
    swap_penalty = estimate_swap_time_s(
        inputs.swap_bytes,
        inputs.swap_bandwidth_bytes_per_s,
    )
    future_penalty = max(inputs.predicted_future_delay_s, 0.0)

    return (
        ttft_penalty
        + alpha * tbt_penalty
        + beta_bandwidth * swap_penalty
        + beta_future * future_penalty
    )


def rank_vlt_candidates(
    candidates: dict[str, VLTInputs] | Iterable[tuple[str, VLTInputs]],
    *,
    alpha: float,
    beta_bandwidth: float,
    beta_future: float,
    slo_ttft: float | None,
    slo_tbt: float | None,
) -> list[VLTScoredCandidate]:
    if isinstance(candidates, dict):
        iterator = candidates.items()
    else:
        iterator = candidates

    scored = [
        VLTScoredCandidate(
            request_id=req_id,
            score=compute_vlt_score(
                inputs,
                alpha=alpha,
                beta_bandwidth=beta_bandwidth,
                beta_future=beta_future,
                slo_ttft=slo_ttft,
                slo_tbt=slo_tbt,
            ),
        )
        for req_id, inputs in iterator
    ]
    return sorted(scored, key=lambda item: (-item.score, item.request_id))


def _normalized_slo_penalty(value_s: float | None, slo_s: float | None) -> float:
    if value_s is None or slo_s is None or slo_s <= 0:
        return 0.0
    if value_s <= slo_s:
        return 0.0

    overflow = (value_s - slo_s) / slo_s
    if math.isnan(overflow) or overflow < 0:
        return 0.0
    return overflow
