# SPDX-License-Identifier: Apache-2.0
"""Pure policy helpers for the optional SuperInfer connector."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SuperInferVLTInputs:
    predicted_ttft_s: float | None = None
    predicted_tbt_s: float | None = None
    predicted_future_delay_s: float = 0.0
    swap_bytes: int = 0
    swap_bandwidth_bytes_per_s: float | None = None


@dataclass(frozen=True, slots=True)
class SuperInferVLTScore:
    request_id: str
    score: float


def estimate_swap_time_s(
    swap_bytes: int, swap_bandwidth_bytes_per_s: float | None
) -> float:
    if swap_bytes <= 0:
        return 0.0
    if swap_bandwidth_bytes_per_s is None or swap_bandwidth_bytes_per_s <= 0:
        return math.inf
    return swap_bytes / swap_bandwidth_bytes_per_s


def compute_vlt_score(
    inputs: SuperInferVLTInputs,
    *,
    alpha: float = 3.0,
    beta_bandwidth: float = 1.0,
    beta_future: float = 1.0,
    slo_ttft_s: float | None = 5.0,
    slo_tbt_s: float | None = 0.1,
) -> float:
    if min(alpha, beta_bandwidth, beta_future) < 0:
        raise ValueError("SuperInfer VLT weights must be non-negative")

    return (
        _slo_penalty(inputs.predicted_ttft_s, slo_ttft_s)
        + alpha * _slo_penalty(inputs.predicted_tbt_s, slo_tbt_s)
        + beta_bandwidth
        * estimate_swap_time_s(
            inputs.swap_bytes, inputs.swap_bandwidth_bytes_per_s
        )
        + beta_future * max(inputs.predicted_future_delay_s, 0.0)
    )


def rank_vlt_candidates(
    candidates: dict[str, SuperInferVLTInputs],
    **kwargs: float | None,
) -> list[SuperInferVLTScore]:
    scored = [
        SuperInferVLTScore(
            request_id=request_id,
            score=compute_vlt_score(inputs, **kwargs),
        )
        for request_id, inputs in candidates.items()
    ]
    return sorted(scored, key=lambda item: (-item.score, item.request_id))


def _slo_penalty(value_s: float | None, slo_s: float | None) -> float:
    if value_s is None or slo_s is None or slo_s <= 0 or value_s <= slo_s:
        return 0.0
    overflow = (value_s - slo_s) / slo_s
    return max(overflow, 0.0) if math.isfinite(overflow) else 0.0
