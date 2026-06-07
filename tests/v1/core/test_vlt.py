# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest

from vllm.v1.core.sched.vlt import (
    VLTInputs,
    compute_vlt_score,
    estimate_swap_time_s,
    rank_vlt_candidates,
)


pytestmark = pytest.mark.cpu_test


def test_estimate_swap_time_s_with_valid_bandwidth():
    assert estimate_swap_time_s(1000, 500.0) == 2.0


def test_estimate_swap_time_s_with_zero_or_negative_bytes():
    assert estimate_swap_time_s(0, 100.0) == 0.0
    assert estimate_swap_time_s(-10, 100.0) == 0.0


def test_estimate_swap_time_s_with_invalid_bandwidth():
    assert math.isinf(estimate_swap_time_s(100, None))
    assert math.isinf(estimate_swap_time_s(100, 0.0))
    assert math.isinf(estimate_swap_time_s(100, -1.0))


def test_compute_vlt_score_uses_slo_overflow_and_weights():
    score = compute_vlt_score(
        VLTInputs(
            predicted_ttft_s=3.0,
            predicted_tbt_s=0.6,
            predicted_future_delay_s=2.0,
            swap_bytes=1000,
            swap_bandwidth_bytes_per_s=500.0,
        ),
        alpha=2.0,
        beta_bandwidth=0.5,
        beta_future=0.25,
        slo_ttft=2.0,
        slo_tbt=0.5,
    )
    # ttft overflow = (3-2)/2 = 0.5
    # tbt overflow = (0.6-0.5)/0.5 = 0.2 -> weighted by alpha=2 => 0.4
    # swap time = 2.0 -> weighted by beta_bandwidth=0.5 => 1.0
    # future delay = 2.0 -> weighted by beta_future=0.25 => 0.5
    assert score == pytest.approx(2.4)


def test_compute_vlt_score_under_slo_has_no_latency_penalty():
    score = compute_vlt_score(
        VLTInputs(
            predicted_ttft_s=1.0,
            predicted_tbt_s=0.1,
            predicted_future_delay_s=0.0,
            swap_bytes=0,
        ),
        alpha=1.0,
        beta_bandwidth=1.0,
        beta_future=1.0,
        slo_ttft=2.0,
        slo_tbt=0.5,
    )
    assert score == 0.0


def test_compute_vlt_score_invalid_weights_raise():
    with pytest.raises(ValueError):
        compute_vlt_score(
            VLTInputs(),
            alpha=-0.1,
            beta_bandwidth=0.0,
            beta_future=0.0,
            slo_ttft=None,
            slo_tbt=None,
        )


def test_rank_vlt_candidates_orders_highest_score_first_then_id():
    ranked = rank_vlt_candidates(
        {
            "b": VLTInputs(predicted_ttft_s=3.0),
            "a": VLTInputs(predicted_ttft_s=3.0),
            "c": VLTInputs(predicted_ttft_s=1.0),
        },
        alpha=0.0,
        beta_bandwidth=0.0,
        beta_future=0.0,
        slo_ttft=2.0,
        slo_tbt=None,
    )
    assert [item.request_id for item in ranked] == ["a", "b", "c"]
    assert ranked[0].score == pytest.approx(0.5)
    assert ranked[2].score == 0.0
