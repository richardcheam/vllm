from vllm.v1.core.sched.vlt import VLTInputs, compute_vlt_score, rank_vlt_candidates


def test_vlt_bandwidth_penalty_increases_with_transfer_size():
    small = compute_vlt_score(
        VLTInputs(swap_bytes=1 << 20, swap_bandwidth_bytes_per_s=1 << 30),
        alpha=0,
        beta_bandwidth=1,
        beta_future=0,
        slo_ttft=None,
        slo_tbt=None,
    )
    large = compute_vlt_score(
        VLTInputs(swap_bytes=8 << 20, swap_bandwidth_bytes_per_s=1 << 30),
        alpha=0,
        beta_bandwidth=1,
        beta_future=0,
        slo_ttft=None,
        slo_tbt=None,
    )
    assert large > small


def test_vlt_ranking_is_deterministic_for_equal_scores():
    ranked = rank_vlt_candidates(
        {"b": VLTInputs(), "a": VLTInputs()},
        alpha=0,
        beta_bandwidth=0,
        beta_future=0,
        slo_ttft=None,
        slo_tbt=None,
    )
    assert [candidate.request_id for candidate in ranked] == ["a", "b"]
