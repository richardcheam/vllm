#!/usr/bin/env python3
"""Summarize staged serving-soak candidates and select a serving profile."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def phase_summary(results: list[dict]) -> dict:
    if results and "reload" in results[0]:
        validation = results[0].get("validation", {})
        return {
            "repeats": len(results),
            "status": "PASS"
            if all(r.get("status") == "PASS" for r in results)
            else "FAIL",
            "successful": 0,
            "failed": 0,
            "throughput_mean": None,
            "throughput_median": None,
            "throughput_min": None,
            "transfer_deltas": {
                "load_bytes": float(
                    validation.get("metric_deltas", {}).get("load_bytes", 0.0)
                ),
                "store_bytes": float(
                    validation.get("metric_deltas", {}).get("store_bytes", 0.0)
                ),
            },
            "max_dirty_blocks": 0.0,
            "max_invalid_blocks": 0.0,
            "quiescent_captures": 0,
            "validation": validation,
            "phase_timings_s": results[0].get("phase_timings_s", {}),
        }

    throughputs = [
        float(result["run"]["api_completion_tokens_per_second"])
        for result in results
        if result.get("run", {}).get("api_completion_tokens_per_second")
        is not None
    ]
    return {
        "repeats": len(results),
        "status": "PASS" if all(r.get("status") == "PASS" for r in results) else "FAIL",
        "successful": sum(int(r.get("successful", 0)) for r in results),
        "failed": sum(int(r.get("failed", 0)) for r in results),
        "throughput_mean": statistics.mean(throughputs) if throughputs else None,
        "throughput_median": statistics.median(throughputs) if throughputs else None,
        "throughput_min": min(throughputs) if throughputs else None,
        "transfer_deltas": {
            key: sum(float(r.get("transfer_deltas", {}).get(key, 0.0)) for r in results)
            for key in (
                "store_events",
                "load_events",
                "store_bytes",
                "load_bytes",
                "native_store_submissions",
                "native_load_submissions",
            )
        },
        "max_dirty_blocks": max(
            (float(r.get("final_gauges", {}).get("dirty_blocks", 0.0)) for r in results),
            default=0.0,
        ),
        "max_invalid_blocks": max(
            (float(r.get("final_gauges", {}).get("invalid_blocks", 0.0)) for r in results),
            default=0.0,
        ),
        "quiescent_captures": sum(
            bool(r.get("transfer_quiescent_at_capture")) for r in results
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for candidate_dir in sorted((args.sequence_dir / "profiles").glob("*")):
        config_path = candidate_dir / "candidate-config.json"
        if not config_path.exists():
            continue
        config = load_json(config_path)
        phase_results: dict[str, list[dict]] = {}
        for path in sorted(candidate_dir.glob("*/repeat_*/phase-result.json")):
            result = load_json(path)
            phase_results.setdefault(result["stage"], []).append(result)
        phases = {name: phase_summary(results) for name, results in phase_results.items()}
        required = {
            "warm_4k",
            "warm_8k",
            "short_peak_16k",
            "mid_32k",
            "mid_64k",
            "long_131k_4u",
            "long_131k_8u",
            "long_131k_16u",
            "reload_131k",
            "recovery_16k",
        }
        complete = required.issubset(phases)
        stable = complete and all(
            phases[name]["status"] == "PASS" for name in required
        ) and (
            phases["long_131k_16u"]["transfer_deltas"].get("store_bytes", 0.0)
            > 0
            and phases["reload_131k"]["status"] == "PASS"
        )
        candidates.append(
            {
                "config": config,
                "candidate": config["name"],
                "complete": complete,
                "stable": stable,
                "phases": phases,
            }
        )

    stable_candidates = [candidate for candidate in candidates if candidate["stable"]]
    short_rank = sorted(
        (
            candidate
            for candidate in candidates
            if candidate["phases"].get("short_peak_16k", {}).get("status") == "PASS"
        ),
        key=lambda candidate: candidate["phases"]["short_peak_16k"]["throughput_median"],
        reverse=True,
    )
    long_rank = sorted(
        (
            candidate
            for candidate in candidates
            if candidate["phases"].get("long_131k_16u", {}).get("status") == "PASS"
        ),
        key=lambda candidate: candidate["phases"]["long_131k_16u"]["throughput_median"],
        reverse=True,
    )
    recommended = max(
        stable_candidates,
        key=lambda candidate: candidate["phases"]["short_peak_16k"]["throughput_median"],
        default=None,
    )
    summary = {
        "status": "PASS" if stable_candidates else "NO_FULLY_STABLE_CANDIDATE",
        "sequence_dir": str(args.sequence_dir),
        "candidates": candidates,
        "recommended_candidate": recommended["candidate"] if recommended else None,
        "best_short_context_candidate": (
            short_rank[0]["candidate"] if short_rank else None
        ),
        "best_long_context_candidate": long_rank[0]["candidate"] if long_rank else None,
    }
    (args.output_dir / "serving-soak-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    report = [
        "# Serving Soak Report",
        "",
        f"Sequence: `{args.sequence_dir}`",
        f"Status: **{summary['status']}**",
        "",
        "## Recommendation",
        "",
        f"- Combined serving candidate: `{summary['recommended_candidate'] or 'none'}`",
        f"- Best short-context candidate: `{summary['best_short_context_candidate'] or 'none'}`",
        f"- Best long-context candidate: `{summary['best_long_context_candidate'] or 'none'}`",
        "",
        "## Candidates",
        "",
        "| Candidate | Config | Full soak | Short median tok/s | 131K/16u median tok/s | 131K store bytes | Reload load bytes | Invalid blocks |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        short = candidate["phases"].get("short_peak_16k", {})
        long = candidate["phases"].get("long_131k_16u", {})
        transfer = long.get("transfer_deltas", {})
        reload = candidate["phases"].get("reload_131k", {})
        reload_transfer = reload.get("transfer_deltas", {})
        report.append(
            "| `{name}` | `{tokens}/{seqs}/budget{budget}` | {stable} | "
            "{short:.1f} | {long:.1f} | {store:.0f} | {reload_load:.0f} | "
            "{invalid:.0f} |".format(
                name=candidate["candidate"],
                tokens=candidate["config"]["max_num_batched_tokens"],
                seqs=candidate["config"]["max_num_seqs"],
                budget=candidate["config"]["proactive_swap_budget"],
                stable="PASS" if candidate["stable"] else "FAIL",
                short=short.get("throughput_median") or 0.0,
                long=long.get("throughput_median") or 0.0,
                store=transfer.get("store_bytes", 0.0),
                reload_load=reload_transfer.get("load_bytes", 0.0),
                invalid=long.get("max_invalid_blocks", 0.0),
            )
        )
    report.extend(
        [
            "",
            "## Selection Rule",
            "",
            "The combined recommendation is the highest short-context median "
            "throughput among candidates that complete every staged phase, "
            "including 131K pressure at 16 users and post-pressure recovery, "
            "with zero failed requests and zero invalid residency blocks.",
            "",
            "A candidate with a higher short-context number but a failed long "
            "phase is reported separately and is not selected as the serving "
            "profile.",
        ]
    )
    (args.output_dir / "serving-soak-report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"Wrote serving soak report to {args.output_dir}")


if __name__ == "__main__":
    main()
