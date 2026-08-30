#!/usr/bin/env python3
"""Create a compact cross-profile comparison from matrix benchmark reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_dir", type=Path)
    args = parser.parse_args()
    rows = []
    profile_dirs = sorted(args.matrix_dir.glob(".env.*"))
    for profile_dir in profile_dirs:
        profile = profile_dir.name.removeprefix(".env.")
        summary_path = profile_dir / "measurement/report/baseline-summary.json"
        if not summary_path.exists():
            rows.append((profile, None, 0, 0, {}, "FAILED: no report"))
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = data["throughput_summary"]
        runs = data["runs"]
        successes = sum(run.get("successful", 0) for run in runs)
        failures = sum(run.get("failed", 0) for run in runs)
        runtime_metrics = data.get("runtime_metrics", {})
        rows.append((profile, summary, successes, failures, runtime_metrics, "OK"))
    if not rows:
        raise SystemExit(f"No profile reports found in {args.matrix_dir}")

    valid_rows = [row for row in rows if row[1] is not None]
    best = max(valid_rows, key=lambda row: row[1]["max"])[1]["max"] if valid_rows else None
    lines = [
        "# SuperInfer Profile Matrix",
        "",
        f"Matrix: `{args.matrix_dir}`",
        "",
        "| Profile | Mean tok/s | Median tok/s | Best tok/s | Min tok/s | Std. dev. | Success | Failures | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for profile, summary, successes, failures, runtime_metrics, status in rows:
        if summary is None:
            lines.append(f"| {profile} | n/a | n/a | n/a | n/a | n/a | {successes} | {failures} | {status} |")
            continue
        kv_keys = len(runtime_metrics)
        lines.append(
            f"| {profile} | {summary['mean']:,.1f} | {summary['median']:,.1f} | "
            f"{summary['max']:,.1f} | {summary['min']:,.1f} | {summary['stdev']:,.1f} | "
            f"{successes} | {failures} | {status} |"
        )
    lines.extend(
        [
            "",
            f"Best observed profile throughput: **{best:,.1f} completion tokens/s**."
            if best is not None
            else "No valid profile throughput was recorded.",
            "",
            "This comparison is only valid when profiles use identical warmup, "
            "model, workload, and hardware conditions.",
        ]
    )
    (args.matrix_dir / "matrix-results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.matrix_dir / 'matrix-results.md'}")


if __name__ == "__main__":
    main()
