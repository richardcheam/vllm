#!/usr/bin/env python3
"""Generate a dependency-free Markdown/SVG report from benchmark JSON runs."""

from __future__ import annotations

import argparse
import datetime
import json
import re
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_runs(input_dir: Path) -> list[dict]:
    runs = []
    for path in sorted(input_dir.glob("run_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["run_name"] = path.stem
        add_latency_summaries(data)
        runs.append(data)
    if not runs:
        raise SystemExit(f"No run_*.json files found in {input_dir}")
    return runs


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def add_latency_summaries(run: dict) -> None:
    """Normalize old precomputed and robust raw-array benchmark schemas."""
    if "ttft_summary_ms" not in run:
        values = [float(value) * 1000 for value in run.get("ttfts", [])]
        run["ttft_summary_ms"] = {
            f"{name}_ms": value
            for name, value in (
                ("mean", statistics.mean(values) if values else None),
                ("p50", percentile(values, 50)),
                ("p90", percentile(values, 90)),
                ("p99", percentile(values, 99)),
            )
        }
    if "tbt_summary_ms" not in run:
        values = [float(value) * 1000 for value in run.get("tbt", [])]
        run["tbt_summary_ms"] = {
            f"{name}_ms": value
            for name, value in (
                ("mean", statistics.mean(values) if values else None),
                ("p50", percentile(values, 50)),
                ("p90", percentile(values, 90)),
                ("p99", percentile(values, 99)),
            )
        }


def log_diagnostics(input_dir: Path) -> dict[str, int]:
    path = input_dir / "server.log"
    if not path.exists():
        return {"error_lines": 0, "traceback_lines": 0, "engine_dead_lines": 0, "model_404_lines": 0}
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "error_lines": sum(" ERROR " in line for line in text.splitlines()),
        "traceback_lines": text.count("Traceback (most recent call last)"),
        "engine_dead_lines": text.count("EngineDeadError"),
        "model_404_lines": text.count("does not exist."),
    }


_PROMETHEUS_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+0-9.eE]+)(?:\s+\d+)?$"
)
_TRANSFER_METRIC_PREFIX = "vllm:simple_cpu_offload_"
_NON_TRANSFER_COUNTER_PREFIXES = (
    "offload_cpu_kv_",
)


def parse_metric_samples(path: Path) -> dict[str, float]:
    """Parse Prometheus samples while retaining their label sets."""
    samples: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        match = _PROMETHEUS_SAMPLE_RE.match(line)
        if match is None:
            continue
        name = match.group("name")
        labels = match.group("labels")
        sample_name = f"{name}{{{labels}}}" if labels else name
        try:
            samples[sample_name] = float(match.group("value"))
        except ValueError:
            continue
    return samples


def metric_totals(input_dir: Path) -> dict[str, float]:
    """Read labeled SimpleCPUOffload samples captured beside repeats."""
    totals: dict[str, float] = {}
    for path in input_dir.glob("run_*.after.metrics.txt"):
        for name, value in parse_metric_samples(path).items():
            if name.split("{", 1)[0].startswith(_TRANSFER_METRIC_PREFIX):
                totals[name] = max(totals.get(name, 0.0), value)
    return totals


def metric_counter_deltas(input_dir: Path) -> dict[str, float]:
    """Sum counter deltas measured around each benchmark repeat."""
    totals: dict[str, float] = {}
    for before_path in sorted(input_dir.glob("run_*.before.metrics.txt")):
        after_path = before_path.with_name(
            before_path.name.replace(".before.", ".after.")
        )
        if not after_path.exists():
            continue
        before = parse_metric_samples(before_path)
        after = parse_metric_samples(after_path)
        for name, after_value in after.items():
            metric_name = name.split("{", 1)[0]
            if not metric_name.startswith(_TRANSFER_METRIC_PREFIX):
                continue
            if not metric_name.endswith("_total"):
                continue
            short_name = metric_name[len(_TRANSFER_METRIC_PREFIX) :]
            if short_name.startswith(_NON_TRANSFER_COUNTER_PREFIXES):
                continue
            delta = after_value - before.get(name, 0.0)
            if delta >= 0:
                totals[name] = totals.get(name, 0.0) + delta
    return totals


def transfer_summary(runtime_metrics: dict[str, float]) -> dict[str, float]:
    """Aggregate labeled transfer samples into a compact summary."""
    summary: dict[str, float] = {}
    for sample_name, value in runtime_metrics.items():
        name = sample_name.split("{", 1)[0]
        key = name[len(_TRANSFER_METRIC_PREFIX) :]
        if key.endswith("_created") or key.endswith("_created_created"):
            continue
        if key.endswith("_total"):
            key = key[: -len("_total")]
        summary[key] = max(summary.get(key, 0.0), value)
    return dict(sorted(summary.items()))


def fmt(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f}"


def metric_values(runs: list[dict], key: str) -> list[float]:
    return [float(run[key]) for run in runs if run.get(key) is not None]


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def svg_bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    unit: str,
    color: str,
) -> None:
    available = [
        (label, value) for label, value in zip(labels, values) if value is not None
    ]
    if not available:
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="440">'
            '<text x="40" y="80" font-family="system-ui,sans-serif" '
            'font-size="24">No persisted data for this metric</text></svg>\n',
            encoding="utf-8",
        )
        return
    labels = [label for label, _ in available]
    values = [value for _, value in available]
    width, height = 900, 440
    left, right, top, bottom = 90, 30, 70, 80
    chart_w = width - left - right
    chart_h = height - top - bottom
    maximum = max(values) if values else 1.0
    bar_w = chart_w / max(len(values), 1) * 0.62
    bars = []
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + (index + 0.5) * chart_w / len(values) - bar_w / 2
        bar_h = chart_h * value / maximum if maximum else 0
        y = top + chart_h - bar_h
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
            f'height="{bar_h:.1f}" rx="6" fill="{color}"/>'
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" '
            f'text-anchor="middle" class="value">{value:,.1f}</text>'
            f'<text x="{x + bar_w / 2:.1f}" y="{height - 42}" '
            f'text-anchor="middle" class="label">{label}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>text{{font-family:system-ui,sans-serif;fill:#233044}}.title{{font-size:24px;font-weight:700}}.value{{font-size:15px;font-weight:700}}.label{{font-size:14px}}.axis{{stroke:#b8c2d1;stroke-width:1}}</style>
<rect width="100%" height="100%" fill="#f7f9fc"/>
<text x="{left}" y="36" class="title">{title}</text>
<line x1="{left}" y1="{top + chart_h}" x2="{width - right}" y2="{top + chart_h}" class="axis"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" class="axis"/>
{''.join(bars)}
<text x="18" y="{top + chart_h / 2}" transform="rotate(-90 18 {top + chart_h / 2})" text-anchor="middle" class="label">{unit}</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_report(runs: list[dict], output_dir: Path, input_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = log_diagnostics(input_dir)
    runtime_metrics = metric_totals(input_dir)
    transfer_metrics = transfer_summary(runtime_metrics)
    transfer_delta_metrics = transfer_summary(metric_counter_deltas(input_dir))
    throughput = metric_values(runs, "api_completion_tokens_per_second")
    total_times = metric_values(runs, "total_time")
    ttft_p50 = [run["ttft_summary_ms"].get("p50_ms") for run in runs]
    ttft_p99 = [run["ttft_summary_ms"].get("p99_ms") for run in runs]
    tbt_p50 = [run["tbt_summary_ms"].get("p50_ms") for run in runs]
    tbt_p99 = [run["tbt_summary_ms"].get("p99_ms") for run in runs]
    throughput_summary = summary(throughput)

    labels = [run["run_name"].replace("run_", "run ") for run in runs]
    svg_bar_chart(
        output_dir / "baseline-throughput.svg",
        "Completion Throughput by Repeat",
        labels,
        throughput,
        "completion tokens / second",
        "#2f7d8c",
    )
    svg_bar_chart(
        output_dir / "baseline-ttft-p99.svg",
        "TTFT P99 by Repeat",
        labels,
        ttft_p99,
        "milliseconds",
        "#d97745",
    )
    svg_bar_chart(
        output_dir / "baseline-tbt-p99.svg",
        "TBT P99 by Repeat",
        labels,
        tbt_p99,
        "milliseconds",
        "#6c63a8",
    )

    first = runs[0]
    total_successful = sum(run.get("successful", 0) for run in runs)
    total_requests = sum(run.get("requests", 0) for run in runs)
    prompt_token_ranges = [
        (run.get("prompt_tokens_min"), run.get("prompt_tokens_max"))
        for run in runs
        if run.get("prompt_tokens_min") is not None
        and run.get("prompt_tokens_max") is not None
    ]
    prompt_token_range = (
        f"{min(item[0] for item in prompt_token_ranges)}-"
        f"{max(item[1] for item in prompt_token_ranges)}"
        if prompt_token_ranges
        else "n/a"
    )
    report = f'''# SuperInfer Benchmark Report

**Run directory:** `{input_dir}`  
**Generated:** {datetime.date.today().isoformat()}  
**Status:** {'PASS' if total_successful == total_requests else 'INCOMPLETE'}; interpret with the workload and transfer scope below.

## Configuration

| Field | Value |
|---|---:|
| Model | `{first.get("model", "n/a")}` |
| Prompt length | {first.get("prompt_len", "n/a")} tokens |
| Measured prompt-token range | {prompt_token_range} |
| Users | {first.get("users", "n/a")} |
| Requests per repeat | {first.get("requests", "n/a")} |
| Max output tokens | {first.get("max_tokens", "n/a")} |
| Successful requests | {sum(run.get("successful", 0) for run in runs)}/{sum(run.get("requests", 0) for run in runs)} |
| Failed requests | {sum(run.get("failed", 0) for run in runs)} |
| Server log ERROR lines | {diagnostics["error_lines"]} |
| Server log Tracebacks | {diagnostics["traceback_lines"]} |
| EngineDeadError occurrences | {diagnostics["engine_dead_lines"]} |
| Model 404 log lines | {diagnostics["model_404_lines"]} |
| Captured Prometheus KV metrics | {len(runtime_metrics)} |

## Transfer Summary

| Metric | Final captured value |
|---|---:|
'''
    for key, value in transfer_metrics.items():
        report += f"| `{key}` | {fmt(value)} |\n"
    report += '''

## Measured Transfer Deltas

These are counter increases captured between each repeat's before/after
snapshots. They exclude transfer activity outside the measurement repeats.

| Metric | Measured delta |
|---|---:|
'''
    for key, value in transfer_delta_metrics.items():
        report += f"| `{key}` | {fmt(value)} |\n"
    if not transfer_delta_metrics:
        report += "| n/a | n/a |\n"
    report += '''

## Repeat Results

| Repeat | Completion tok/s | Total time (s) | TTFT mean (ms) | TTFT P50 (ms) | TTFT P99 (ms) | TBT mean (ms) | TBT P50 (ms) | TBT P99 (ms) | Success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''
    for run in runs:
        report += (
            f'| {run["run_name"]} | '
            f'{fmt(run.get("api_completion_tokens_per_second"))} | '
            f'{fmt(run.get("total_time"))} | '
            f'{fmt(run["ttft_summary_ms"].get("mean_ms"))} | '
            f'{fmt(run["ttft_summary_ms"].get("p50_ms"))} | '
            f'{fmt(run["ttft_summary_ms"].get("p99_ms"))} | '
            f'{fmt(run["tbt_summary_ms"].get("mean_ms"))} | '
            f'{fmt(run["tbt_summary_ms"].get("p50_ms"))} | '
            f'{fmt(run["tbt_summary_ms"].get("p99_ms"))} | '
            f'{run.get("successful", 0)}/{run.get("failed", 0)} |\n'
        )

    def aggregate_optional(values: list[float | None], name: str) -> str:
        available = [value for value in values if value is not None]
        if not available:
            return f"| {name} | n/a | n/a | n/a | n/a | n/a |\n"
        return (
            f"| {name} | {fmt(statistics.mean(available))} | "
            f"{fmt(statistics.median(available))} | {fmt(min(available))} | "
            f"{fmt(max(available))} | n/a |\n"
        )

    report += f'''
## Aggregate View

| Metric | Mean | Median | Min | Max | Std. dev. |
|---|---:|---:|---:|---:|---:|
| Completion tokens/s | {fmt(throughput_summary["mean"])} | {fmt(throughput_summary["median"])} | {fmt(throughput_summary["min"])} | {fmt(throughput_summary["max"])} | {fmt(throughput_summary["stdev"])} |
| Total time (s) | {fmt(statistics.mean(total_times))} | {fmt(statistics.median(total_times))} | {fmt(min(total_times))} | {fmt(max(total_times))} | {fmt(statistics.stdev(total_times) if len(total_times) > 1 else 0)} |
{aggregate_optional(ttft_p50, "TTFT P50 (ms)")}{aggregate_optional(ttft_p99, "TTFT P99 (ms)")}{aggregate_optional(tbt_p50, "TBT P50 (ms)")}{aggregate_optional(tbt_p99, "TBT P99 (ms)")}

## Visualizations

![Completion throughput by repeat](baseline-throughput.svg)

![TTFT P99 by repeat](baseline-ttft-p99.svg)

![TBT P99 by repeat](baseline-tbt-p99.svg)

## Interpretation

- {total_successful}/{total_requests} requests succeeded across {len(runs)} measurement repeat(s).
- Best observed completion throughput was **{throughput_summary["max"]:,.1f} tokens/s**.
- Median repeat throughput was **{throughput_summary["median"]:,.1f} tokens/s**.
- Repeat variability is {'not estimable from one repeat' if len(runs) == 1 else f'{throughput_summary["stdev"]:,.1f} tokens/s standard deviation'}.
- The final captured Prometheus values are cumulative for the server lifetime; use **Measured Transfer Deltas** for activity attributable to the benchmark repeats.
- Unique-prefix pressure primarily measures D2H store pressure. It does not measure H2D reload performance unless the measured requests reuse prefixes retained in CPU KV.
- Log diagnostics are reported separately from request success. Engine-dead or transfer warnings still require investigation even when clients complete.
- Captured runtime metrics are summarized in `runtime-metrics.json` when the server exposes `/metrics`.
'''
    (output_dir / "baseline-results.md").write_text(report, encoding="utf-8")
    summary_data = {
        "input_dir": str(input_dir),
        "runs": runs,
        "throughput_summary": throughput_summary,
        "log_diagnostics": diagnostics,
        "runtime_metrics": runtime_metrics,
        "transfer_metrics": transfer_metrics,
        "transfer_delta_metrics": transfer_delta_metrics,
    }
    (output_dir / "baseline-summary.json").write_text(
        json.dumps(summary_data, indent=2), encoding="utf-8"
    )
    (output_dir / "runtime-metrics.json").write_text(
        json.dumps(runtime_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "transfer-summary.json").write_text(
        json.dumps(transfer_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "transfer-deltas.json").write_text(
        json.dumps(transfer_delta_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    write_report(load_runs(args.input_dir), args.output_dir, args.input_dir)
    print(f"Wrote report to {args.output_dir}")


if __name__ == "__main__":
    main()
