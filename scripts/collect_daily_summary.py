#!/usr/bin/env python3
import csv
import json
import re
import statistics
import sys
from pathlib import Path


def _read_lines(path: Path):
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def _extract_metric_values(lines, metric_name):
    values = []
    prefix = metric_name
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith(prefix):
            continue
        try:
            v = float(line.split()[-1])
            values.append(v)
        except Exception:
            continue
    return values


def _extract_counter_delta(lines, metric_name):
    vals = _extract_metric_values(lines, metric_name)
    if len(vals) < 2:
        return None
    return vals[-1] - vals[0]


def _stats(vals):
    if not vals:
        return None
    return {
        "min": min(vals),
        "max": max(vals),
        "avg": statistics.fmean(vals),
    }


def summarize(run_dir: Path):
    metrics_lines = _read_lines(run_dir / "vllm_metrics_snapshots.prom")
    gpu_rows = []
    gpu_file = run_dir / "gpu_metrics.csv"
    if gpu_file.exists():
        with gpu_file.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for r in reader:
                gpu_rows.append(r)

    docker_rows = []
    docker_file = run_dir / "docker_stats.csv"
    if docker_file.exists():
        with docker_file.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for r in reader:
                docker_rows.append(r)

    kv_usage = _extract_metric_values(metrics_lines, "vllm:kv_cache_usage_perc")
    running = _extract_metric_values(metrics_lines, "vllm:num_requests_running")
    waiting = _extract_metric_values(metrics_lines, "vllm:num_requests_waiting")

    success_delta = _extract_counter_delta(metrics_lines, "vllm:request_success_total")
    prompt_tokens_delta = _extract_counter_delta(metrics_lines, "vllm:prompt_tokens_total")
    gen_tokens_delta = _extract_counter_delta(metrics_lines, "vllm:generation_tokens_total")

    ttft_sum_delta = _extract_counter_delta(metrics_lines, "vllm:time_to_first_token_seconds_sum")
    ttft_count_delta = _extract_counter_delta(metrics_lines, "vllm:time_to_first_token_seconds_count")
    ttft_avg = None
    if ttft_sum_delta is not None and ttft_count_delta and ttft_count_delta > 0:
        ttft_avg = ttft_sum_delta / ttft_count_delta

    itl_sum_delta = _extract_counter_delta(metrics_lines, "vllm:inter_token_latency_seconds_sum")
    itl_count_delta = _extract_counter_delta(metrics_lines, "vllm:inter_token_latency_seconds_count")
    itl_avg = None
    if itl_sum_delta is not None and itl_count_delta and itl_count_delta > 0:
        itl_avg = itl_sum_delta / itl_count_delta

    gpu_used_by_index = {}
    for row in gpu_rows:
        idx = row.get("index", "unknown")
        used = row.get("memory_used_mib")
        if used is None or used == "":
            continue
        try:
            gpu_used_by_index.setdefault(idx, []).append(float(used))
        except Exception:
            pass

    gpu_summary = {f"gpu{idx}": _stats(vals) for idx, vals in gpu_used_by_index.items()}

    cpu_perc_vals = []
    for row in docker_rows:
        val = row.get("cpu_perc", "").strip().rstrip("%")
        if not val:
            continue
        try:
            cpu_perc_vals.append(float(val))
        except Exception:
            pass

    summary = {
        "run_dir": str(run_dir),
        "requests_success_delta": success_delta,
        "prompt_tokens_delta": prompt_tokens_delta,
        "generation_tokens_delta": gen_tokens_delta,
        "running_requests": _stats(running),
        "waiting_requests": _stats(waiting),
        "kv_cache_usage_pct": _stats([v * 100.0 for v in kv_usage]),
        "ttft_avg_seconds": ttft_avg,
        "inter_token_latency_avg_seconds": itl_avg,
        "gpu_memory_used_mib": gpu_summary,
        "container_cpu_pct": _stats(cpu_perc_vals),
    }
    return summary


def main():
    if len(sys.argv) != 2:
        print("Usage: collect_daily_summary.py <run_dir>", file=sys.stderr)
        sys.exit(1)

    run_dir = Path(sys.argv[1]).resolve()
    summary = summarize(run_dir)

    out_json = run_dir / "daily_summary.json"
    out_txt = run_dir / "daily_summary.txt"

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"run_dir: {summary['run_dir']}",
        f"requests_success_delta: {summary['requests_success_delta']}",
        f"prompt_tokens_delta: {summary['prompt_tokens_delta']}",
        f"generation_tokens_delta: {summary['generation_tokens_delta']}",
        f"running_requests: {summary['running_requests']}",
        f"waiting_requests: {summary['waiting_requests']}",
        f"kv_cache_usage_pct: {summary['kv_cache_usage_pct']}",
        f"ttft_avg_seconds: {summary['ttft_avg_seconds']}",
        f"inter_token_latency_avg_seconds: {summary['inter_token_latency_avg_seconds']}",
        f"gpu_memory_used_mib: {summary['gpu_memory_used_mib']}",
        f"container_cpu_pct: {summary['container_cpu_pct']}",
    ]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(str(out_json))
    print(str(out_txt))


if __name__ == "__main__":
    main()
