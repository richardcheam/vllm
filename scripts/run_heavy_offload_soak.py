#!/usr/bin/env python3
"""Exercise long-context GPU KV pressure against a staging vLLM server.

This runner does not start, stop, or reconfigure a server. It only sends
requests to ``--url`` and records numeric Prometheus telemetry. Use a staging
port and a server started from the reasoning tree.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

import aiohttp


DEFAULT_FILLER = (
    "Discuss capacity planning, failure recovery, measurable acceptance "
    "criteria, and operational safeguards for this test workload."
)


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse numeric samples, retaining aggregate and labeled keys."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sample, separator, raw_value = line.rpartition(" ")
        if not separator:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        name = sample.split("{", 1)[0]
        values[name] = values.get(name, 0.0) + value
        if "{" in sample:
            values[sample] = value
    return values


def metric(metrics: dict[str, float], name: str, default: float = 0.0) -> float:
    """Read either colon-style or underscore-style Prometheus names."""
    candidates = (
        name,
        f"{name}_total",
        name.replace(":", "_"),
        f"{name.replace(':', '_')}_total",
    )
    for candidate in candidates:
        if candidate in metrics:
            return metrics[candidate]
    return default


def labeled_metric(
    metrics: dict[str, float], name: str, label: str, value: str
) -> float:
    """Sum samples whose Prometheus label matches exactly."""
    prefix = f"{name}{{"
    return sum(
        sample_value
        for sample_name, sample_value in metrics.items()
        if sample_name.startswith(prefix) and f'{label}="{value}"' in sample_name
    )


class PromptBuilder:
    def __init__(self, tokenizer: Any, seed: int) -> None:
        self.tokenizer = tokenizer
        self.seed = seed
        self.filler_ids = tokenizer.encode(
            " " + DEFAULT_FILLER,
            add_special_tokens=False,
        )
        if not self.filler_ids:
            raise ValueError("The filler text produced no tokenizer tokens")

    def build(self, request_id: int, prompt_len: int, unique: bool = True) -> list[int]:
        if prompt_len < 1:
            raise ValueError("prompt length must be positive")
        prefix = (
            f"Staging offload request {request_id} seed {self.seed}. "
            if unique
            else "Staging offload shared prefix. "
        )
        token_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        while len(token_ids) < prompt_len:
            token_ids.extend(self.filler_ids)
        return token_ids[:prompt_len]


async def fetch_metrics(session: aiohttp.ClientSession, url: str) -> dict[str, float]:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status != 200:
                return {}
            return parse_prometheus(await response.text())
    except Exception:
        return {}


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: list[int],
    max_tokens: int,
    request_id: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": request_id,
        "stream": False,
    }
    started = time.monotonic()
    try:
        async with session.post(
            f"{url}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_seconds),
        ) as response:
            body = await response.text()
            elapsed = time.monotonic() - started
            if response.status != 200:
                return {
                    "request_id": request_id,
                    "ok": False,
                    "status": response.status,
                    "elapsed_seconds": elapsed,
                    "error": body[:240],
                }
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                data = {}
            usage = data.get("usage") or {}
            return {
                "request_id": request_id,
                "ok": True,
                "status": response.status,
                "elapsed_seconds": elapsed,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
    except Exception as exc:
        return {
            "request_id": request_id,
            "ok": False,
            "status": None,
            "elapsed_seconds": time.monotonic() - started,
            "error": str(exc)[:240],
        }


async def sample_metrics(
    session: aiohttp.ClientSession,
    metrics_url: str,
    output_path: Path,
    interval_seconds: float,
    samples: list[dict[str, Any]],
    stop: asyncio.Event,
) -> None:
    with output_path.open("w", encoding="utf-8") as output:
        while not stop.is_set():
            values = await fetch_metrics(session, metrics_url)
            sample = {
                "timestamp": time.time(),
                "metrics": values,
            }
            samples.append(sample)
            output.write(json.dumps(sample) + "\n")
            output.flush()
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
            except asyncio.TimeoutError:
                pass


async def wait_ready(session: aiohttp.ClientSession, url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            async with session.get(
                f"{url}/health", timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        await asyncio.sleep(2)
    raise RuntimeError(f"Server did not become ready: {url}/health")


async def health_check(session: aiohttp.ClientSession, url: str) -> bool:
    try:
        async with session.get(
            f"{url.rstrip('/')}/health",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as response:
            return response.status == 200
    except Exception:
        return False


async def wait_for_capacity(
    session: aiohttp.ClientSession, metrics_url: str, timeout: float
) -> dict[str, float]:
    """Wait until the server exposes the real usable GPU KV block count."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        values = await fetch_metrics(session, metrics_url)
        if metric(values, "vllm:kv_cache_total_blocks") > 0:
            return values
        await asyncio.sleep(2)
    raise RuntimeError(
        "The server did not expose vllm:kv_cache_total_blocks; "
        "refusing to infer a long-context capacity limit"
    )


async def run_stage(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    builder: PromptBuilder,
    stage_name: str,
    users: int,
    requests: int,
    prompt_len: int,
    request_id_start: int,
) -> list[dict[str, Any]]:
    print(
        f"[stage] {stage_name}: users={users} requests={requests} "
        f"prompt_tokens={prompt_len} max_tokens={args.max_tokens}"
    )
    semaphore = asyncio.Semaphore(users)

    async def bounded_request(request_id: int) -> dict[str, Any]:
        async with semaphore:
            prompt = builder.build(request_id, prompt_len, unique=True)
            result = await send_request(
                session,
                args.url,
                args.model,
                prompt,
                args.max_tokens,
                request_id,
                args.request_timeout_seconds,
            )
            result["requested_prompt_tokens"] = prompt_len
            return result

    results = await asyncio.wait_for(
        asyncio.gather(
            *(bounded_request(request_id_start + index) for index in range(requests))
        ),
        timeout=args.stage_timeout_seconds,
    )
    successful = sum(result["ok"] for result in results)
    print(f"[stage] {stage_name}: successful={successful}/{len(results)}")
    return results


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}
    first = samples[0]["metrics"]
    last = samples[-1]["metrics"]
    counter_names = (
        "offload_store_events",
        "offload_load_events",
        "offload_store_bytes",
        "offload_load_bytes",
        "local_swap_out_bytes",
        "local_swap_in_bytes",
        "remote_swap_out_bytes",
        "remote_swap_in_bytes",
    )
    deltas = {
        name: metric(last, f"vllm:{name}") - metric(first, f"vllm:{name}")
        for name in counter_names
    }
    free_values = [metric(sample["metrics"], "vllm:kv_cache_free_blocks") for sample in samples]
    usage_values = [metric(sample["metrics"], "vllm:kv_cache_usage_perc") for sample in samples]
    waiting_values = [metric(sample["metrics"], "vllm:num_requests_waiting") for sample in samples]
    running_values = [metric(sample["metrics"], "vllm:num_requests_running") for sample in samples]
    stale_seconds = 0.0
    current_stale_start: float | None = None
    for sample, waiting, running in zip(
        samples, waiting_values, running_values, strict=False
    ):
        capacity_waiting = (
            labeled_metric(
                sample["metrics"],
                "vllm:num_requests_waiting_by_reason",
                "reason",
                "capacity",
            )
            > 0
            and running <= 0
        )
        if capacity_waiting:
            if current_stale_start is None:
                current_stale_start = sample["timestamp"]
            stale_seconds = max(stale_seconds, sample["timestamp"] - current_stale_start)
        else:
            current_stale_start = None
    return {
        "sample_count": len(samples),
        "kv_total_blocks": metric(last, "vllm:kv_cache_total_blocks"),
        "kv_free_blocks_min": min(free_values),
        "kv_free_blocks_final": free_values[-1],
        "kv_usage_max": max(usage_values),
        "waiting_max": max(waiting_values),
        "running_max": max(running_values),
        "max_stale_capacity_seconds": stale_seconds,
        "counter_deltas": deltas,
    }


async def async_main(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    expected_pressure_prompt_tokens = (
        args.expected_pressure_prompt_tokens
        if args.expected_pressure_prompt_tokens is not None
        else args.pressure_prompt_tokens
        if args.pressure_prompt_tokens is not None
        else 131072
    )
    metrics_url = f"{args.url.rstrip('/')}/metrics"
    snapshots: list[dict[str, Any]] = []
    stop_sampling = asyncio.Event()
    all_results: list[dict[str, Any]] = []
    expected_error_ids: set[int] = set()
    server_healthy_final = False

    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(
        args.tokenizer_path or args.model,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=True,
    )
    builder = PromptBuilder(tokenizer, args.seed)

    connector = aiohttp.TCPConnector(limit=max(args.users, 1) * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        await wait_ready(session, args.url.rstrip("/"), args.startup_timeout_seconds)
        # Scheduler capacity gauges are first populated when the engine emits
        # scheduler stats. Send one tiny request to bootstrap that update before
        # attempting to discover the real KV pool size.
        bootstrap_result = await send_request(
            session,
            args.url.rstrip("/"),
            args.model,
            builder.build(-1, args.block_size),
            1,
            -1,
            args.request_timeout_seconds,
        )
        if not bootstrap_result["ok"]:
            raise RuntimeError(f"Bootstrap request failed: {bootstrap_result}")
        initial_metrics = await wait_for_capacity(
            session, metrics_url, args.startup_timeout_seconds
        )
        sampler_task = asyncio.create_task(
            sample_metrics(
                session,
                metrics_url,
                args.output_dir / "metrics.jsonl",
                args.metrics_interval_seconds,
                snapshots,
                stop_sampling,
            )
        )
        request_id = 0
        try:
            total_blocks = int(metric(initial_metrics, "vllm:kv_cache_total_blocks"))
            block_size = args.block_size
            if total_blocks < 1:
                raise RuntimeError("Could not determine a usable KV pool capacity")
            print(
                f"[capacity] usable_gpu_blocks={total_blocks}; "
                "logical token capacity is not inferred from raw blocks"
            )

            if args.pressure_prompt_tokens is not None:
                pressure_prompt = args.pressure_prompt_tokens
            elif args.pressure_request_fraction is not None:
                pressure_prompt = max(
                    block_size,
                    int(total_blocks * block_size * args.pressure_request_fraction),
                )
                print(
                    "[capacity] using legacy raw-block pressure sizing because "
                    "--pressure-request-fraction was explicitly supplied"
                )
            else:
                pressure_prompt = 131072
            if args.warmup_prompt_tokens is not None:
                warmup_prompt = args.warmup_prompt_tokens
            elif args.warmup_pool_fraction is not None:
                warmup_prompt = max(
                    block_size,
                    int(total_blocks * block_size * args.warmup_pool_fraction),
                )
            else:
                warmup_prompt = 4096
            if pressure_prompt > args.max_prompt_tokens:
                raise RuntimeError(
                    f"Pressure prompt requires {pressure_prompt} tokens, above "
                    f"--max-prompt-tokens={args.max_prompt_tokens}"
                )
            if pressure_prompt != expected_pressure_prompt_tokens:
                raise RuntimeError(
                    "Pressure prompt length must match "
                    f"--expected-pressure-prompt-tokens={expected_pressure_prompt_tokens}"
                )
            warmup_results = await run_stage(
                session, args, builder, "warmup", args.users, args.warmup_requests,
                min(warmup_prompt, args.max_prompt_tokens), request_id,
            )
            all_results.extend(warmup_results)
            request_id += args.warmup_requests

            pressure_started = time.monotonic()
            round_index = 0
            while (
                round_index < args.pressure_rounds
                or (
                    args.soak_seconds > 0
                    and time.monotonic() - pressure_started < args.soak_seconds
                )
            ):
                results = await run_stage(
                    session,
                    args,
                    builder,
                    f"aggregate_pressure_{round_index + 1}",
                    args.users,
                    args.pressure_requests,
                    pressure_prompt,
                    request_id,
                )
                all_results.extend(results)
                request_id += args.pressure_requests
                round_index += 1

            if args.impossible_prompt_tokens is not None:
                impossible_prompt = args.impossible_prompt_tokens
                if impossible_prompt <= args.max_prompt_tokens:
                    expected_error_ids.add(request_id)
                    results = await run_stage(
                        session, args, builder, "impossible_request", 1, 1,
                        impossible_prompt, request_id,
                    )
                    all_results.extend(results)
                else:
                    print("[stage] impossible_request: skipped by prompt/model limit")
            server_healthy_final = await health_check(session, args.url)
        finally:
            stop_sampling.set()
            await sampler_task

    summary = summarize_samples(snapshots)
    successful = sum(result["ok"] for result in all_results)
    failed = len(all_results) - successful
    counters = summary.get("counter_deltas", {})
    transfers_observed = any(
        counters.get(name, 0) > 0
        for name in (
            "offload_store_events",
            "offload_load_events",
            "local_swap_out_bytes",
            "local_swap_in_bytes",
        )
    )
    pool_known = summary.get("kv_total_blocks", 0) > 0
    stale_ok = summary.get("max_stale_capacity_seconds", math.inf) <= args.max_stale_seconds
    pending_final = metric(
        snapshots[-1]["metrics"] if snapshots else {},
        "vllm:offload_pending_store_events",
    ) + metric(
        snapshots[-1]["metrics"] if snapshots else {},
        "vllm:offload_pending_load_reqs",
    )
    pending_age_final = metric(
        snapshots[-1]["metrics"] if snapshots else {},
        "vllm:offload_pending_transfer_age_ms",
    )
    pending_ok = pending_final <= args.max_pending_transfers
    unexpected_failures = [
        result
        for result in all_results
        if not result["ok"] and result["request_id"] not in expected_error_ids
    ]
    prompt_token_mismatches = [
        result
        for result in all_results
        if result.get("requested_prompt_tokens") == expected_pressure_prompt_tokens
        and result.get("prompt_tokens") != expected_pressure_prompt_tokens
    ]
    no_request_loss = not unexpected_failures or args.allow_request_errors
    verdict = {
        "pass": (
            pool_known
            and stale_ok
            and pending_ok
            and no_request_loss
            and not prompt_token_mismatches
            and server_healthy_final
            and (args.allow_no_transfers or transfers_observed)
        ),
        "pool_known": pool_known,
        "transfers_observed": transfers_observed,
        "long_context_offload_proven": (
            pool_known
            and transfers_observed
            and stale_ok
            and pending_ok
            and no_request_loss
            and not prompt_token_mismatches
            and server_healthy_final
        ),
        "successful_requests": successful,
        "failed_requests": failed,
        "unexpected_failures": unexpected_failures,
        "prompt_token_mismatches": prompt_token_mismatches,
        "expected_error_ids": sorted(expected_error_ids),
        "pending_transfers_final": pending_final,
        "pending_transfer_age_ms_final": pending_age_final,
        "server_healthy_final": server_healthy_final,
        "workload": {
            "url": args.url,
            "model": args.model,
            "tokenizer_path": args.tokenizer_path or args.model,
            "users": args.users,
            "pressure_requests": args.pressure_requests,
            "pressure_rounds": args.pressure_rounds,
            "pressure_prompt_tokens": pressure_prompt,
            "expected_pressure_prompt_tokens": expected_pressure_prompt_tokens,
            "warmup_requests": args.warmup_requests,
            "block_size": args.block_size,
            "max_tokens": args.max_tokens,
            "request_timeout_seconds": args.request_timeout_seconds,
            "stage_timeout_seconds": args.stage_timeout_seconds,
        },
        "started_at": started_at,
        "finished_at": time.time(),
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2))
    if not verdict["pass"]:
        return 1
    if not transfers_observed:
        print("[warning] CPU KV transfer activity was not observed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8203")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--tokenizer-mode", default="deepseek_v4")
    parser.add_argument("--output-dir", type=Path, default=Path("heavy_offload_artifacts"))
    parser.add_argument("--users", type=int, default=24)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--pressure-requests", type=int, default=24)
    parser.add_argument("--pressure-rounds", type=int, default=1)
    parser.add_argument(
        "--soak-seconds",
        type=float,
        default=0.0,
        help="Continue pressure rounds for this duration after the initial rounds.",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument(
        "--pressure-prompt-tokens",
        type=int,
        default=327680,
        help="Exact pressure prompt length; do not derive it from raw KV blocks.",
    )
    parser.add_argument("--warmup-prompt-tokens", type=int, default=None)
    parser.add_argument(
        "--pressure-request-fraction",
        type=float,
        default=None,
        help="Deprecated compatibility option; ignored when prompt tokens are explicit.",
    )
    parser.add_argument(
        "--warmup-pool-fraction",
        type=float,
        default=None,
        help="Deprecated compatibility option; ignored when prompt tokens are explicit.",
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=327680)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--metrics-interval-seconds", type=float, default=2.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--stage-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--expected-pressure-prompt-tokens", type=int, default=327680)
    parser.add_argument("--max-stale-seconds", type=float, default=30.0)
    parser.add_argument("--max-pending-transfers", type=float, default=0.0)
    parser.add_argument(
        "--impossible-prompt-tokens",
        type=int,
        default=None,
        help="Explicit prompt length for an expected capacity rejection.",
    )
    parser.add_argument(
        "--run-impossible",
        action="store_true",
        help="Deprecated; use --impossible-prompt-tokens with an explicit length.",
    )
    parser.add_argument("--allow-request-errors", action="store_true")
    parser.add_argument(
        "--allow-no-transfers",
        action="store_true",
        help="Allow a GPU-only control run; otherwise transfer activity is required.",
    )
    args = parser.parse_args()
    if args.users < 1 or args.pressure_rounds < 1:
        parser.error("--users and --pressure-rounds must be positive")
    if args.pressure_prompt_tokens is not None and args.pressure_prompt_tokens < 1:
        parser.error("--pressure-prompt-tokens must be positive")
    if args.warmup_prompt_tokens is not None and args.warmup_prompt_tokens < 1:
        parser.error("--warmup-prompt-tokens must be positive")
    if (
        args.expected_pressure_prompt_tokens is not None
        and args.expected_pressure_prompt_tokens < 1
    ):
        parser.error("--expected-pressure-prompt-tokens must be positive")
    if args.run_impossible and args.impossible_prompt_tokens is None:
        parser.error("--run-impossible requires --impossible-prompt-tokens")
    if args.impossible_prompt_tokens is not None and args.impossible_prompt_tokens < 1:
        parser.error("--impossible-prompt-tokens must be positive")
    try:
        return asyncio.run(async_main(args))
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {"pass": False, "error": str(exc)}
        (args.output_dir / "summary.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        print(json.dumps(failure, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
