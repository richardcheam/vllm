#!/usr/bin/env python3
"""Validate deterministic CPU-KV store/load behavior against a running server.

This client is intentionally independent of the pressure runner. It sends exact
token-length prompts, forces GPU eviction with distinct prompts, and checks that
reloading the original prompts produces identical output. It never starts or
stops a server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import aiohttp


FILLER = (
    " Preserve deterministic facts about CPU KV offload, GPU eviction, and "
    "reload validation while keeping this request stable."
)


class PromptBuilder:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.filler_ids = tokenizer.encode(FILLER, add_special_tokens=False)
        if not self.filler_ids:
            raise ValueError("reload filler produced no tokenizer tokens")

    def build(self, index: int, length: int) -> list[int]:
        prefix = self.tokenizer.encode(
            f"Reload validation prompt {index}. ", add_special_tokens=False
        )
        tokens = list(prefix)
        while len(tokens) < length:
            tokens.extend(self.filler_ids)
        return tokens[:length]


async def request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: list[int],
    request_id: int,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        async with session.post(
            f"{url.rstrip('/')}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "seed": 20260813,
                "stream": False,
            },
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            body = await response.text()
            result: dict[str, Any] = {
                "request_id": request_id,
                "status": response.status,
                "elapsed_seconds": time.monotonic() - started,
            }
            if response.status != 200:
                result["ok"] = False
                result["error"] = body[:240]
                return result
            data = json.loads(body)
            choices = data.get("choices") or []
            usage = data.get("usage") or {}
            if not choices:
                raise ValueError("response contained no choices")
            result.update(
                ok=True,
                text=choices[0].get("text", ""),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )
            return result
    except Exception as exc:
        return {
            "request_id": request_id,
            "ok": False,
            "status": None,
            "elapsed_seconds": time.monotonic() - started,
            "error": str(exc)[:240],
        }


async def wait_ready(session: aiohttp.ClientSession, url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            async with session.get(
                f"{url.rstrip('/')}/health",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        await asyncio.sleep(2)
    raise TimeoutError("server did not become ready")


async def health_check(session: aiohttp.ClientSession, url: str) -> bool:
    try:
        async with session.get(
            f"{url.rstrip('/')}/health",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as response:
            return response.status == 200
    except Exception:
        return False


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(
        args.tokenizer_path or args.model,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=True,
    )
    builder = PromptBuilder(tokenizer)
    store_prompts = [builder.build(i, args.prompt_tokens) for i in range(args.store_requests)]
    evict_prompts = [
        builder.build(10000 + i, args.prompt_tokens)
        for i in range(args.evict_requests)
    ]
    server_healthy_final = False
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(
            limit=max(
                args.store_concurrency,
                args.evict_concurrency,
                args.reload_concurrency,
                1,
            )
            * 2
        )
    ) as session:
        await wait_ready(session, args.url, args.startup_timeout_seconds)

        async def run_batch(
            prompts: list[list[int]], start_id: int, concurrency: int
        ) -> list[dict[str, Any]]:
            semaphore = asyncio.Semaphore(concurrency)

            async def bounded(index: int, prompt: list[int]) -> dict[str, Any]:
                async with semaphore:
                    return await request(
                        session,
                        args.url,
                        args.model,
                        prompt,
                        start_id + index,
                        args.max_tokens,
                        args.request_timeout_seconds,
                    )

            return await asyncio.gather(
                *(bounded(index, prompt) for index, prompt in enumerate(prompts))
            )

        phases: list[dict[str, Any]] = []
        started = time.monotonic()
        store = await run_batch(store_prompts, 0, args.store_concurrency)
        phases.append({"name": "store", "elapsed_seconds": time.monotonic() - started, "results": store})

        for repeat in range(args.repeats):
            started = time.monotonic()
            evict = await run_batch(
                evict_prompts,
                10000 + repeat * args.evict_requests,
                args.evict_concurrency,
            )
            phases.append({"name": f"evict_{repeat + 1}", "elapsed_seconds": time.monotonic() - started, "results": evict})
            started = time.monotonic()
            reload_results = await run_batch(
                store_prompts,
                20000 + repeat * args.store_requests,
                args.reload_concurrency,
            )
            phases.append({"name": f"reload_{repeat + 1}", "elapsed_seconds": time.monotonic() - started, "results": reload_results})
        server_healthy_final = await health_check(session, args.url)

    mismatches = 0
    for phase in phases:
        if not phase["name"].startswith("reload"):
            continue
        mismatches += sum(
            left.get("text") != right.get("text")
            for left, right in zip(store, phase["results"], strict=False)
        )
    failures = [
        result
        for phase in phases
        for result in phase["results"]
        if not result.get("ok", False)
    ]
    prompt_token_mismatches = [
        result
        for phase in phases
        for result in phase["results"]
        if result.get("prompt_tokens") != args.prompt_tokens
    ]
    prompt_lengths = [
        result.get("prompt_tokens")
        for phase in phases
        for result in phase["results"]
        if result.get("prompt_tokens") is not None
    ]
    return {
        "status": "PASS"
        if not failures
        and not mismatches
        and not prompt_token_mismatches
        and server_healthy_final
        else "FAIL",
        "prompt_tokens_requested": args.prompt_tokens,
        "prompt_tokens_min": min(prompt_lengths) if prompt_lengths else None,
        "prompt_tokens_max": max(prompt_lengths) if prompt_lengths else None,
        "store_requests": args.store_requests,
        "evict_requests": args.evict_requests,
        "repeats": args.repeats,
        "transfer_counters_collected": False,
        "transfer_counters_source": (
            "Use run_heavy_offload_soak.py metrics.jsonl for store/load counters."
        ),
        "output_mismatches": mismatches,
        "prompt_token_mismatches": prompt_token_mismatches,
        "server_healthy_final": server_healthy_final,
        "failures": failures,
        "phases": phases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8202")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--tokenizer-mode", default="deepseek_v4")
    parser.add_argument("--prompt-tokens", type=int, default=131072)
    parser.add_argument("--store-requests", type=int, default=4)
    parser.add_argument("--evict-requests", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--store-concurrency", type=int, default=1)
    parser.add_argument("--evict-concurrency", type=int, default=24)
    parser.add_argument("--reload-concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--startup-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "prompt_tokens",
        "store_requests",
        "evict_requests",
        "repeats",
        "store_concurrency",
        "evict_concurrency",
        "reload_concurrency",
        "max_tokens",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    result: dict[str, Any]
    try:
        result = asyncio.run(run(args))
    except Exception as exc:
        result = {"status": "FAIL", "error": str(exc)}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result.get(key) for key in ("status", "output_mismatches", "prompt_tokens_min", "prompt_tokens_max")}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
