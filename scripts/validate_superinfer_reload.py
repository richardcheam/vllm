#!/usr/bin/env python3
"""Validate CPU KV store/load using repeated deterministic prefixes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_prompt import load_tokenizer, tokenize_to_length


class PromptBuilder:
    """Build deterministic prompts with an exact token count."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def build(self, index: int, prompt_len: int) -> list[int]:
        prefix = (
            f"Reload validation fixed prefix {index}. "
            "The answer must be deterministic. "
            "Return a concise numbered list of three facts about KV cache testing."
        )
        return tokenize_to_length(
            prefix,
            prompt_len,
            self.tokenizer,
            "Keep the facts stable across repeated requests.",
        )


def request_once(
    session: requests.Session,
    url: str,
    model: str,
    prompt: list[int],
    max_tokens: int,
) -> dict[str, Any]:
    response = session.post(
        f"{url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 20260813,
            "stream": False,
        },
        timeout=1800,
    )
    body = response.json()
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {body}")
    choices = body.get("choices") or []
    usage = body.get("usage") or {}
    if not choices or usage.get("completion_tokens") is None:
        raise RuntimeError(f"missing choices/usage: {body}")
    return {
        "text": choices[0].get("text", ""),
        "completion_tokens": usage["completion_tokens"],
        "prompt_tokens": usage.get("prompt_tokens"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8202")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-len", type=int, default=131072)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--evict-requests", type=int, default=24)
    parser.add_argument("--evict-concurrency", type=int, default=24)
    parser.add_argument("--reload-repeats", type=int, default=1)
    parser.add_argument("--reload-concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    session = requests.Session()
    session.trust_env = False
    health = session.get(f"{args.url}/health", timeout=30)
    health.raise_for_status()

    tokenizer_path = (
        os.environ.get("TOKENIZER_PATH")
        or os.environ.get("MODEL_PATH")
        or args.model
    )
    prompt_builder = PromptBuilder(
        load_tokenizer(tokenizer_path)
    )
    prompts = [
        prompt_builder.build(i, args.prompt_len) for i in range(args.requests)
    ]
    if args.reload_repeats < 1:
        parser.error("--reload-repeats must be >= 1")
    if args.reload_concurrency < 1:
        parser.error("--reload-concurrency must be >= 1")

    def run_concurrent(phase_prompts: list[list[int]]) -> list[dict[str, Any]]:
        def run_request(prompt: list[int]) -> dict[str, Any]:
            with requests.Session() as worker_session:
                worker_session.trust_env = False
                return request_once(
                    worker_session,
                    args.url,
                    args.model,
                    prompt,
                    args.max_tokens,
                )

        with ThreadPoolExecutor(max_workers=args.reload_concurrency) as executor:
            return list(executor.map(run_request, phase_prompts))

    phases: list[dict[str, Any]] = []
    started = time.monotonic()
    store_results = [
        request_once(session, args.url, args.model, prompt, args.max_tokens)
        for prompt in prompts
    ]
    phases.append(
        {
            "name": "store",
            "elapsed_s": time.monotonic() - started,
            "results": store_results,
        }
    )

    for repeat in range(args.reload_repeats):
        eviction_prompts = [
            prompt_builder.build(
                10000 + repeat * args.evict_requests + i,
                args.prompt_len,
            )
            for i in range(args.evict_requests)
        ]
        phase_suffix = "" if args.reload_repeats == 1 else f"_{repeat + 1}"
        started = time.monotonic()
        eviction_results = run_concurrent(eviction_prompts)
        phases.append(
            {
                "name": f"evict{phase_suffix}",
                "elapsed_s": time.monotonic() - started,
                "results": eviction_results,
            }
        )

        started = time.monotonic()
        reload_results = run_concurrent(prompts)
        phases.append(
            {
                "name": f"reload{phase_suffix}",
                "elapsed_s": time.monotonic() - started,
                "results": reload_results,
            }
        )

    reload_phases = [
        phase for phase in phases if phase["name"].startswith("reload")
    ]
    exact_matches = [
        left["text"] == right["text"]
        for phase in reload_phases
        for left, right in zip(store_results, phase["results"])
    ]
    prompt_token_counts = [
        result["prompt_tokens"]
        for phase in phases
        for result in phase["results"]
        if result.get("prompt_tokens") is not None
    ]
    result = {
        "model": args.model,
        "prompt_len": args.prompt_len,
        "requests": args.requests,
        "reload_repeats": args.reload_repeats,
        "reload_concurrency": args.reload_concurrency,
        "max_tokens": args.max_tokens,
        "prompt_tokens_min": min(prompt_token_counts)
        if prompt_token_counts
        else None,
        "prompt_tokens_max": max(prompt_token_counts)
        if prompt_token_counts
        else None,
        "phases": phases,
        "exact_output_matches": sum(exact_matches),
        "output_mismatches": len(exact_matches) - sum(exact_matches),
        "status": "PASS" if all(exact_matches) else "FAIL",
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "exact_output_matches", "output_mismatches")}))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
