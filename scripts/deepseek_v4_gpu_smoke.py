#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from collections import defaultdict
from typing import Any

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek-V4 GPU smoke test")
    parser.add_argument("--model", required=True)
    parser.add_argument("--swap-cpu-memory-gb", type=float, default=None)
    parser.add_argument("--proactive-swap-budget", type=int, default=0)
    parser.add_argument("--vlt-beta-bandwidth", type=float, default=0.0)
    parser.add_argument("--num-gpu-blocks-override", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--prompt-count", type=int, default=1)
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--pin-memory-fix", action="store_true")
    parser.add_argument("--swapper-block-first", action="store_true")
    return parser.parse_args()


def _accumulate_numeric_dict(target: dict[str, float], data: Any) -> None:
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        if isinstance(value, bool):
            target[key] += int(value)
        elif isinstance(value, int | float):
            target[key] += value


def main() -> None:
    args = parse_args()
    prompt_base = (
        "DeepSeek-V4 SuperInfer pressure validation. "
        "Keep this deterministic and continue the technical explanation. "
    )
    prompts = [f"case {idx}: " + prompt_base * args.prompt_repeat for idx in range(args.prompt_count)]

    llm_kwargs = {
        "model": args.model,
        "tokenizer": args.model,
        "dtype": "bfloat16",
        "seed": 123,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "enforce_eager": True,
        "trust_remote_code": False,
        "disable_log_stats": False,
    }
    if args.swap_cpu_memory_gb is not None:
        llm_kwargs["swap_cpu_memory_gb"] = args.swap_cpu_memory_gb
    if args.proactive_swap_budget > 0:
        llm_kwargs["proactive_swap_budget"] = args.proactive_swap_budget
    if args.vlt_beta_bandwidth > 0:
        llm_kwargs["vlt_beta_bandwidth"] = args.vlt_beta_bandwidth
    if args.num_gpu_blocks_override is not None:
        llm_kwargs["num_gpu_blocks_override"] = args.num_gpu_blocks_override
    if args.pin_memory_fix:
        llm_kwargs["pin_memory_fix"] = True
    if args.swapper_block_first:
        llm_kwargs["swapper_block_first"] = True

    llm = LLM(**llm_kwargs)
    rotary_stats = defaultdict(float)
    connector_stats = defaultdict(float)
    original_get_output = llm.llm_engine.engine_core.get_output

    def get_output_with_stats():
        output = original_get_output()
        stats = output.scheduler_stats
        if stats is not None:
            rotary_stats["num_rotary_preempted_reqs"] += stats.num_rotary_preempted_reqs
            rotary_stats["num_rotary_synced_blocks"] += stats.num_rotary_synced_blocks
            rotary_stats["num_rotary_unsynced_blocks"] += stats.num_rotary_unsynced_blocks
            rotary_stats["num_rotary_dirty_tail_tokens"] += (
                stats.num_rotary_dirty_tail_tokens
            )
            _accumulate_numeric_dict(connector_stats, stats.kv_connector_stats)
        return output

    llm.llm_engine.engine_core.get_output = get_output_with_stats
    for pass_idx in range(args.passes):
        outputs = llm.generate(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        )
        for output in outputs:
            completion = output.outputs[0]
            print(f"PASS={pass_idx}")
            print(f"PROMPT={output.prompt!r}")
            print(f"TOKENS={list(completion.token_ids)}")
            print(f"TEXT={completion.text!r}")
    print("ROTARY_STATS=" + json.dumps(dict(sorted(rotary_stats.items()))))
    print("CONNECTOR_STATS=" + json.dumps(dict(sorted(connector_stats.items()))))


if __name__ == "__main__":
    main()
