#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse

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
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--kv-cache-dtype", default="fp8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = [
        f"Write one short sentence about GPU validation case {idx}."
        for idx in range(args.prompt_count)
    ]

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
    }
    if args.swap_cpu_memory_gb is not None:
        llm_kwargs["swap_cpu_memory_gb"] = args.swap_cpu_memory_gb
    if args.proactive_swap_budget > 0:
        llm_kwargs["proactive_swap_budget"] = args.proactive_swap_budget
    if args.vlt_beta_bandwidth > 0:
        llm_kwargs["vlt_beta_bandwidth"] = args.vlt_beta_bandwidth
    if args.num_gpu_blocks_override is not None:
        llm_kwargs["num_gpu_blocks_override"] = args.num_gpu_blocks_override

    llm = LLM(**llm_kwargs)
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )
    for output in outputs:
        completion = output.outputs[0]
        print(f"PROMPT={output.prompt!r}")
        print(f"TOKENS={list(completion.token_ids)}")
        print(f"TEXT={completion.text!r}")


if __name__ == "__main__":
    main()
