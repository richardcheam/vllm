# Project Overview

## Objective

Serve `DeepSeek-V4-Flash-0731` efficiently for long-context, multi-user
workloads on a dual-GH200 system. The objective is broader than “make the model
start”: the system must preserve request correctness under GPU-KV pressure,
move KV state between GPU HBM and Grace/CPU memory, and remain operable as a
production-style service.

## Hardware

| Item | Verified value |
|---|---|
| GPUs | 2x NVIDIA GH200 144 GB HBM3e |
| GPU memory | 144 GiB per GPU, approximately 288 GiB total HBM |
| GPU interconnect | NVLink `NV18` in the newer topology audit |
| GPU NUMA mapping | GPU 0 -> NUMA node 0; GPU 1 -> NUMA node 1 |
| Host memory | Approximately 1.2 TiB observed |
| Tensor parallelism | TP=2 |
| Pipeline parallelism | PP=1 |
| Expert parallelism | Enabled in validated DS4 recipe |
| Runtime container | `richard-base-dev-sysnice` |

## Model And Runtime

- Model: `DeepSeek-V4-Flash-0731` local snapshot.
- Architecture: `DeepseekV4ForCausalLM`.
- Model computation dtype: `bfloat16`.
- Model quantization path: `deepseek_v4_fp8`.
- GPU KV dtype: FP8.
- KV block size: 256 tokens.
- Maximum model length: 1,048,576 tokens.
- Prefix caching: enabled.
- Chunked prefill: enabled.
- CUDA graph mode: `FULL_AND_PIECEWISE`.
- Speculative decoding in the validated reasoning tree: disabled.
- Speculative decoding in the newer v0.26 integration: DSpark, initially five tokens.

## Contribution Summary

The main contribution is a capability-oriented forward-port rather than a
line-by-line copy of an old SuperInfer implementation. The modern reasoning
tree adds and hardens:

- a SimpleCPUOffloadConnector-based CPU-KV tier;
- D2H store and H2D reload paths;
- bounded asynchronous transfer state;
- scheduler-visible capacity and transfer telemetry;
- proactive movement under guarded pressure;
- allocator/free-list transaction safety;
- exact-token pressure and reload harnesses;
- process-local shutdown cleanup;
- production-oriented startup, shutdown, diagnostics, and user-service design.

The newer v0.26 integration expands the research direction with DSpark,
empty CPU allocation, native DMA experiments, transfer-quiescence gates, and
high-risk rotation. It is not interchangeable with the validated production
track and is not yet a final parity result.

## Central Thesis

```text
GPU HBM is the hot working tier
        |
        | asynchronous KV movement
        v
Grace/CPU memory is the warm capacity tier
        |
        v
The scheduler must reason about capacity, residency, movement, and correctness
```

## Safe Presentation Claim

> The project demonstrates that SuperInfer-inspired CPU-KV capacity extension
> can be integrated into modern vLLM V1 and exercised under exact-token GPU-KV
> pressure on dual GH200 hardware. It does not yet establish complete original
> SuperInfer parity or a universal throughput improvement.

## Sources

- `docs/100_reasoning_tree_offload_results.md`: primary accepted configuration/results.
- `docs/090_launch_recipe.md`: argument-level serving recipe.
- `docs/110_systemd_service_guide.md`: lifecycle design.
- `docs/000_environment_inventory.md`: checkout and environment scope.
- `/home/az04297/SUPERINFER_KNOWLEDGE_BASE.md:11-91,214-251`: project motivation and hierarchy.
- `/home/az04297/re-SuperInfer/vllm-superinfer-v4-integration/docs/superinfer/README.md:3-19`: newer integration target.
