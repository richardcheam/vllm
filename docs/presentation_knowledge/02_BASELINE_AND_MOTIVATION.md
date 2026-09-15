# Baseline And Motivation

## Baseline Serving Shape

```text
DeepSeek-V4-Flash-0731
          ↓
      modern vLLM
          ↓
      TP=2 / PP=1
          ↓
  2x GH200 HBM3e GPUs
          ↓
 realistic concurrent serving
```

The baseline already benefits from continuous batching, paged KV cache,
prefix caching, tensor parallelism, CUDA graphs, and optimized DeepSeek
kernels. The problem is not that baseline vLLM is unusable; it is that the
working set eventually approaches GPU HBM capacity under long contexts and
concurrency.

## Why Throughput Alone Is Insufficient

Historical baseline context from the project knowledge base:

| Concurrency | Aggregate throughput | Approx. throughput/user | TTFT p50 |
|---:|---:|---:|---:|
| 4 | 352.2 tok/s | 88.0 tok/s/user | 1,594 ms |
| 8 | 724.1 tok/s | 90.5 tok/s/user | 1,717 ms |
| 16 | 624.4 tok/s | 39.0 tok/s/user | 13,253 ms |

These are historical baseline examples, not the accepted DS4-0731 pressure
results. They show why an aggregate throughput increase can coexist with a
worse interactive experience.

The presentation should report at least:

- aggregate completion throughput;
- per-user throughput;
- TTFT percentiles;
- TPOT/TBT percentiles;
- end-to-end latency;
- successful/failed requests;
- GPU KV utilization;
- CPU-KV transfer activity;
- queueing and capacity waits.

## Resource Imbalance

The system has approximately 288 GiB of total HBM and approximately 1.2 TiB
of host/Grace memory. GPU HBM is the constrained tier:

```text
HBM:        hot KV, active weights, workspace, active requests
Grace RAM:  larger warm capacity, inactive/swapped KV, future tier
```

The goal is not to lower GPU usage because CPU memory exists. The goal is to
keep latency-sensitive state hot while using CPU memory to extend capacity when
the scheduler would otherwise over-admit, stall, or fail.

## Motivation For SuperInfer-Inspired Work

The key questions are:

1. Can inactive KV blocks move to CPU memory without losing correctness?
2. Can they return to GPU before a request needs them?
3. Can movement overlap useful GPU work?
4. Can the scheduler distinguish normal queueing from true capacity waits?
5. Can topology and NUMA information improve placement decisions?
6. Can the service remain healthy under repeated full GPU-KV pressure?

## What The Baseline Does Not Prove

Baseline vLLM success does not prove:

- CPU-KV stores or loads occur;
- large prompts fit beyond GPU KV capacity;
- stale-capacity waits are avoided;
- CPU movement is quiescent after the last response;
- a topology-aware policy improves performance.

Those require separate observability and pressure/reload experiments.

## Sources

- `SUPERINFER_KNOWLEDGE_BASE.md:95-146`.
- `SUPERINFER_KNOWLEDGE_BASE.md:60-91`.
- `docs/100_reasoning_tree_offload_results.md:122-174`.
- `docs/080_gpu_validation_checklist.md`.
