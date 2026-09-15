# Metric Definitions

## Throughput

### Aggregate Completion Throughput

```text
generated output tokens / benchmark wall-clock time
```

It measures aggregate decoding work. It does not guarantee good per-user
latency.

### Per-User Throughput

```text
aggregate completion throughput / active concurrency
```

It is a rough interactive-service indicator.

## Latency

### TTFT

Time from request arrival to first generated token. It includes queueing,
prefill, scheduler delay, and memory restoration before first token.

### TPOT/TBT

Average time between generated tokens after the first token. Lower is better for
interactive decode. Approximate relationship:

```text
```

## Request Success

```text
successful requests / submitted requests
```

An incomplete benchmark is not better because its successful subset is fast.

## GPU And KV Metrics

- `gpu_memory_used`: resident GPU allocation including model, graphs, allocator,
  workspace, and active buffers.
- `kv_cache_usage_perc`: logical active GPU KV usage, not `nvidia-smi` usage.
- `kv_cache_free_blocks`: current free GPU KV blocks.
- `kv_cache_used_blocks`: current active GPU KV blocks.
- `kv_cache_total_blocks`: usable GPU KV blocks excluding the null block.

## CPU-KV Movement

- `offload_store_events`: completed GPU-to-CPU store events.
- `offload_load_events`: completed CPU-to-GPU load events.
- `offload_store_bytes`: GPU -> CPU bytes.
- `offload_load_bytes`: CPU -> GPU bytes.
- `offload_pending_transfer_age_ms`: age of oldest pending transfer state.
- `offload_cpu_total_blocks`: CPU-KV capacity in blocks.
- `offload_cpu_used_blocks`: CPU-KV blocks currently occupied.

Positive store/load counts prove movement. They do not prove overlap or a
performance benefit.

## Capacity And Queueing

- `reason="queue"`: ordinary scheduler queueing.
- `reason="capacity"`: full-sequence admission waiting for KV capacity.
- `reason="deferred"`: transient deferral such as transfer or budget constraint.

## Locality Metrics

- topology discovered/fallback gauges;
- local/remote swap byte counters;
- remote fallback count.

These are logical planning metrics and do not prove physical page placement.

## Reliability Metrics

- final health;
- EngineCore death/error markers;
- worker exit status/signal;
- pending transfer age;
- queue depth;
- container restart/OOM state;
- `/dev/shm` inventory;
- server supervisor exit code/signal.

## Overlap Gap

The key missing distinction is:

```text
total DMA time
        versus
DMA time visible on the request critical path
```

An eventual overlap metric could be:

```text
hidden transfer time / total transfer time
```

## Sources

- `docs/100_reasoning_tree_offload_results.md:252-290`.
- `docs/heavy_offload_staging.md:132-146`.
- `vllm/v1/metrics/loggers.py`.
- `SUPERINFER_KNOWLEDGE_BASE.md:574-802`.
