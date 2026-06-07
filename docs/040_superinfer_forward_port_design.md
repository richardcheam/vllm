# SuperInfer Forward-Port Design

## Goals

- Enable optional GH200 CPU DRAM KV-cache swap tier.
- Preserve DeepSeek-V4 correctness.
- Improve TTFT/TBT only after correctness is proven.
- Keep vanilla vLLM behavior unchanged when disabled.

## Feature Flags

Initial no-op flags:

```bash
--swap-cpu-memory-gb
--proactive-swap-budget
--swapper-block-first
--pin-memory-fix
--prefix-cache-fix
--vlt-alpha
--vlt-beta-bandwidth
--vlt-beta-future
--slo-ttft
--slo-tbt
```

These are parsed and stored in config first. No scheduler, cache, executor, or model behavior changes in the first layer.

## Components

| Component | Modern design |
|---|---|
| RotaSched policy | Start as a pure scoring helper and telemetry. Later integrate into `vllm/v1/core/sched/scheduler.py`. |
| VLT scoring | Pure function/class with request timing inputs and swap-cost estimates. Unit-test independently. |
| Request rotary state | Metadata-only enum initially, separate from `RequestStatus` until scheduler integration is safe. |
| CPU swap allocator | Prefer modern `kv_offload`/`simple_kv_offload` allocator interfaces. Add GH200-specific allocator only if necessary. |
| Block metadata states | Track outside core block state first: `GPU_ONLY`, `CPU_ONLY`, `GPU_CPU_SYNCED`, `GPU_DIRTY`, `H2D_IN_FLIGHT`, `D2H_IN_FLIGHT`, `PINNED_PREFIX`, `UNSWAPPABLE`. |
| Transfer engine | Initially use existing offload copy paths or a debug copy backend; native batched DuplexKV comes later. |
| Telemetry | Add counters before behavior: active/waiting/running/swapped, KV usage, swap bytes/time, TTFT/TBT observations if available. |
| Failure fallback | All behavior gated by config. On uncertainty, recompute or keep request GPU-resident. |

## Block State Model

- `GPU_ONLY`: block exists only in GPU cache.
- `CPU_ONLY`: block exists only in CPU swap tier; request cannot execute until restored.
- `GPU_CPU_SYNCED`: GPU and CPU copies match.
- `GPU_DIRTY`: GPU copy is newer than CPU copy.
- `H2D_IN_FLIGHT`: CPU to GPU copy is outstanding.
- `D2H_IN_FLIGHT`: GPU to CPU copy is outstanding.
- `PINNED_PREFIX`: prefix-cache/shared block; initially unswappable.
- `UNSWAPPABLE`: block/request uses unsupported state such as MTP-active or unpaired indexer metadata.

## First Safe Scope

- DeepSeek-V4-Flash with vanilla modern vLLM support preserved.
- No-op flags first.
- Correctness tests before performance tests.
- MTP disabled for first manual swap tests.
- Prefix cache disabled or prefix-shared blocks pinned.
- One request swapped out and restored in a debug-only path.
- No block-first layout or full-duplex optimization until single-request correctness is proven.
