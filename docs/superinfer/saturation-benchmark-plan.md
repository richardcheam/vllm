# Saturation Benchmark Plan

## Objective

Find the maximum stable performance of DeepSeek-V4-Flash-0731 with DSpark when
the GPU KV cache is continuously pressured and SuperInfer must rotate KV state
through GH200 CPU memory.

## Required comparison profiles

1. Vanilla vLLM 0.26.0 with DSpark and no KV offload.
2. vLLM 0.26.0 native KV offload without topology-aware policy.
3. SuperInfer transfer path with topology policy disabled.
4. SuperInfer topology-aware transfer with conservative rotation.
5. SuperInfer topology-aware transfer with the final aggressive saturation
   configuration.
6. Experimental SuperInfer high-risk mode with aggressive DSpark rotation.

Every comparison must use the same model revision, TP/PP layout, prompts,
sampling parameters, and CUDA graph settings.

The high-risk profile is reported separately. A throughput win is not accepted
as a production result unless its output equivalence, rejection/rollback,
restore, cancellation, and repeated-preemption checks also pass.

## Initial deployment

```text
GPUs:                 0,1
Tensor parallel:      2
Pipeline parallel:    1
Model max length:     1,048,576
DSpark tokens:        5 initially
Prefix caching:       disabled for primary pressure tests
Prompt identity:      unique per request
Output lengths:       128 and 256
```

Prefix caching must remain enabled for the native `SimpleCPUOffloadConnector`
path. The primary pressure workload therefore uses unique prompts to prevent
prefix hits from hiding KV usage. A repeated-prefix workload is tested
separately because it exercises a different residency model.

## Pressure ladder

Sweep the following dimensions until the workload reaches a stable plateau:

- Prompt lengths: 4K, 16K, 64K, 128K, 256K where feasible.
- Concurrent users: 2, 4, 8, 16, 32 and higher when stable.
- Requests: at least three repeated trials per point.
- GPU memory utilization: increase only while startup and execution remain
  stable; retain headroom for activations, graphs, and transfer buffers.
- CPU KV capacity: enough to observe local residency, then enough to test
  capacity exhaustion and remote fallback.
- DSpark speculative tokens: 5 first, then larger valid values.

The saturation point is the highest-throughput stable region with active H2D and
D2H transfers, no growing queue, no OOM, and no output divergence.

`PROACTIVE_SWAP_BUDGET` limits rotation work per step. It is not interpreted as
an unconditional number of GPU blocks to keep free; doing so can add overhead
on GPU-fitting workloads and would conflate the paper's transfer budget with
HBM admission headroom.

## Warmup

The benchmark runner performs an exhaustive six-stage warmup by default before
recording measurement repeats:

1. 4K prompt, 4 users, 8 requests.
2. 16K prompt, 8 users, 32 requests.
3. 64K prompt, 16 users, 64 requests.
4. 128K prompt, 24 users, 96 requests.
5. 256K prompt, 32 users, 128 requests.
6. 512K prompt, 32 users, 128 requests.

The final stage is followed by a saturation soak using the exact configured
benchmark workload. Warmup uses `test_concurrent_robust.py` with unique request
prefixes by default, preventing prefix caching from hiding KV pressure.

Warmup pays one-time costs such as CUDA graph capture, kernel autotuning,
allocator initialization, DSpark setup, and transfer-worker initialization. It
can improve first-request latency and make benchmark results more repeatable.
It does not inherently increase steady-state saturated throughput. The
exhaustive ladder can itself activate CPU KV residency, transfers, and
preemption, which is intentional for this workload. Warmup must be identical
for vanilla, native-offload, and SuperInfer profiles.

The default 512K stage is intentionally aggressive. Set the comma-separated
`WARMUP_*` arrays in `.env` to include 1M-token stages if the host has enough
time and CPU KV capacity; do not assume configured `max_model_len` alone proves
that a 1M pressure run is stable.

Warmup outputs are saved beside measurement outputs so startup effects are not
mistaken for serving performance. Measurement also uses the robust unique-prefix
client by default; this keeps the workload matched to warmup and prevents the
basic client's repeated topic strings from hiding KV pressure. Set
`BENCH_CLIENT=basic` only for an intentionally different workload.

## Metrics

- Completion tokens per second and requests per second.
- TTFT P50/P95/P99.
- TBT P50/P95/P99.
- End-to-end latency.
- GPU utilization and HBM usage.
- CPU memory and NUMA-local allocation.
- H2D/D2H bytes, bandwidth, and queue depth.
- Local versus remote CPU-pool bytes.
- KV residency, rotation count, and dirty-tail count.
- Transfer stalls and exposed scheduler time.
- DSpark acceptance rate and rollback count.
- Errors, cancellations, starvation, and request completion rate.

For transfer attribution, every pressure artifact must also retain the labeled
`vllm:simple_cpu_offload_*` Prometheus samples. The report writes these to
`report/transfer-summary.json` and includes them in the Markdown report; do not
rely on the old unlabelled metric filter. For one matched run per backend, use
the opt-in profiler:

```bash
scripts/run_superinfer_bench.sh \
  --env-file .env.superinfer-pressure \
  --run-id 20260816_pressure_default \
  --nsys
```

The profiler writes an Nsight Systems trace under the run's `nsys/` directory.
Keep profiling runs separate from throughput-only runs because tracing adds
overhead. Inspect queue depth, native submission time, descriptor counts,
H2D/D2H overlap, and local/remote transfer attribution before changing
admission or copy batching.

The bounded pressure profile uses 24 concurrent users and 96 total requests.
This still exceeds the approximately 1.45M-token GPU KV capacity with unique
131K prompts while leaving more activation headroom for matched backend
comparison. The profile intentionally does not enable expandable CUDA allocator
segments because the connector pins/registers KV memory and rejects remappable
CUDA virtual addresses.

## Acceptance gates

- Output equivalence with the vanilla DSpark profile.
- No unbounded transfer backlog.
- No deadlock, request starvation, or cleanup leak.
- No repeated GPU block reuse before transfer completion.
- Actual CPU KV residency during pressure runs.
- Stable throughput across repeated trials.
- Improvement over vanilla at equivalent pressure, or a documented reason why
  a point is not beneficial.
- No material regression in GPU-fitting workloads.
- Transfer attribution must identify either queue starvation, exposed copy
  submission overhead, or NUMA locality loss before implementation tuning.

Bounded multi-store admission is now enabled up to `transfer_queue_depth` and
has passed default and native reload correctness gates. The next pressure pair
must use the stabilized 24-user profile and compare descriptor counts and
throughput with the bounded admission change. Do not increase the cap until
queue growth, CPU residency pressure, and output equivalence remain stable.

The DMA path now coalesces only payload-contiguous physical spans. Pressure
artifacts must record coalesced spans and descriptor counts for both backends;
do not activate broader layout merging without exact reload correctness.

The first coalesced pressure pair completed successfully for both backends.
Default measured `107.83` completion tokens/s with `8.49 GB` stores; native
measured `108.47` tokens/s with `8.09 GB` stores, `12,412` native descriptors,
and zero native errors. Treat the `+0.59%` difference as directional because
transfer volumes differed. Repeat stable-volume samples are required before
calling coalescing a throughput improvement.

Implementation pause boundary: reusable DMA workspaces with reallocation
telemetry, generic queue and descriptor telemetry, indexed residency state, and observational DSpark
boundary bookkeeping are complete. The next code changes must wait for GPU
evidence covering queue overlap, NUMA page placement, workspace impact, and
rollback-safe DSpark rotation.

## Interpretation

Do not claim a SuperInfer gain from a run that never moved KV to CPU. Do not
claim saturation solely from high configured memory utilization. Confirm it with
connector transfer events, CPU residency, and stable queue behavior.

## Current Measurement Boundary

The latest clean sequence at
`benchmark_artifacts/gpu_sequence/20260824_200505` demonstrates that reload
and pressure are different experiments:

- The repeated-prefix reload phase measured `2.961 s` default versus `2.660 s`
  native with identical `342.5 MB` H2D traffic and zero output mismatches.
- The unique-prefix pressure phase measured D2H stores and zero H2D bytes in
  its own before/after counter delta. Its `176.48` default versus `171.48`
  native result cannot establish reload throughput.
- The benchmark harness now records reload phase timings and
  `report/transfer-deltas.json`, which excludes earlier reload traffic from the
  pressure-stage transfer attribution.
- `.env.superinfer-service128` now requests three reload cycles with four
  concurrent reload requests for the next clean sequence. Use the median reload
  phase time as the primary native/default comparison, and report D2H pressure
  separately.

Historical artifacts used a character limit for `prompt_len`; the nominal
`131072` value produced `21850` actual prompt tokens in the latest prior
sequence. The benchmark and reload clients now send exact-length pre-tokenized
prompts. Every new artifact must verify the reported prompt-token range equals
the configured target, and historical character-limited results must remain
separate from the corrected pressure ladder.

## Serving Profile

The maximum-throughput service candidate is `.env.superinfer-serving`. It
enables high-risk proactive DSpark rotation and uses the validated 128 GiB CPU
KV tier, `empty` allocation, NUMA binding, GH200 topology tuning, and the
24-sequence/32768-batched-token shape. It keeps block-first disabled because
DeepSeek-V4/TP=2 layout equivalence is not validated. High-risk service must be
reported separately from conservative normal-mode validation, with
`.env.superinfer-service128` available as rollback.

In the current worker, high-risk mode selects `InlineCopyBackend` before the
optional native DMA backend. Native DMA therefore remains a separate reload
experiment and is not combined with this serving profile.

The automated one-lifetime serving soak is implemented in
`scripts/run_serving_soak.sh`. It evaluates short-context peak, 32K/64K
progression, 131K pressure at 4/8/16 users, repeated-prefix 131K reload, and
16K recovery for each candidate. The report selects the highest short-context
median only when all long-context phases pass and show real CPU-KV movement.

The latest exact-token reload gate at
`benchmark_artifacts/gpu_sequence/token_exact_reload/20260824_220842` passed
for both backends. It used exact `131072`-token prompts and three reload cycles:
default mean reload time was `7.135 s`, native was `6.284 s`, with identical
`6.889 GB` load traffic, `1088` lookup hits, and zero output mismatches. Use
this as reload-path evidence, not as a final saturation claim. High-risk serving
is a separate performance-first operating mode and must retain the normal-mode
rollback profile.
