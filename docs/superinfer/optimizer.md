# SuperInfer Optimizer

This is the implementation-first control loop for the DeepSeek-V4/DSpark
target. It is intentionally not a benchmark matrix. It runs one decisive
reload attempt at a time and does not repeat an identical source revision.

## Start

Run inside `richard-base-dev-sysnice`:

```bash
scripts/run_superinfer_optimizer.sh \
  --env-file .env.superinfer-reload \
  --max-attempts 0
```

`--max-attempts 0` means continue until a valid reload is observed. The loop:

1. Waits for all visible GPUs to remain below the configured utilization limit.
2. Starts one clean reload validation.
3. Saves the normal reload artifact plus optimizer logs.
4. Stops on positive store/load deltas and zero output mismatches.
5. Waits for a source fingerprint change before another failed attempt.

It does not kill arbitrary processes. `run_reload_validation.sh --restart`
still uses the repository's `.run/server.pid`; use `--no-restart` when an
already-running server must be preserved.

## Decision Signals

The current target is not throughput tuning. The first gate is a real CPU KV
reload:

- `offload_cpu_cached_keys` must increase after stores.
- `offload_cpu_lookup_hits` must become nonzero.
- `offload_load_requests_created` and `offload_load_events_assigned` must become
  nonzero.
- `offload_load_bytes` must be positive.
- `output_mismatches` must remain zero.

Store-completion logs include:

```text
cached_key_groups={...} source_primary_groups={...}
```

For DeepSeek-V4, the map must include the relevant groups instead of being
dominated by group `0`. If the map is still group-0-only, fix lazy alias
indexing. If all groups are present but lookup hits remain zero, fix hash
boundary/reconciliation. If loads occur but throughput does not improve, move
to persistent native duplex submission.

## Breakthrough Sequence

1. Complete group-aware CPU reload.
2. Replace Python-thread DMA submission with a persistent native duplex backend,
   while retaining current metadata and residency contracts.
3. Coalesce DeepSeek physical spans and reduce descriptor count per group.
4. Add accepted-token-boundary aggressive DSpark rotation.
5. Measure one saturated workload and only then tune budgets.

Do not start full paper RotaSched/LVF parity first. The official reference is a
heuristic swap scheduler, and current VLT is already more advanced. Scheduling
cannot produce the intended working-set extension while CPU KV reload is not
working.

Do not enable global DeepSeek block-first as a shortcut. The runtime has
heterogeneous groups `[256, 64, 64, 4, 8]`; use group-specific descriptors.

## Artifacts

- Reload attempts: `benchmark_artifacts/optimizer/reload/`
- Optimizer controller log: `benchmark_artifacts/optimizer/optimizer.log`
- Per-attempt gate: `validation.json`
- Server diagnostics: per-attempt `server.log`

The optimizer is allowed to run continuously, but it must only consume GPU time
when the source fingerprint changed or the previous attempt was not executed.

## First Successful Gate

Attempt 12 passed on 2026-08-16:

- Store events/bytes: `8` / `827.2 MB`.
- Load events/bytes: `4` / `456.7 MB`.
- CPU lookup hits: `181`.
- Cached keys after run: `80,272`.
- Output mismatches: `0`.

Artifact: `benchmark_artifacts/optimizer/reload/20260816_103017_reload`.

This closes the first breakthrough boundary: CPU offload extends the working
set through actual group-aware reload rather than store-and-recompute.

## Native Duplex Slice

The next implementation slice is an optional native CUDA batch-submission
backend. Enable it only for an optimizer experiment:

```bash
NATIVE_COPY_BACKEND=1 scripts/run_superinfer_optimizer.sh \
  --env-file .env.superinfer-reload \
  --max-attempts 1
```

The native slice retains the current Python queue, separate H2D/D2H streams,
compute wait events, heterogeneous source/destination strides, and completion
polling. C++/CUDA replaces only the batch API submission. It is disabled by
default and must be measured separately.

The first native attempt passed reachability/correctness at
`benchmark_artifacts/optimizer/reload/20260816_135757_attempt1`: both workers
selected the native backend, the bridge compiled, stores/loads completed, and
output mismatches were zero. This does not yet establish native performance
superiority; the next experiment should capture native bandwidth and overlap.

The corrected native attempt is the baseline for the matched performance probe.

The persistent-plan native attempt also passed at
`benchmark_artifacts/optimizer/reload/20260816_144543_attempt1`:

- Store events/bytes: `7` / `743.2 MB`.
- Load events/bytes: `3` / `342.5 MB`.
- CPU lookup hits: `151`.
- Cached keys after run: `71,566`.
- Output mismatches: `0`.
- Native submission errors: `0`.

This revision moves per-direction descriptor storage and pointer/size-array
construction into the persistent C++/CUDA plan and reports submission counters,
descriptor counts, bytes, submission time, and errors through worker metadata and
Prometheus. It is validated for correctness, but no performance gain is claimed
until the default and native paths are measured with the same reload workload.

## Matched Reload Probe

The default and native paths were run with the same reload profile and harness:

- Default artifact: `benchmark_artifacts/optimizer/reload/20260816_151757_matched_default`.
- Native artifact: `benchmark_artifacts/optimizer/reload/20260816_155500_matched_native`.
- Both passed with three load events, `342.5 MB` loaded, and zero output mismatches.
- Reload phase: `2.353982 s` default versus `2.344421 s` native (`-0.41%`).
- Store phase: `7.784673 s` default versus `7.646898 s` native (`-1.77%`).
- Eviction phase: `18.539051 s` default versus `18.956881 s` native (`+2.25%`).
- Store volume differed (`816.5 MB` versus `914.5 MB`), so this single probe
  establishes reload-time parity, not a statistically significant native win.
- Native submission telemetry reported zero errors; the final capture had `6`
  load submissions / `636` descriptors and `20` store submissions / `1,698`
  descriptors.

The next performance gate is a matched pressure run or repeated reload sample
with stable transfer volumes and CUDA overlap tracing. Keep native mode opt-in
until that measurement shows a repeatable benefit.

The first default pressure attribution run completed at
`benchmark_artifacts/vllm-superinfer-v4/20260816_pressure_default`:

- `128/128` requests succeeded with no server errors.
- Completion throughput: `91.25` tokens/s.
- D2H stores: `44` events, `4,819` blocks, `5.19 GB`.
- CPU-only residency: `25,626` blocks; CPU lookup hits and H2D loads were zero.
- The unique-prefix pressure workload proves real store pressure, but cannot
  prove reload overlap. The matched native run must first compare D2H behavior;
  a repeated-prefix pressure workload is required for H2D attribution.
- The native profiled pressure attempt was invalid: `84/128` requests completed
  before rank 1 hit a `1.99 GiB` CUDA allocation OOM in DeepSeek's fused KV
  insertion path. Native copy telemetry showed zero native submission errors;
  do not compare its partial throughput against the valid default result.
- The benchmark harness now writes `benchmark-status.json` and final snapshots
  on client or engine failure, so invalid pressure attempts remain diagnosable.

## Next Gate: Pressure Attribution

The reload probe did not show a meaningful native speedup, so the next work is
measurement rather than another submission micro-optimization. Use
`.env.superinfer-pressure` and retain the labeled transfer metrics in
`report/transfer-summary.json`. Run one default and one native pressure sample
with `--nsys`, keeping the artifacts separate. The decision order is:

1. If queues are empty while pressure has active transfers, investigate
   scheduler admission and bounded multi-store overlap.
2. If queues are busy and native submission time or descriptor counts dominate,
   implement group-aware physical-span coalescing.
3. If local/remote attribution or page placement is poor, validate NUMA first.
4. Only after transfer overlap is understood, tune DSpark rotation or budgets.

The first native pressure attempt indicates that the profiled configuration
needs an apples-to-apples memory-stability check before scheduler tuning. Keep
multi-store admission deferred until native and default runs both complete the
same request set without OOM.

The pressure profile is now bounded to 24 concurrent users and 96 requests
while retaining unique 131K prompts, enough to exceed the approximately
1.45M-token GPU KV cache. Expandable CUDA allocator segments remain disabled
because pinned/registered KV memory rejects remappable CUDA virtual addresses.
The next default/native Nsight pair uses this stabilized profile.

The stable unprofiled pair completed both request sets successfully. It showed
native directional throughput improvement (`94.92` versus `92.79` completion
tokens/s) but different store volumes. The default log repeatedly reported a
single pending store event while the backend capacity was `8`, and native
telemetry recorded `92` submissions. Lazy store admission is therefore being
expanded to a bounded maximum of `transfer_queue_depth` pending events, not
made unbounded.

Bounded lazy-admission reload validation passed at
`benchmark_artifacts/optimizer/reload/20260816_200000_bounded_default`:

- `12` store events / `1.105 GB` stored.
- `3` load events / `342.5 MB` loaded.
- `149` lookup hits and `106,872` cached keys.
- Zero output mismatches.
- The server log reported `max_inflight_store_events=8` and showed multiple
  store events admitted before prior events completed.

The default lifecycle gate is safe with bounded multi-store admission. Native
bounded reload validation also passed, so the admission change is correctness
validated in both backends.

Native bounded reload validation then passed at
`benchmark_artifacts/optimizer/reload/20260816_203000_bounded_native`:

- `12` store events / `1.097 GB` stored.
- `4` load events / `456.7 MB` loaded.
- `180` lookup hits and `106,601` cached keys.
- Zero output mismatches and zero native errors.
- Native telemetry recorded `16` store submissions / `2,036` descriptors and
  `8` load submissions / `848` descriptors.

Bounded multi-store admission is now correctness-validated in both backends.
The next implementation target is group-aware physical-span coalescing; keep
the queue cap and measure descriptor reduction before changing layout gates.

The coalescing slice is now implemented for both Python and native submission:
contiguous source/destination block runs merge only when both physical strides
equal the payload bytes. Padded or heterogeneous DeepSeek spans remain separate.
The worker reports coalesced-span counts through Prometheus so the next reload
and pressure runs can verify descriptor reduction rather than infer it from
elapsed time.

Coalesced default reload validation passed at
`benchmark_artifacts/optimizer/reload/20260816_210000_coalesced_default`:

- `12` store events / `1.087 GB` stored.
- `4` load events / `456.7 MB` loaded.
- `180` lookup hits and zero output mismatches.
- The telemetry observed `38` store spans and `30` load spans coalesced.

Native coalesced reload validation passed at
`benchmark_artifacts/optimizer/reload/20260816_213000_coalesced_native`:

- `12` store events / `1.113 GB` stored.
- `3` load events / `342.5 MB` loaded.
- `150` lookup hits and zero output mismatches.
- Native telemetry observed `38` store spans and `38` load spans coalesced,
  with `24` store submissions / `1,466` descriptors and `6` load submissions /
  `454` descriptors.

Both coalesced backends now pass exact reload correctness. The next gate is a
stabilized pressure run measuring throughput and descriptor reduction.

The coalesced default pressure run completed at
`benchmark_artifacts/vllm-superinfer-v4/20260816_pressure24_coalesced_default`:

- `96/96` requests succeeded with no server errors.
- Completion throughput: `107.83` tokens/s.
- D2H stores: `77` events, `7,878` blocks, `8.49 GB`.
- Coalesced store spans: `924`.
- Unique prefixes produced zero H2D loads, so this is D2H pressure evidence.

The native coalesced pressure run is the remaining matched throughput gate.

The coalesced native pressure run completed at
`benchmark_artifacts/vllm-superinfer-v4/20260816_pressure24_coalesced_native`:

- `96/96` requests succeeded with no server errors.
- Completion throughput: `108.47` tokens/s versus `107.83` for coalesced
  default (`+0.59%` directional difference).
- D2H stores: `71` events, `7,507` blocks, `8.09 GB`.
- Native telemetry: `140` submissions, `12,412` descriptors, `934` coalesced
  spans, and zero native errors.

Transfer volume differed from the default (`8.49 GB` versus `8.09 GB`), so the
single pair does not establish a causal speedup. It does establish stable
coalesced native execution under pressure. Repeat stable-volume samples are
the next measurement gate; no broader layout merge is justified yet.

## Code-Only Phase Complete

- Python and native batch workspaces are reusable per direction, with
  workspace-reallocation counters.
- Generic transfer telemetry now includes submitted blocks, descriptors, bytes,
  coalesced spans, queue peak depth, enqueue wait, queue-full counts, and
  workspace reallocations.
- Residency has indexed lookup maps by GPU block and request, with compatible
  cleanup/reset behavior and lifecycle tests.
- DSpark accepted-token and dirty-tail bookkeeping is implemented and exported,
  but remains observational; it does not enable rotation or alter admission.

Further implementation now requires GPU evidence for workspace impact, queue
overlap, NUMA page placement, and any DSpark rotation or higher admission limit.
Pause code changes here until repeated stable-volume GPU samples are available.

## Reload Versus Pressure Evidence

The clean unprofiled sequence at
`benchmark_artifacts/gpu_sequence/20260824_200505` clarifies the measurement
boundary:

- Default reload phase: `2.961 s`.
- Native reload phase: `2.660 s`.
- Both reload runs loaded `342.5 MB` in three events and passed exact output
  comparison with zero mismatches.
- The native reload phase was directionally `10.2%` faster in this sample.
- The subsequent unique-prefix pressure stage measured D2H stores only. Its
  pressure-window H2D delta was zero for both backends, so its default/native
  throughput comparison cannot measure reload benefit.
- Pressure throughput was `176.48` tokens/s default and `171.48` tokens/s
  native. Treat this as a D2H pressure result, not a contradiction of the
  favorable reload-time signal.

The harness now persists reload phase timings, supports three repeated reload
cycles with four concurrent reload requests in `.env.superinfer-service128`,
waits for transfer queues to quiesce at stage boundaries, and writes measured
counter deltas to `report/transfer-deltas.json`. The corrected exact-token
reload comparison has now completed; the next GPU gate is high-risk serving
validation, followed by a corrected pressure sample. Do not combine reload-time
and D2H-pressure claims.

The historical prompt generator exposed a fidelity gap: `prompt_len` was a
character limit, not a tokenizer-token target. The latest nominal `131072`
reload prompt contained `21850` actual prompt tokens. The clients now use the
model tokenizer and send exact-length pre-tokenized prompts, so new artifacts
must verify that reported prompt-token minima and maxima equal the configured
target. Historical artifacts remain a separate, weaker workload.

The corrected reload-only sequence at
`benchmark_artifacts/gpu_sequence/token_exact_reload/20260824_220842` is now
the primary reload evidence. It used exact `131072`-token prompts and three
reload cycles with four concurrent requests. Default reload times were
`4.647/6.504/10.256 s`; native times were `4.264/6.462/8.125 s`. Both paths
loaded `6.889 GB` in nine events, recorded `1088` lookup hits, and produced
zero mismatches. Native was `11.9%` faster by mean reload time, while median
improvement was `0.6%`; keep the result directional until more fresh samples
are collected.

## Maximum-Throughput Serving Candidate

The deployment objective permits high-risk scheduling, so the recommended
service profile is `.env.superinfer-serving`. It enables proactive DSpark
rotation with the validated 128 GiB CPU-KV tier, `empty` allocation, explicit
NUMA binding, GH200 topology tuning, and the stable 24-sequence/32768-batched-
token shape. It intentionally keeps `SWAPPER_BLOCK_FIRST=0` because
DeepSeek-V4/TP=2 block-first layout equivalence remains unvalidated.

High-risk mode currently selects the inline transfer backend in the worker and
therefore takes precedence over `NATIVE_COPY_BACKEND`. The
serving profile therefore does not enable `NATIVE_COPY_BACKEND`; native DMA is
an independent reload experiment. The corrected exact-token reload result
showed a directional native reload benefit, while the D2H pressure result did
not show a native end-to-end advantage.

Start inside `richard-base-dev-sysnice` with
`scripts/launch_superinfer.sh .env.superinfer-serving`. Roll back by stopping
that service and launching `.env.superinfer-service128`, which preserves the
same CPU tier, NUMA settings, and topology tuning but disables high-risk
rotation. During high-risk service, monitor output correctness, dirty
residency, pending transfers, queue depth, and server errors.

The serving profile is currently live and passed the launch warmup plus a
bounded smoke at `benchmark_artifacts/serving_smoke/20260825_000506_high_risk`:
`8/8` exact-4096-token requests, `219.8` completion tokens/s, and zero
failures. This is operational validation only; the next serving gate is
exact-token long-context pressure with high-risk rotation enabled.

The complete profile search is automated by
`scripts/run_serving_soak.sh`. It must run with no service active, because each
candidate needs a fresh server lifetime for startup-fixed scheduler settings:

```bash
bash scripts/run_serving_soak.sh \
  --service-env .env.superinfer-serving
```

The runner measures short-context peak, 32K/64K progression, 131K pressure at
increasing concurrency, repeated-prefix 131K reload, and 16K recovery. It
selects a combined profile only when long-context movement and recovery pass.

## Temporary GPU Sequence

The post-change backend gates can be run sequentially inside
`richard-base-dev-sysnice` with:

```bash
bash scripts/run_gpu_validation_sequence.sh
```

The driver uses one server lifetime per backend, for two server startups total.
It waits for both GPUs to be idle before launching a backend, launches once, runs
reload against that live server with explicit `--no-restart`, runs the inference
pressure benchmark with `--no-start`, then stops the server before the other
backend. It stops on the first invalid result, cleans up on interruption, and
writes a timestamped sequence directory under `benchmark_artifacts/gpu_sequence/`.
The default service profile is `.env.superinfer-service128` (128 GiB total CPU
KV capacity, 64 GiB per TP rank). Use `--skip-default` to resume at native after
a completed default backend. Add `--profile-pressure` only when profiling is
needed; it profiles those same server lifetimes, including paired reload and
pressure stages, rather than launching a second benchmark pass.

The 128 GiB setting is an opt-in serving/validation recommendation for the
observed 1.2 TiB host, not a new default. It provides approximately 64 GiB per
TP rank and should be monitored for NUMA placement and interaction with other
services before production adoption. More capacity improves retention and
reduces CPU-tier eviction risk; it does not by itself increase transfer
bandwidth or inference throughput.
Use `--skip-reload` or `--skip-pressure` to resume a subset without changing
the underlying harnesses.

Do not enable block-first layout based on the reload result alone. Aggressive
DSpark rotation is enabled deliberately in `.env.superinfer-serving` because
the deployment objective is maximum throughput, but it remains an operational
risk and must be monitored with the normal-mode rollback available.
