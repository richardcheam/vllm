# SuperInfer Continuation Context

## Target

- Checkout: `/data1/home/az04297/re-SuperInfer/vllm-superinfer-v4`
- Runtime: vLLM 0.26-compatible port, two GH200 144G GPUs, TP=2, PP=1
- Model: DeepSeek-V4-Flash-0731 with DSpark, five speculative tokens
- Production tree `/data1/home/az04297/re-SuperInfer/vllm-modern` is out of scope

## Forward-Port Boundary

- v0.27.1 is a patch release on v0.27.0, whose broad changes include a
  PyTorch 2.13 upgrade and major DeepSeek/KV-offload work.
- Do not wholesale-upgrade this checkout yet. NUMA support and selected newer
  KV mechanisms are already present.
- Use `docs/superinfer/forward-port-plan.md` for staged backport candidates and
  the eventual v0.27.x upgrade gate.
- Upstream accelerators do not equal SuperInfer parity. Native C++/CUDA/ZMQ
  DuplexKV, complete RotaSched/LVF queues, and paper-level retry/residency
  semantics remain separate long-term objectives.

## Evidence

- Valid matrix: normal SuperInfer mean `1289.2` tok/s vs vanilla `1154.6`.
- High-risk SuperInfer mean `1249.4` tok/s and remains unvalidated for production.
- Native offload is invalid under 512K warmup due to the upstream
  `OffloadingConnector._build_store_jobs` key/block assertion.
- Corrected run `20260812_100239_superinfer` validated custom Prometheus export;
  all 20 snapshots contained `vllm:simple_cpu_offload_*` metrics.
- That workload recorded zero CPU KV transfer counters. Do not attribute the
  throughput gain to offloading until a pressure run proves real movement.
- Pressure runs `20260812_105727_superinfer`, `115813_superinfer`, and
  `120211_superinfer` produced no valid benchmark artifacts; the first timed
  out and the later directories were empty or used the normal profile.

## Current Direction

Prioritize implementation gains over parameter tuning. The intended first
breakthrough is a real duplex transfer engine, not full paper parity.

1. Independent load/store admission and bounded queues.
2. Per-transfer CUDA compute events and safe event lifetime.
3. Pre-forward load submission after ordering is proven.
4. Explicit per-group/per-block residency state.
5. Descriptor-driven heterogeneous DeepSeek DMA.
6. Actual NUMA placement and measured locality.
7. DSpark-safe rotation at accepted-token boundaries.

## Latest Code Slice

- `manager.py`: lazy store admission now checks only pending stores, allowing a
  store to coexist with an unrelated load.
- `simple_cpu_offload_connector.py`: exposes `has_pending_store_transfers`.
- `worker.py`: each store submission records a distinct compute-done CUDA event.
- `test_worker.py`: regression test verifies consecutive stores do not reuse the
  same wait event.
- `copy_backend.py`: DMA queues are bounded per direction, expose queue-depth
  snapshots, and latch asynchronous copy-thread failures for worker-side
  propagation. `transfer_queue_depth` defaults to `8`.
- `metadata.py`/`worker.py`/`manager.py`: worker load/store queue depths are
  propagated to scheduler telemetry independently of configured capacity.
- `worker.py`: H2D loads are submitted from `start_load_kv()` before forward;
  duplicate event submission is suppressed. Stores remain in `get_finished()`
  and retain compute-done ordering.
- `simple_cpu_offload_connector.py`: production connector now delegates
  `start_load_kv()` to the worker, making pre-forward loads reachable.
- `copy_backend.py`: contiguous fallback block mappings are coalesced into
  sliced copies; worker event queues use `deque` for O(1) polling.
- `manager.py`: observational per-request/group/block residency records expose
  GPU-only, store/load-in-flight, CPU-only, synced, and invalid counts.
- Eager store scanning now reconciles tracked CPU blocks against the CPU cache;
  middle-block evictions become GPU-only and can be re-stored instead of being
  skipped by the old cursor.
- Finished requests are included in eager store preparation so confirmed final
  full blocks are flushed; unconfirmed trailing positions are classified dirty.
- Finished request residency records are compacted after cleanup; in-flight
  transfer records remain until completion.
 - Lazy `prepare_store_specs()` now routes `finished_req_ids` through confirmed
  eager-style store preparation before applying the normal lazy watermark.
  This is the first implementation change intended to force real CPU KV stores
  for completed requests.
 - Lazy finished state is retained when final block IDs exist; immediate cleanup
   is limited to requests with no blocks, preserving the next finished-request
   store pass.
- `cuda_mem_ops.py`: batch DMA parameters now carry independent source stride,
  destination stride, and payload-byte arrays, removing the block-first slab
  stride-equality blocker without enabling DeepSeek block-first mode.

## Rules

- Keep normal mode safe: `SUPERINFER_HIGH_RISK_MODE=0` and
  `SWAPPER_BLOCK_FIRST=0` until correctness gates pass.
- Do not claim a transfer-driven gain without nonzero store/load metrics.
- Do not run production benchmark warmups as a substitute for pressure tests.
- Production launch warmup is separate, opt-in, best-effort, and calls
  `scripts/warmup_superinfer.sh`.
- Run runtime validation only in `richard-base-dev-sysnice`; avoid interrupting
  unrelated GPU workloads.

## Next Implementation Slice

- Add worker-to-scheduler queue-depth/error telemetry if needed for live
  attribution.
- Validate bounded queues and error propagation under a controlled GPU test.
- Replace cursor/count residency with explicit per-group/per-block states,
  including authoritative dirty tails, eviction reconciliation, and finish-time
  stores. Eager re-store and confirmed final-block handling are implemented;
  DSpark rollback/dirty-tail GPU semantics remain to validate.
- Then make `OffloadLayoutDescriptor` drive heterogeneous DeepSeek DMA.
- Validate the new stride-aware batch path with real GPU K/V equivalence before
  changing DeepSeek layout activation gates.

## Runtime Gates Still Required

- Pressure run with `.env.superinfer-pressure` and nonzero store/load metrics.
- The pressure profile must keep enough KV memory for `MAX_MODEL_LEN=1048576`;
  `GPU_MEMORY_UTILIZATION=0.70` is invalid because vLLM requires about 31.49
  GiB KV cache for one 1M-token request. Current bounded pressure settings use
  `0.92`, 16 users, and 32 requests.
- A pressure startup retry found and fixed a worker constructor mismatch for
  `transfer_queue_depth`; rerun startup after this fix before interpreting any
  pressure result.
- NUMA backport audit found the full v0.27.1-style NUMA utility/config/hooks
  already present in this checkout. `.env.superinfer` now enables explicit
  `NUMA_BIND_NODES=0,1`; container probing reports auto nodes `[0,1]` and
  `/usr/bin/numactl`. Live page-placement validation is still required.
- Pressure run `20260813_110324_superinfer` was clean but negative: 32 unique
  131K-token requests peaked at about 72% GPU KV usage and generated zero CPU
  stores/loads. The next workload must exceed the roughly 2.96M-token GPU KV
  capacity using more total unique prompt tokens while retaining 0.92 GPU
  utilization and a valid 1M max sequence length.
- Pressure run `20260813_111829_superinfer` was also clean but negative because
  `MAX_NUM_SEQS=16` capped the active batch below GPU KV capacity. The profile
  now uses `MAX_NUM_SEQS=32` and `BENCH_USERS=32` so 32 concurrent 131K prompts
  can exceed the approximately 2.96M-token GPU cache.

## Reload Harness

- `scripts/validate_superinfer_reload.py` sends deterministic repeated prefixes
  in store and reload phases and compares exact output text.
- `scripts/run_reload_validation.sh` wraps it with metrics/GPU snapshots.
- Acceptance requires positive before/after store/load event and byte deltas and
  zero output mismatches; the result is written to `validation.json`.
- The first reload attempt proved stores but not loads because the 4 GB-per-rank
  CPU tier could evict target prefixes. Use `.env.superinfer-reload` for the
  next gate: 64 GB CPU tier, 4 target prefixes, and 8 eviction prefixes.
- The 64 GB reload run passed exact output equality and NUMA binding but still
  had zero loads because eviction was serial. The harness now runs eviction
  concurrently; use it for the next H2D validation.
- Reload run `20260814_105153_reload` passed NUMA, output equality, and CPU
  stores (`9` events, `829 MB`), but CPU loads stayed zero despite CPU-only
  residency. Next code target: add CPU-hit lookup/admission counters and fix
  the path that should create `_reqs_to_load` for a GPU-evicted prefix.
- Lookup/admission telemetry is now implemented; inspect the five
  `offload_cpu_lookup_*` and `offload_load_*` counters in the next reload run.
- The latest valid artifact `20260814_111350_reload` has exact output equality
  (`4/4`), `8` store events (`737.8 MB`), `92` CPU-only blocks, `140` CPU lookup
  requests, zero lookup hits, and zero load events. This proves the remaining
  boundary is lookup/admission, not basic store completion or output equality.
- DeepSeek-V4 is hybrid in the live runtime. The CPU coordinator reports group
  block sizes `[256, 64, 64, 4, 8]`, scheduler block size `256`, and hash block
  size `4`. Do not apply a single-group hash fix without first inspecting the
  one-shot per-group lookup diagnostic.
- `run_reload_validation.sh` now skips launch warmup, uses proxy-safe health
  checks, and writes `validation.json` with metric deltas. A valid reload gate
  requires positive store/load event and byte deltas plus zero output
  mismatches.
- Implemented a targeted hybrid lookup fallback: if aggregate reconciliation
  returns zero, retry each KV group and admit only their common prefix rounded
  to the `256`-token scheduler boundary. This preserves all-group restore
  semantics and is covered by the focused scheduler suite (`32 passed`).
- Added `offload_cpu_cached_keys` telemetry and reload artifact fields
  `cached_keys_before`/`cached_keys_after`. The next run must distinguish missing
  CPU cache keys from a hybrid request-hash miss.
- Added direct first-block probes for each DeepSeek group. The next server log
  will report `direct_group_first_hits` and `direct_group_hash_indices`, which
  identify key-boundary mismatches without another speculative code change.
- Optimizer attempt 12 passed: `8` store events (`827.2 MB`), `4` load events
  (`456.7 MB`), `181` lookup hits, `80,272` cached keys, and zero output
  mismatches. Group-aware CPU KV reload is proven. Next slice: optional
  persistent native duplex submission.
- Added optional `NativeDmaCopyBackend` and `native_copy.py`. It replaces only
  batch-copy submission with a lazily compiled C++/CUDA bridge; Python queues,
  streams, wait events, and metadata remain unchanged. It is disabled by
  default and must be measured with `NATIVE_COPY_BACKEND=1`.
- Native backend source checks pass (`52` focused tests). The next GPU action is
  one explicitly labeled native optimizer attempt; do not mix its result with
  the default Python DMA backend.
- Native attempt passed at
  `benchmark_artifacts/optimizer/reload/20260816_135757_attempt1`: both workers
  logged `NativeDmaCopyBackend`, native batch submission compiled, `8` stores
  (`805.7 MB`), `1` load (`114.2 MB`), `58` lookup hits, and zero mismatches.
  Native performance/overlap remains unmeasured.
- Native persistent-plan attempt passed at
  `benchmark_artifacts/optimizer/reload/20260816_144543_attempt1`: `7` stores
  (`743.2 MB`), `3` loads (`342.5 MB`), `151` lookup hits, `71,566` cached keys,
  and zero mismatches. The native bridge now owns persistent descriptors and
  native pointer/size-array construction. One server metrics interval observed
  `8` store submissions / `680` descriptors and `4` load submissions / `424`
  descriptors, with zero native errors. This is correctness and reachability
  evidence only; matched default/native performance is still required.
- Matched default/native reload probe completed at
  `benchmark_artifacts/optimizer/reload/20260816_151757_matched_default` and
  `benchmark_artifacts/optimizer/reload/20260816_155500_matched_native`.
  Both passed with `3` load events, `342.5 MB` loaded, and zero output
  mismatches. Reload elapsed time was `2.353982 s` default versus `2.344421 s`
  native (`-0.41%`, `9.6 ms`). Store and eviction volumes/times varied, so
  this is parity evidence rather than a native speedup claim. Native counters
  had zero errors; the final metrics capture reported `6` load submissions /
  `636` descriptors and `20` store submissions / `1,698` descriptors.
- A prior native attempt failed before activation because worker environment
  propagation and the optimizer fingerprint were incomplete. Those issues are
  fixed; the corrected native result above is the baseline for performance
  attribution.
- The 20260815 run confirmed `cached_key_groups={0:85}` at the first valid
  lookup, so only primary group-0 keys were visible. Store completion now copies
  all GPU block-hash aliases to the CPU cache block and logs the key-group map
  after each store event. The next run should produce group IDs beyond `0`, then
  positive lookup/load counters.
- The 19:40 run further showed bulk lazy stores had
  `source_primary_groups={0:85}`. The lazy scanner now uses request-derived
  aliases indexed during GPU block tracking, rather than requiring only the
  primary `node.block_hash`. Focused coverage is `41 passed`.
- The 19:59 run showed that indexed aliases were not yet wired into CPU store
  completion, so bulk CPU key maps remained group-0-heavy. Completion now
  mirrors the indexed aliases directly. The next run should show group-complete
  key maps before lookup/load admission can succeed.
- Numerical restore correctness with real DeepSeek KV data.
- CUDA trace showing H2D overlap with model kernels.
- Fallback/block-first trace showing reduced copy dispatches.
- Matched default-vs-native pressure measurement with native submission,
  bandwidth, and overlap telemetry. The reload probe is complete but did not
  establish a statistically meaningful speedup.

## Current Measurement Gate

- The report harness now preserves labeled `vllm:simple_cpu_offload_*` samples
  and writes `report/transfer-summary.json` for each benchmark.
- `run_superinfer_bench.sh --nsys` opt-in wraps the launched server with Nsight
  Systems and writes the trace under the benchmark artifact's `nsys/` folder.
- Next action: matched default/native `.env.superinfer-pressure` runs with
  Nsight tracing. Do not change scheduler admission until queue depth,
  transfer overlap, and local/remote placement are observed under pressure.
- The default profiled pressure run proved D2H pressure: `128/128` successful,
  `91.25` completion tokens/s, `44` stores, `5.19 GB` stored, and `25,626`
  CPU-only blocks. Unique prefixes produced zero H2D loads.
- The native profiled pressure run is invalid: `84/128` requests completed
  before rank 1 hit a `1.99 GiB` CUDA OOM in DeepSeek fused KV insertion.
  Native submission errors remained zero, so this is not evidence of a native
  copy failure. Do not implement multi-store admission from this run.
- The pressure profile is now stabilized at 24 concurrent users and 96 total
  unique 131K-token requests, still above GPU KV capacity. Expandable CUDA
  allocator segments are intentionally disabled because pinned/registered KV
  memory rejects remappable virtual addresses. The next matched default/native
  runs use this profile before any scheduler admission change.
- The stable unprofiled pair completed `96/96` requests for both backends.
  Default measured `92.79` completion tokens/s with `43` stores / `5.10 GB`;
  native measured `94.92` tokens/s with `46` stores / `4.63 GB`, zero native
  errors, `92` native submissions, and `8,590` descriptors. Because store
  volumes differed, this is directional evidence rather than a causal speedup.
- The default pressure log showed one pending store event while the configured
  backend capacity was eight. Lazy admission is now bounded by
  `transfer_queue_depth`, allowing multiple in-flight store events without
  removing the safety cap. Focused validation is required before another GPU
  run.
- Bounded default reload validation passed at
  `benchmark_artifacts/optimizer/reload/20260816_200000_bounded_default`: `12`
  stores / `1.105 GB`, `3` loads / `342.5 MB`, `149` lookup hits, `106,872`
  cached keys, and zero output mismatches. The log reported
  `max_inflight_store_events=8` and multiple admitted store events. Native
  bounded reload validation is the next gate.
- Native bounded reload validation passed at
  `benchmark_artifacts/optimizer/reload/20260816_203000_bounded_native`: `12`
  stores / `1.097 GB`, `4` loads / `456.7 MB`, `180` lookup hits, `106,601`
  cached keys, zero mismatches, zero native errors, `16` store submissions /
  `2,036` descriptors, and `8` load submissions / `848` descriptors. Bounded
  multi-store admission is correctness-validated in both backends. Next target:
  group-aware physical-span coalescing.
- Implemented stride-safe physical-span coalescing in `cuda_mem_ops.py` and the
  native C++ bridge. Only contiguous source/destination IDs with payload-sized
  strides merge; padded heterogeneous spans remain one descriptor per block.
  Coalesced-span telemetry is propagated through worker metadata and metrics.
  Focused coverage is now `57 passed`; the next gate is bounded reload plus
  descriptor-count evidence.
- Coalesced default reload validation passed at
  `benchmark_artifacts/optimizer/reload/20260816_210000_coalesced_default`:
  `12` stores / `1.087 GB`, `4` loads / `456.7 MB`, `180` lookup hits, zero
  mismatches, `38` store spans coalesced, and `30` load spans coalesced. Native
  coalesced reload validation also passed at
  `benchmark_artifacts/optimizer/reload/20260816_213000_coalesced_native`: `12`
  stores / `1.113 GB`, `3` loads / `342.5 MB`, `150` lookup hits, zero
  mismatches, zero native errors, `38` store spans, `38` load spans, `24`
  store submissions / `1,466` descriptors, and `6` load submissions / `454`
  descriptors. The next gate is stabilized coalesced pressure.
- Coalesced default pressure completed at
  `/workspace/benchmark_artifacts/vllm-superinfer-v4/20260816_pressure24_coalesced_default`:
  `96/96` successful, `107.83` completion tokens/s, `77` store events,
  `8.49 GB` stored, `924` coalesced store spans, and zero server errors. Unique
  prefixes produced no H2D loads. Native coalesced pressure also completed:
  `96/96`, `108.47` completion tokens/s, `71` stores / `8.09 GB`, `140` native
  submissions, `12,412` descriptors, `934` coalesced spans, and zero native
  errors. The `+0.59%` difference is directional because volumes differed.
  Next gate: repeated stable-volume pressure samples.

## Code-Only Pause Boundary

- Reusable Python/native batch workspaces are implemented with reallocation
  telemetry.
- Generic transfer telemetry covers submitted blocks, descriptors, bytes,
  coalesced spans, queue peaks, enqueue wait, queue-full events, and workspace
  reallocations.
- Residency lookup maps are indexed by GPU block and request and preserve reset
  and lightweight test-fixture compatibility.
- DSpark accepted-token/dirty-tail bookkeeping is observational only and does
  not enable rotation.
- Stop implementation work here until GPU evidence selects the next target:
  workspace impact, NUMA placement, queue overlap, or DSpark rotation.
- CPU capacity experiment is isolated in `.env.superinfer-cpu128`: 128 GiB total,
  64 GiB per TP rank, and `CPU_KV_ALLOCATION_MODE=empty`. Existing baseline,
  reload, and pressure profiles retain their prior capacities and `zero` mode.
- Temporary sequential driver: `scripts/run_gpu_validation_sequence.sh` runs
  one server lifetime per backend: launch once, reload with the live server,
  benchmark with `--no-start`, then stop. It waits for idle GPUs only before
  each launch and uses separate timestamped artifacts. The default launch
  profile is `.env.superinfer-service128`; `--skip-default` resumes at native.
  Use `--profile-pressure` only when a server-lifetime Nsight trace is needed.
- The paired sequence uses `.env.superinfer-service128`: 128 GiB total CPU KV
  capacity, or 64 GiB per TP rank, with `CPU_KV_ALLOCATION_MODE=empty`. This is
  reasonable on the observed 1.2 TiB host for retention, but is not itself a
  throughput optimization and remains opt-in. The sequence launches only once
  per backend, reuses that server for reload plus inference, and then stops it.
- Default pressure run completed at
  `/workspace/benchmark_artifacts/vllm-superinfer-v4/20260816_pressure_default`:
  `128/128` successful, `91.25` completion tokens/s, `44` store events,
  `5.19 GB` stores, `25,626` CPU-only blocks, zero lookup hits, and zero loads.
  It is a valid D2H pressure result but not an H2D/reload result because all
  pressure prefixes were unique.

## Serving Decision

The performance-first serving candidate is `.env.superinfer-serving`. It
enables `PROFILE=superinfer-high-risk` and `SUPERINFER_HIGH_RISK_MODE=1`, uses
128 GiB total CPU KV capacity with `empty` allocation, binds NUMA nodes `0,1`,
and enables GH200 topology tuning. It uses the stable 24-sequence/32768-batched-
token shape and keeps `SWAPPER_BLOCK_FIRST=0` because DeepSeek-V4/TP=2
block-first layout is not validated.

High-risk mode currently takes the inline transfer path; native DMA is not
enabled in the serving profile. The exact-token reload gate is passed, but
high-risk DSpark rejection/rollback stress remains incomplete. Roll back to
`.env.superinfer-service128` if output divergence, dirty-residency growth,
pending transfers, queue growth, or server errors appear.

The worker gives high-risk mode precedence over `NATIVE_COPY_BACKEND`, so the
serving profile intentionally measures/uses `InlineCopyBackend`. Native DMA is
kept as a separate reload-focused experiment rather than mixed into the
high-risk service decision.

The high-risk serving candidate is currently running from
`.env.superinfer-serving` in `richard-base-dev-sysnice`. Health/model checks
returned HTTP `200`; launch warmup passed `2/2` and `8/8`; and the bounded smoke
artifact `benchmark_artifacts/serving_smoke/20260825_000506_high_risk` passed
`8/8` exact-4096-token requests at `219.8` completion tokens/s. This is basic
serving validation, not long-context saturation evidence. The next step is a
bounded exact-token high-risk pressure run and rollback/output stress check.

The staged runner is implemented as `scripts/run_serving_soak.sh`. It generates
candidate environments, refuses to start while `.run/server.pid` is live, and
writes per-phase results plus `report/serving-soak-report.md`. It has not been
executed yet; stop the current service before running it.

## Latest Context Refresh

The clean paired sequence at
`benchmark_artifacts/gpu_sequence/20260824_200505` completed both backends
successfully. The important result is the separation between stages:

- Reload correctness passed for both backends with three load events,
  `342.5 MB` of H2D traffic, and zero output mismatches.
- Native reload took `2.660 s`; default took `2.961 s`, a directional `10.2%`
  native reduction in that sample.
- The following unique-prefix pressure benchmark created D2H stores but no
  measurement-window H2D loads. It measured `176.48` tokens/s default versus
  `171.48` native, so it is not a reload-performance test.
- The earlier `20260824_191747` sequence also favored native reload, but the
  reload-time difference was only `2.8%`. Repeated reload samples are required.

The harness was updated after that sequence to retain reload phase timings,
repeat reload cycles three times with four concurrent reload requests, wait for
transfer queues to quiesce at stage boundaries, and write per-repeat transfer
deltas. Future pressure reports must use those deltas rather than cumulative
server-lifetime load counters.

Historical benchmark fidelity issue: the old `prompt_len` argument truncated
strings by character count. The nominal `131072` reload prompt was only
`21850` tokenizer tokens in the recorded artifact. The clients now construct
exact-length pre-tokenized prompts and report actual prompt-token usage. Keep
the historical artifact separate from new token-accurate measurements.

The latest exact-token reload sequence is
`benchmark_artifacts/gpu_sequence/token_exact_reload/20260824_220842`. Both
backends passed three reload cycles using exact `131072`-token prompts, with
`6.889 GB` loaded in nine events, `1088` lookup hits, and zero mismatches.
Default reload times were `4.647/6.504/10.256 s`; native times were
`4.264/6.462/8.125 s`. Native was `11.9%` faster by mean time, but only `0.6%`
faster by median, so repeated fresh samples remain useful.
