# Validation Status

## Completed

- Target checkout remains based on vLLM `v0.26.0`.
- Target `.venv` exists under `vllm-superinfer-v4/.venv`.
- Runtime work was performed in `richard-base-dev-sysnice`.
- The venv resolves the container's Torch `2.10.0a0+b558c986e8.nv25.11`
  with CUDA `13.0`; the venv-local replacement Torch packages were removed.
- GH200 discovery passed against the real host: two GPUs, NUMA nodes 0 and 1,
  and `NV18` peer connectivity.
- Target source syntax compilation passed.
- Focused SuperInfer suite passed:

  ```text
  33 passed, 2 skipped
  ```

  This includes offload scheduler, worker, integration, topology, and VLT
  tests, run with `--noconftest` to avoid unrelated global test fixtures.
- Successful GPU SuperInfer benchmark completed on 2026-08-03:
  - 3 repeats, 120 requests each, 360/360 successful.
  - Best observed completion throughput: 1,829.6 tokens/s.
  - Best repeat TTFT P50/P99: 689/3,250 ms.
  - Best repeat TBT P50/P99: 35/275 ms.
  - Artifact: `benchmark_artifacts/vllm-superinfer-v4/20260803_182055_superinfer`.
- A report generator now produces Markdown tables, SVG charts, and aggregate
  JSON from the saved benchmark runs.
- Matched three-profile matrix completed at
  `benchmark_artifacts/matrix/20260803_185536`:
  - Vanilla DSpark mean: `1338.1` completion tokens/s.
  - Native offload mean: `1374.1` completion tokens/s (`+2.7%` vs vanilla).
  - SuperInfer mean: `1395.6` completion tokens/s (`+4.3%` vs vanilla,
    `+1.6%` vs native offload).
  - Every saved measurement repeat reports `120/120` successful requests.
  - Best saved repeat: SuperInfer at `1639.9` completion tokens/s.
- Cross-profile report: `benchmark_artifacts/matrix/20260803_185536/matrix-results.md`.
- The benchmark client and runner now persist TTFT/TBT summaries, GPU snapshots,
  Prometheus metrics, and log diagnostics for future matrix runs. Existing
  matrix artifacts predate this capture and therefore cannot be retrofitted
  with latency samples.
- Four-profile matrix `benchmark_artifacts/matrix/20260811_163049` completed:
  - Vanilla: mean `1154.6` completion tokens/s, 360/360 successful.
  - Normal SuperInfer: mean `1289.2` completion tokens/s, 360/360 successful.
  - High-risk SuperInfer: mean `1249.4` completion tokens/s, 360/360 successful.
  - Native offload: invalid; its engine crashed in v0.26's
    `OffloadingConnector` assertion while building store jobs.
- The matrix report now marks failed profiles as `FAILED` instead of treating
  them as zero-throughput results.
- Standalone normal SuperInfer run `20260811_224020_superinfer` completed all
  warmup stages and three measurement repeats with `120/120` successful
  requests per repeat. Mean measurement throughput was approximately
  `1190` completion tokens/s.
- The run emitted the full custom SuperInfer telemetry payload in the server
  log, including CPU pool size, topology discovery, local/remote pool sizes,
  and transfer counters. Its saved `/metrics` files were empty because the
  container HTTP proxy intercepted localhost curl requests; the benchmark
  capture helper now uses `curl --noproxy '*'` and clears failed captures.
- Corrected run `20260812_100239_superinfer` validated live Prometheus export:
  all 20 captured metrics snapshots were nonempty and contained
  `vllm:simple_cpu_offload_*` series. All three measurement repeats completed
  `120/120` successfully, with mean throughput approximately `1296` completion
  tokens/s and zero logged errors or engine-dead events.
- The same corrected run reported topology and CPU-pool gauges, including
  `offload_gh200_topology_discovered=1`, `3987` CPU blocks, and `2988` local
  pool blocks. Transfer counters remained zero, so actual KV movement was not
  triggered by this workload and transfer-bandwidth attribution remains open.
- Added opt-in production warmup: `launch_superinfer.sh` now calls
  `scripts/warmup_superinfer.sh` after health readiness. It runs a small robust
  latency canary and basic throughput check, captures JSON/GPU/metrics data,
  and continues best-effort if either stage fails or times out.
- Began the implementation-first optimization phase: lazy store admission no
  longer serializes stores behind unrelated loads, and each queued store owns a
  distinct compute-done CUDA event. A worker regression test covers event
  identity; runtime overlap and transfer correctness remain unvalidated.
- Added bounded per-direction DMA queues with configurable
  `transfer_queue_depth` (default `8`) and latched copy-thread error propagation
  through the worker. CPU-only queue admission and worker failure tests pass;
  GPU queue saturation and overlap remain unvalidated.
- Added separate live load/store queue-depth telemetry propagated through worker
  metadata. The implementation-focused tests pass `10 passed`; broader
  scheduler tests remain blocked by the environment's unrelated OpenSSL/OPT
  model-inspection failure.
- Moved SimpleCPUOffload H2D load submission to the v1 pre-forward
  `start_load_kv()` hook with duplicate event suppression. Stores remain
  post-forward and compute-gated. Worker/copy/telemetry tests now pass
  `11 passed`; GPU overlap and numerical restore behavior remain unvalidated.
- Wired the production `SimpleCPUOffloadConnector.start_load_kv()` to the
  worker, so pre-forward loading is reachable in the real v1 connector path.
  The non-batched fallback now coalesces contiguous block runs into sliced
  copies, and worker event polling uses `deque`. Focused tests pass `13 passed`.
- Added observational per-request/group/block residency records and Prometheus
  counts for GPU-only, store/load-in-flight, CPU-only, synced, and invalid
  states. Lifecycle and telemetry tests pass `18 passed`; these records are not
  yet authoritative for admission or eviction reconciliation.
- Updated batch DMA descriptors to carry independent source stride,
  destination stride, and payload bytes, enabling descriptor-compatible
  heterogeneous/block-first transfers in source. Container tests pass `19
  passed`; DeepSeek GPU numerical and performance validation remains pending.
- Eager residency reconciliation now invalidates tracked CPU entries after
  individual eviction and allows confirmed blocks to be re-stored. The
  container-focused suite passes `20 passed`; dirty-tail/final-block GPU
  semantics remain unvalidated.
- Eager preparation now processes `finished_req_ids`, flushes confirmed final
  full blocks, and marks unconfirmed tails dirty. The complete container-focused
  suite passes `50 passed`; this is the point where GPU validation is required
  before further residency/DMA changes.
- Residency telemetry now compacts finished request records while retaining
  records needed by in-flight transfers. The container-focused suite passes
  `51 passed`; live pressure and NUMA validation remain the next gates.
- Lazy completed-request handling now routes `finished_req_ids` through
  confirmed-block store preparation before the normal watermark check. The full
  container-focused suite passes `52 passed`; the next pressure run is required
  to verify nonzero CPU stores and subsequent H2D reloads.
- Fixed the remaining lazy-final-store lifecycle bug: `request_finished()` now
  retains finished state when final block IDs exist, instead of deleting it
  before the next `finished_req_ids` pass. The full container-focused suite
  remains `52 passed` after this fix.
- Added CPU-hit lookup/admission counters: lookup requests, lookup hits, hit
  tokens, load requests created, and load events assigned. The focused suite
  passes `48 passed`; the next reload run will identify the remaining zero-load
  boundary without guessing.
- Added a dedicated repeated-prefix reload harness:
  `scripts/run_reload_validation.sh`. It separates deterministic store and
  eviction/reload phases, captures metrics, and checks exact output equality.
  The first run proved stores but did not force reloads; the harness now adds a
  distinct eviction phase before the repeated-prefix reload.
- Reload run `20260814_001535_reload` proved stores (`56` events, `5.06 GB`)
  but produced zero loads and five output mismatches. It is not a valid restore
  correctness result because the 4 GB-per-rank CPU tier could evict target
  prefixes during the eviction phase. A dedicated `.env.superinfer-reload`
  profile now uses a 64 GB CPU tier and smaller target/eviction sets to retain
  target prefixes while forcing GPU eviction.
- Reload run `20260814_004510_reload` passed exact output equality (`4/4`) and
  proved NUMA binding plus stores (`12` events, `1.10 GB`), but still had zero
  loads because eviction requests were serial. The reload harness now runs the
  eviction phase concurrently; the next run is required for H2D load proof.
- Reload run `20260814_105153_reload` validated NUMA binding, exact output
  equality (`4/4`), and CPU stores (`9` events, `829 MB` local writes). It still
  produced zero CPU loads despite `172` CPU-only blocks, so the remaining gate
  is CPU-hit lookup/admission visibility rather than store or NUMA correctness.
- DeepSeek-V4 startup diagnostics show that the model uses a hybrid KV layout,
  not a single MLA group. The CPU coordinator reports group block sizes
  `[256, 64, 64, 4, 8]`, scheduler block size `256`, and request-hash
  granularity `4`. The next lookup investigation must compare per-group hit
  results rather than treating a zero aggregate hit as a global hash mismatch.
- Reload harness reliability was tightened: launcher health checks bypass the
  container proxy, reload startup disables production warmup, and each run now
  writes `validation.json` with before/after metric deltas. A reload is invalid
  unless store/load event and byte deltas are positive and exact output matches
  pass.
- Optimizer attempt 12 passed the real group-aware reload gate at
  `benchmark_artifacts/optimizer/reload/20260816_103017_reload`: `8` store
  events (`827.2 MB`), `4` load events (`456.7 MB`), `181` lookup hits,
  `80,272` cached keys, and zero output mismatches. All five DeepSeek groups
  were present in the store completion maps. This closes the first CPU working-
  set extension milestone.
- Added an optional `NativeDmaCopyBackend` behind `--native-copy-backend` and
  `NATIVE_COPY_BACKEND=1`. It preserves current Python queues, H2D/D2H streams,
  compute wait events, heterogeneous descriptors, and completion polling while
  replacing only batch-copy submission with a lazy C++/CUDA bridge. It is
  GPU-validated in the dedicated reload harness.
- Native optimizer attempt passed at
  `benchmark_artifacts/optimizer/reload/20260816_135757_attempt1`: both workers
  selected `NativeDmaCopyBackend`, the native batch bridge compiled and enabled,
  stores were `8` / `805.7 MB`, loads were `1` / `114.2 MB`, lookup hits were
  `58`, and output mismatches were `0`. This is native correctness/reachability
  evidence, not yet a performance win; native-vs-default overlap and bandwidth
  still need one matched measurement.
- Native persistent-plan attempt passed at
  `benchmark_artifacts/optimizer/reload/20260816_144543_attempt1`. The C++/CUDA
  bridge now owns reusable per-direction descriptors and per-batch pointer/size
  construction. The run recorded `7` store events (`743.2 MB`), `3` load events
  (`342.5 MB`), `151` lookup hits, `71,566` cached keys, and zero output
  mismatches. Native telemetry reported zero submission errors; the server log
  observed `8` store submissions / `680` descriptors and `4` load submissions /
  `424` descriptors during one metrics interval. This validates the optimized
  native path, but does not establish a throughput or overlap improvement.
- Matched default/native reload probe completed in
  `benchmark_artifacts/optimizer/reload/20260816_151757_matched_default` and
  `benchmark_artifacts/optimizer/reload/20260816_155500_matched_native`.
  Both passed exact output validation with `3` load events and `342.5 MB` of
  reload traffic. The reload phase was `2.353982 s` for the default Python DMA
  backend and `2.344421 s` for native, a `9.6 ms` (`0.41%`) native reduction.
  Store phases were `7.784673 s` versus `7.646898 s`; eviction phases were
  `18.539051 s` versus `18.956881 s`. Store volume differed between runs
  (`816.5 MB` default versus `914.5 MB` native), so this single probe shows
  reload-time parity, not a statistically established performance win.
  Native Prometheus counters reported zero submission errors, `6` load
  submissions / `636` descriptors, and `20` store submissions / `1,698`
  descriptors in the final capture.
- A later native attempt initially failed before activation because worker
  environment propagation and the optimizer fingerprint were incomplete. Those
  issues are fixed; the corrected native result above is the baseline for
  native performance attribution.
- Native-offload diagnosis: the server completed warmup stages 1-5, then died
  during the 512K stage in the upstream
  `OffloadingConnector._build_store_jobs` assertion that the per-group
  offload-key and GPU-block lists have equal length. The target changes do not
  modify that scheduler, and native offload does not enter the SuperInfer
  proactive path. Treat this as an unresolved v0.26 control-profile failure,
  not as a valid throughput result or evidence against SuperInfer.
- Pressure profile startup at `GPU_MEMORY_UTILIZATION=0.70` was rejected before
  serving because only `7.15 GiB` KV memory was available while the configured
  1M max sequence length requires `31.49 GiB`. The pressure profile was corrected
  to retain `0.92` GPU utilization and use bounded 16-user/32-request pressure;
  no pressure benchmark result exists yet.
- The first corrected pressure retry exposed a separate source integration bug:
  `SimpleCPUOffloadConnector` passed `transfer_queue_depth` to the worker before
  the worker constructor accepted it. The constructor and DMA backend wiring are
  now fixed; container import and focused tests pass.
- v0.27.1 NUMA support is already present in this checkout's
  `vllm/utils/numa_utils.py`, `ParallelConfig`, and multiprocessing hooks. The
  launcher now enables it for production `.env.superinfer` with explicit
  `NUMA_BIND_NODES=0,1`; container probing detected GPU NUMA nodes `[0, 1]` and
  `/usr/bin/numactl`. Actual page placement under transfer remains unvalidated.
- Forward-port review: v0.27.1 is only a patch release on v0.27.0, while
  v0.27.0 includes a broad PyTorch 2.13/core change. A staged backport plan is
  documented; no wholesale baseline upgrade is recommended before clean
  v0.27.x vanilla reproduction and dependency validation.
- Valid pressure run `20260813_110324_superinfer` completed `32/32` requests at
  131K-token unique prompts with no errors. GPU KV usage peaked around `72%`,
  while store/load events, bytes, queue depth, and CPU-only residency remained
  zero. This is a valid negative pressure result: the workload did not exceed
  the approximately 2.96M-token GPU KV capacity and therefore did not exercise
  CPU movement.
- Pressure run `20260813_111829_superinfer` completed `128/128` requests at
  131K-token unique prompts with no errors, but remained negative for transfer
  attribution. The profile allowed only 16 active sequences, so its live batch
  could hold about 2.1M prompt tokens, below the approximately 2.96M GPU KV
  capacity; transfer counters and CPU residency remained zero.

## Historical and Remaining Work

## Serving Decision

Because the operating objective is maximum throughput and high-risk mode is
acceptable, `.env.superinfer-serving` is now the recommended serving candidate.
It combines the validated memory/transfer configuration with
`SUPERINFER_HIGH_RISK_MODE=1`:

- `SWAP_CPU_MEMORY_GB=128`, approximately 64 GiB per TP rank.
- `CPU_KV_ALLOCATION_MODE=empty`.
- NUMA binding on nodes `0,1` and GH200 topology tuning.
- `MAX_NUM_SEQS=24` and `MAX_NUM_BATCHED_TOKENS=32768`.
- High-risk proactive DSpark rotation enabled.
- `SWAPPER_BLOCK_FIRST=0`; high-risk mode does not bypass the unvalidated
  DeepSeek-V4/TP=2 layout gate.
- Default Python/inline transfer path. Native DMA remains an opt-in experiment
  and is not selected by the serving profile.

High-risk mode takes precedence over `NATIVE_COPY_BACKEND` in the current worker
and selects `InlineCopyBackend`; the serving profile leaves native DMA disabled
intentionally. This keeps the service decision separate from the native reload
experiment.

This is a deliberate performance-first candidate, not a final high-risk
correctness claim. The corrected exact-token reload gate passed, but sustained
DSpark rejection/rollback under proactive rotation still needs stress proof.
The normal-mode rollback profile is `.env.superinfer-service128`.

## Previous Clean Paired Sequence

The clean unprofiled sequence completed successfully at
`benchmark_artifacts/gpu_sequence/20260824_200505`:

- Both backends completed `96/96` pressure requests with zero server errors.
- The default pressure sample measured `176.48` completion tokens/s.
- The native pressure sample measured `171.48` completion tokens/s. This is
  `2.83%` below default and is not evidence against the reload optimization,
  because the pressure workload uses unique prefixes and generated no H2D
  traffic during its measurement window.
- Default pressure transfer deltas were `74` store events / `8.079 GB` and
  `0` load events / `0` load bytes.
- Native pressure transfer deltas were `79` store events / `8.017 GB` and
  `0` load events / `0` load bytes. Native copy counters reported zero errors.
- Both repeated-prefix reload gates passed with `3` load events,
  `342.5 MB` loaded, and zero output mismatches.
- The reload phase was `2.961 s` for default and `2.660 s` for native, a
  directional `10.2%` native reduction in this sample. An earlier sequence
  also favored native reload (`3.672 s` versus `3.777 s`, `2.8%`). These are
  reload-time signals, not yet repeated statistical claims.

The benchmark/report harness has since been tightened:

- Reload phase timings are persisted in each sequence result.
- Reload validation defaults can repeat the eviction/reload cycle three times
  with four concurrent reload requests.
- Reload artifacts record and fail closed on actual prompt-token minima/maxima
  that do not equal the configured prompt length.
- Transfer queues are polled to quiescence before reload snapshots and before
  pressure repeat snapshots.
- `report/transfer-deltas.json` reports counter increases attributable to the
  pressure measurement repeats rather than cumulative server-lifetime values.
- The report no longer contains hard-coded claims about 360 requests or the old
  `1,829.6` tokens/s result.

The historical sequence above used character-truncated strings: its nominal
`131072` prompt length was only `21850` actual prompt tokens. The benchmark and
reload clients now construct exact-length pre-tokenized prompts and send token
ID lists to the completion API. New artifacts must verify
`prompt_tokens_min == prompt_tokens_max == prompt_len`; old artifacts must not
be compared directly with the corrected token-accurate workload.

## Latest Exact-Token Reload Sequence

The corrected reload-only sequence completed successfully at
`benchmark_artifacts/gpu_sequence/token_exact_reload/20260824_220842`:

- Both backends used exact `131072`-token prompts. Every request reported
  `prompt_tokens_min=prompt_tokens_max=131072`.
- Both backends completed three reload cycles with four concurrent reload
  requests, `28` store events, `9` load events, `1088` lookup hits, and zero
  output mismatches.
- Default loaded `6.889 GB`; native loaded the identical `6.889 GB`.
- Default reload times were `4.647`, `6.504`, and `10.256` seconds, with a
  mean of `7.135` seconds and median of `6.504` seconds.
- Native reload times were `4.264`, `6.462`, and `8.125` seconds, with a mean
  of `6.284` seconds and median of `6.462` seconds.
- Native was `11.9%` faster by mean reload time in this sample. The median
  improvement was only `0.6%`, so this is strong directional evidence but not
  a final statistical claim.
- Native workers selected `NativeDmaCopyBackend`; native load/store submission
  counters were nonzero and native submission errors remained zero.
- Final queue depths and pending load/store event gauges were zero for both
  backends after the quiescence barrier.

This is the current evidence for the reload/native path. The maximum-throughput
serving candidate remains `.env.superinfer-serving`, which enables high-risk
proactive DSpark rotation but uses the inline transfer path. Use
`.env.superinfer-service128` for normal-mode rollback.

## Live Serving Validation

The recommended high-risk serving profile was launched successfully inside
`richard-base-dev-sysnice`:

- Profile: `.env.superinfer-serving`.
- Process ownership: `.run/server.profile` records the serving profile and the
  launcher refuses to reuse a running server with another profile.
- Health and model endpoints returned HTTP `200`.
- Both GH200 GPUs remained visible after startup.
- The launch warmup passed `2/2` robust-canary requests and `8/8` basic requests.
- The live serving smoke artifact is
  `benchmark_artifacts/serving_smoke/20260825_000506_high_risk`.
- The smoke used exact `4096`-token prompts and completed `8/8` requests at
  `219.8` completion tokens/s with zero failures.
- Live metrics confirmed topology discovery/tuning, block-first disabled, no
  invalid residency blocks, and no native DMA submissions. High-risk mode uses
  the intended inline transfer backend.

This validates startup and basic serving operation, not maximum long-context
throughput. The service is intentionally left running. The next measurement is
a staged one-lifetime exact-token high-risk serving soak, followed by repeated
output and rollback checks. The soak framework is
`scripts/run_serving_soak.sh`; it preserves phase artifacts even when transfer
quiescence is delayed. Roll back with `.env.superinfer-service128` if those
checks show divergence, dirty-residency growth, queue growth, or server errors.

The soak framework is implemented but has not been executed from this checkout.
It must be run only after stopping the current serving process, because each
candidate requires a fresh server lifetime.

- The four-profile matrix Priority-0 run is complete, but its native-offload
  control profile is invalid under the exhaustive 512K warmup workload.
- The full editable vLLM installation was not completed. It attempted a native
  CMake build and exceeded the available execution window.
- No final saturation claim has been made.
- The pressure measurement harness now retains labeled
  `vllm:simple_cpu_offload_*` samples in `report/transfer-summary.json` and
  supports opt-in Nsight Systems capture through `run_superinfer_bench.sh
  --nsys`. Profiled pressure remains attribution-only; reload timing is the
  current performance signal.
- Default profiled pressure run completed at
  `benchmark_artifacts/vllm-superinfer-v4/20260816_pressure_default` with
  `128/128` successful requests, `91.25` completion tokens/s, `44` store
  events (`5.19 GB`), `25,626` CPU-only blocks, zero lookup hits, zero loads,
  and zero server errors. This validates D2H pressure only; unique prefixes do
  not exercise H2D reload.
- Native profiled pressure run is invalid: `84/128` requests completed before
  rank 1 hit a `1.99 GiB` CUDA allocation OOM in DeepSeek fused KV insertion.
  The native bridge reported zero submission errors before the model OOM. Its
  partial throughput and transfer counters must not be compared with the valid
  default run. The next gate is a stable apples-to-apples pressure configuration
  with failure artifacts captured by `benchmark-status.json`.
- The next pressure pair is stabilized at 24 concurrent users and 96 unique
  131K-token requests without expandable CUDA allocator segments, which are
  incompatible with pinned/registered KV memory. This remains above GPU KV
  capacity while reducing peak activation pressure.
- The stable unprofiled matched pair completed `96/96` requests for both
  backends. Default measured `92.79` completion tokens/s with `43` stores /
  `5.10 GB`; native measured `94.92` tokens/s with `46` stores / `4.63 GB`,
  zero native errors, `92` native submissions, and `8,590` descriptors. Because
  transfer volumes differed, this is directional, not causal, performance
  evidence. The default log showed one pending store event against queue depth
  eight; lazy admission is now bounded by that configured depth.
- Default post-change reload in sequence `20260824_104213` passed before the
  cron workload occupied the GPUs. Artifact:
  `benchmark_artifacts/gpu_sequence/20260824_104213/reload/postchange_default_20260824_104213`.
  It completed in `7.571 s` store, `17.778 s` eviction, and `1.747 s` reload;
  transfer deltas were `12` stores (`1.099 GB`) and `4` loads (`456.7 MB`),
  with `180` lookup hits and zero output mismatches. Generic telemetry recorded
  `2,040` store blocks / `1,440` descriptors and `848` load blocks / `780`
  descriptors, with queue peak depth `1` in each direction and no queue-full
  events. The native stage was blocked by the unrelated cron service, not by
  the default implementation.
- Bounded default reload validation passed at
  `benchmark_artifacts/optimizer/reload/20260816_200000_bounded_default` with
  `12` store events (`1.105 GB`), `3` load events (`342.5 MB`), `149` lookup
  hits, `106,872` cached keys, and zero output mismatches. The log reported
  `max_inflight_store_events=8` and multiple admitted stores. Native bounded
  reload validation remains pending.
- Native bounded reload validation passed at
  `benchmark_artifacts/optimizer/reload/20260816_203000_bounded_native` with
  `12` store events (`1.097 GB`), `4` load events (`456.7 MB`), `180` lookup
  hits, `106,601` cached keys, zero mismatches, and zero native errors. Native
  telemetry recorded `16` store submissions / `2,036` descriptors and `8` load
  submissions / `848` descriptors. Bounded multi-store admission is now
  correctness-validated in both backends.
- Implemented stride-safe physical-span coalescing for Python and native DMA.
  Contiguous source/destination runs merge only when both strides equal payload
  bytes; padded heterogeneous spans remain unmerged. Coalesced-span counters are
  now exported through worker metadata and Prometheus. Focused checks pass `57`.
- Coalesced default reload validation passed at
  `benchmark_artifacts/optimizer/reload/20260816_210000_coalesced_default` with
  `12` stores (`1.087 GB`), `4` loads (`456.7 MB`), `180` lookup hits, zero
  mismatches, `38` coalesced store spans, and `30` coalesced load spans. Native
  coalesced validation also passed at
  `benchmark_artifacts/optimizer/reload/20260816_213000_coalesced_native` with
  `12` stores (`1.113 GB`), `3` loads (`342.5 MB`), `150` lookup hits, zero
  mismatches, zero native errors, `38` store spans, `38` load spans, `24` store
  submissions / `1,466` descriptors, and `6` load submissions / `454`
  descriptors. Both coalesced backends pass exact reload correctness.
- Coalesced default pressure completed at
  `benchmark_artifacts/vllm-superinfer-v4/20260816_pressure24_coalesced_default`
  with `96/96` successful requests, `107.83` completion tokens/s, `77` store
  events (`8.49 GB`), `924` coalesced store spans, and zero server errors. This
  is D2H pressure evidence; unique prefixes produced no H2D loads. Native
  coalesced pressure also completed with `96/96` successful requests,
  `108.47` completion tokens/s, `71` store events (`8.09 GB`), `140` native
  submissions, `12,412` descriptors, `934` coalesced spans, and zero native
  errors. The `+0.59%` directional difference is confounded by transfer volume
  variation; repeated stable-volume samples are required for a causal claim.

- The later clean sequence at `benchmark_artifacts/gpu_sequence/20260824_200505`
  confirms that reload and pressure must be reported separately. Its native
  reload phase was directionally faster, while its unique-prefix pressure phase
  was directionally slower and had zero measurement-window H2D bytes.
- The GPU-independent implementation phase is complete: reusable batch
  workspaces with reallocation telemetry, generic submission/queue telemetry,
  indexed residency lookups,
  and observational DSpark accepted-token/dirty-tail bookkeeping are present.
  No DSpark rotation or higher admission policy was enabled. Further
  implementation requires repeated stable-volume GPU evidence.
- Added an isolated CPU-capacity experiment profile
  `.env.superinfer-cpu128` with 128 GiB total CPU KV capacity and opt-in empty
  allocation. Existing profiles remain unchanged in effective capacity and use
  zero allocation. Capacity extension is not a performance claim until measured
  under pressure.
- Temporary GPU sequence driver `scripts/run_gpu_validation_sequence.sh` now
  automates idle-GPU waits before launch, one-start-per-backend paired reload and
  pressure gates, optional server-lifetime Nsight traces, per-stage validation,
  and timestamped artifacts. It uses `.env.superinfer-service128` by default and
  supports `--skip-default` for native resumption. It stops at the first failed
  stage and composes the existing harnesses without changing their runtime
  behavior.
- The sequence now uses a unified one-start-per-backend lifecycle through
  `.env.superinfer-service128`: launch once, reload with `--no-restart`, run the
  inference benchmark with `--no-start`, then stop. The 128 GiB total CPU KV
  capacity is an opt-in retention setting for the 1.2 TiB host, not a claimed
  throughput gain.
- The paired sequence uses explicit `--no-restart` for reload and `--no-start`
  for inference while each server remains alive. A complete default/native run
  therefore uses two server startups rather than four. `--profile-pressure`
  profiles those same lifetimes and does not add duplicate benchmark launches.
- The robust matrix JSON did not persist TTFT/TBT samples, so regenerated
  latency sections are explicitly marked unavailable.
- Native offload remains a v0.26 upstream connector failure under this pressure
  workload and must not be used as a valid performance baseline until isolated.
- Live `/metrics` validation passed after the localhost proxy bypass fix.
  Nonzero transfer attribution remains pending a workload that forces CPU KV
  stores or loads.
- CPU reload lookup remains unresolved. The latest valid artifact
  `20260814_111350_reload` has exact output equality (`4/4`), CPU stores (`8`
  events, `737.8 MB`), `92` CPU-only blocks, `140` lookup requests, zero lookup
  hits, and zero load events. Do not claim CPU H2D reload from this artifact.
- Implemented the first targeted reload fix: when hybrid aggregate lookup
  returns zero, the scheduler now retries per-group and admits only a common
  `256`-token-aligned prefix across all DeepSeek groups. Focused scheduler
  coverage passes `32 passed`; GPU reload verification is the only remaining
  evidence gate for this slice.
- Added `offload_cpu_cached_keys` plus `cached_keys_before`/`cached_keys_after`
  to the reload evidence so the next run can separate missing CPU key
  registration from hybrid lookup mismatch.
- Added direct per-group first-block probes to the one-shot post-store
  diagnostic. The next run will identify whether the `[256, 64, 64, 4, 8]`
  groups use mismatched hash boundaries or whether aggregate reconciliation is
  at fault.
- The 20260815 reload artifact captured `cached_keys_after=765`, but the first
  post-store lookup saw `cached_key_groups={0:85}` and all direct group probes
  false. This identified the concrete bug: CPU store completion published only
  primary group-0 keys, omitting GPU block-hash aliases for the other DeepSeek
  groups.
- CPU store completion now mirrors all GPU block-hash aliases to the CPU cache
  block and logs the key-group map after each completed store event. Focused
  scheduler and telemetry coverage passes `40 passed`; the next GPU run is the
  reload gate for alias visibility and H2D loads.
- The 20260815 19:40 run showed bulk lazy stores with
  `source_primary_groups={0:85}`, proving the lazy scanner still admitted only
  primary group-0 hashes. Lazy block tracking now derives and indexes request
  aliases before scanning free blocks. Focused coverage passes `41 passed`;
  the next GPU run is the gate for group-complete lazy stores and reloads.
- The 19:59 run showed the alias index was populated only for later store
  batches, but CPU completion was not yet consuming that index; bulk events
  still reported `cached_key_groups={0:...}`. Store completion now mirrors the
  indexed aliases directly. Focused coverage remains `41 passed`; the next run
  should show non-primary group keys on bulk events.
- The 20:17 run still showed bulk events with `source_primary_groups={0:85}`;
  only a later event gained group-4 aliases. The optimizer controller is now
  available to avoid repeated manual runs: it waits for GPU idle, executes one
  reload attempt per source fingerprint, preserves diagnostics, and stops only
  on a valid H2D reload.
- The robust benchmark now fails fast when readiness is lost or a run has zero
  successful requests, preventing connection-refused runs from being reported
  as performance data.
- Matrix diagnosis found that the original SuperInfer scheduler path treated
  `proactive_swap_budget=2400` as an unconditional GPU free-block watermark,
  while normal DSpark mode excluded DSpark requests from proactive victims.
  This could add lazy scan overhead without enabling rotation. The target code
  now keeps the budget as a rotation cap, derives lazy storage targets from
  actual pressure, and skips needless normal-mode DSpark scans.
- The affected offload scheduler test suite previously passed after the change:
  `29 passed`. A later combined focused command is still running an
  integration case and has not produced a final result; no additional pass
  claim is made from that process.
- Broad vLLM scheduler/config tests remain blocked when the source checkout is
  not represented by an installed `vllm` distribution. In that mode v0.26's
  platform resolver selects `UnspecifiedPlatform` and fails before those tests
  reach the modified code.

## Interpretation

The current result validates source-level wiring and includes one successful GPU
SuperInfer workload. It does not establish model output equivalence, native
CUDA transfer correctness under load, DSpark compatibility under preemption, or
performance improvement over vanilla. Those require matched controls and a
controlled saturation comparison in `richard-base-dev-sysnice`.
