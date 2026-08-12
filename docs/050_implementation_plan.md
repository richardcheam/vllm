# Implementation Plan

## Tracking

- Source-of-truth map: `docs/070_source_of_truth_traceability.md`
- Living decisions and divergences: `docs/071_progress_decision_log.md`
- Deferred-test ledger (deadline mode): `docs/081_deferred_test_ledger.md`
- Parity exit checklist: `docs/082_parity_exit_checklist.md`
- Block-first/native design spike: `docs/090_block_first_native_design_spike.md`

1. Add SuperInfer/GH200 flags as no-op config fields and CLI flags.
2. Add tests that prove flags parse and land in `CacheConfig`/`SchedulerConfig` without changing defaults.
3. Add telemetry-only counters for scheduler/cache/offload state.
4. Add pure VLT scoring helper and unit tests.
5. Add request rotary metadata enum without changing scheduling behavior.
6. Study and extend modern KV offload allocator for GH200 CPU DRAM limits.
7. Add debug-only single-request swap-out/swap-in for the smallest safe non-MTP, non-prefix-shared case.
8. Compare outputs against vanilla for small model first, then DeepSeek-V4-Flash.
9. Add budgeted proactive movement only after correctness tests pass.
10. Add RotaSched/LVF policy integration.
11. Add DuplexKV batching/block-first/full-duplex optimizations only after correctness and basic telemetry are stable.

## Status Snapshot (2026-06-06)

- Done: 1, 2, 3, 4, 5.
- Partially done: 6 (allocator wiring via `swap_cpu_memory_gb` -> `SimpleCPUOffloadConnector`).
- Done: 7 (debug single-request swap-out/swap-in gate in connector/manager path).
- CPU-testable Step 9 scope done: guarded budgeted proactive movement wired to lazy offload scan depth and scheduler-side decode-tail/small-decode-backlog preemption via `proactive_swap_budget`; VLT helper participates in guarded victim choice when VLT knobs are set, with runtime-estimated connector swap bandwidth plus request timing hints for TTFT/TBT terms. Covered default-off, unsafe-candidate guards, mixed fallback/VLT victim selection, and preempt/resume sequencing.
- Partially done and GPU-smoke validated: 10 (RotaSched/LVF policy integration now preempts multiple guarded candidates in one scheduler step until the configured free-block target is reached, while preserving connector/prefix-sharing/pending-transfer gates and VLT victim ranking. The target is at least `proactive_swap_budget` and can grow to cover the head waiting request's immediate block pressure. Proactive swap no longer enables the debug single-request transfer throttle by default, so lazy offload can batch budgeted movement unless explicitly debug-throttled. Scheduler-side proactive movement also respects available CPU swap blocks.)
- Transfer-observability slice done: `pin_memory_fix` and `swapper_block_first` now propagate through `swap_cpu_memory_gb` extra config, scheduler metadata, telemetry stats, and worker-side validation/logging. `pin_memory_fix` documents the existing allocate-then-`cudaHostRegister` worker path; `swapper_block_first` remains visibility-only and does not change CPU KV layout.
- Transfer-overlap slice done: `DmaCopyBackend` now uses independent load/store submission queues and background threads feeding the existing separate CUDA streams. This removes Python-side load/store submission serialization while preserving the current `cuMemcpyBatchAsync` helper and GPU-derived CPU KV layout.
- Stage-1 layout/copy refactor slice done (behavior-preserving):
  - added explicit worker offload layout descriptor in
    `vllm/v1/simple_kv_offload/layout.py` and worker wiring,
  - split `cuda_mem_ops` into address preparation + submission helpers,
  - kept default runtime behavior as GPU-derived layout + existing
    `cuMemcpyBatchAsync` submit path.
- Stage-2 block-first activation slice done (strict-gated):
  - eligible low-risk paths (`TP=1`, single KV group, non-DeepSeek, tensor
    cache values, architecture allowlist currently `OPTForCausalLM`) now use
    active `cpu_layout=block_first`,
  - transfer address mapping now supports different source/destination
    block strides,
  - unsupported paths auto-fallback to `gpu_derived` with explicit reason.
- DeepSeek-focused DuplexKV parity slice started: lazy offload now opportunistically
  stores confirmed full blocks while requests are still running, giving DeepSeek a
  safe synced-block CPU residency path without changing its forced `gpu_derived`
  layout fallback.
- Proactive victim cost now accounts for synced full blocks already resident in
  CPU cache. VLT bandwidth cost and CPU capacity checks use only unsynced blocks,
  moving closer to DuplexKV's cheap preemption after eager block rotation.
- Proactive preempted requests now carry `ROTARY_SWAPPED` metadata state instead
  of appearing as plain waiting requests, aligning modern request metadata with
  SuperInfer's rotary lifecycle while keeping modern `PREEMPTED` scheduling
  status for compatibility.
- Proactive pressure estimation now considers both regular waiting and
  skipped/rotary-waiting queue heads, so blocked rotary work can contribute to
  the free-block target instead of only the currently selected scheduling queue.
- Proactive preemption now records rotary accounting on the request: full blocks
  already synced to CPU, full blocks still unsynced, dirty tail tokens, and the
  preemption timestamp. This preserves the metadata needed for DeepSeek-safe
  synced/dirty behavior without changing DeepSeek cache layout.
- Rotary preemption accounting is emitted in scheduler stats so GPU pressure runs
  can confirm synced/unsynced/dirty-tail behavior was actually exercised.
- Finish-time synced-block flush is now implemented in the simple offload
  scheduler: when a request finishes, confirmed full blocks are retained in
  store state and can still be emitted in a subsequent store event even if no
  later scheduler step includes that request in `yield_req_data`.
- Scheduler resume now applies rotary dirty-tail safeguards: external CPU-hit
  tokens are capped for rotary-swapped requests, `ROTARY_SWAPPED` transitions
  to `ROTARY_PENDING_IN` before async reload, and post-recv promotion clamps
  `num_computed_tokens` to force dirty-tail recomputation before running.
- Shared-prefix offload handling moved toward refcount-aware ownership:
  eager store scanning now skips shared-prefix GPU blocks (`ref_cnt > 1`), and
  proactive synced-cost estimation can query owned CPU-resident block count via
  connector API when available.
- Proactive candidate/cost logic now uses owned full-block estimates (refcount
  aware) instead of total per-request blocks, allowing rotation on owned suffix
  blocks even when part of the prefix is shared.
- Connector-side CPU hit estimation now applies rotary ownership caps before
  returning matched external tokens: zero owned-synced blocks produce no async
  remote load, and synced/dirty-tail caps bound rotary reload hit length.
- Scheduler stats now include remote-wait flow counters (`num_remote_wait_entries`,
  `num_remote_wait_promotions`) to debug blocked-queue behavior during
  concurrent serving runs.
- Ownership consistency follow-up completed for connector/scheduler reload caps:
  rotary matched-token caps are now aligned (owned synced + dirty-tail bounds),
  and finish-time flush remains functional with owned-block finish-touch tracking.
- Focused scheduler hardening checks are green for zero-external-token gating,
  blocked-queue FCFS fairness, remote-wait entry/promotion accounting, and
  owned-suffix proactive candidacy.
- Still pending: refcount-aware shared-prefix offload and native DuplexKV/block-first transfer. Both require ownership/transfer metadata that spans multiple request block lists or worker-copy backends, so they remain separate high-risk implementation phases rather than quick extensions to the guarded scheduler path.
- Latest CPU validation: 22 VLT/request/config tests, 31 proactive/timing/priority scheduler tests, 20 simple-offload scheduler tests, 1 copy-backend routing test, 2 Stage-1 layout/cuda-mem helper tests, 5 focused SuperInfer swap config tests, and 2 SuperInfer CLI tests passed in `richard-base-dev`.
- Blocked: 8 (GPU model-output comparison) until GPU availability. Validation handoff: `docs/080_gpu_validation_checklist.md`.
- CPU phase closed: prefix-cache/shared-block safety audit is complete for the guarded proactive path; avoid expanding runtime behavior further before GPU smoke tests.

## Commit Boundaries

- Commit 1: no-op flags and parsing tests.
- Commit 2: telemetry only.
- Commit 3: VLT helper and tests.
- Commit 4: rotary metadata only.
- Commit 5: CPU swap allocator or adapter.
- Commit 6: debug manual swap.

No commit should mix benchmark scripts with serving-engine changes.
