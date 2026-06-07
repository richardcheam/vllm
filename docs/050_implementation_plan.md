# Implementation Plan

## Tracking

- Source-of-truth map: `docs/070_source_of_truth_traceability.md`
- Living decisions and divergences: `docs/071_progress_decision_log.md`

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
- Latest CPU validation: 22 VLT/request/config tests, 31 proactive/timing/priority scheduler tests, 16 simple-offload scheduler tests, and 2 SuperInfer CLI tests passed in `richard-base-dev`.
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
