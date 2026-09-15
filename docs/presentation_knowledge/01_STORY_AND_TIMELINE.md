# Engineering Story And Timeline

The chronology below is reconstructed from Git history, repository documents,
validation logs, and the two checkout roles. It intentionally separates the
validated production track from the newer integration track.

## Stage 0 — Serving Objective

**Problem:** Serve a very large DeepSeek-V4 model on two GH200 GPUs while
supporting realistic long-context and concurrent workloads.

**Why it mattered:** GPU HBM was the constrained resource while large Grace
memory remained available.

**Evidence:** `docs/100_reasoning_tree_offload_results.md:122-141`;
`SUPERINFER_KNOWLEDGE_BASE.md:11-91`.

## Stage 1 — Modern vLLM Baseline

**Observation:** Standard vLLM already provides continuous batching, paged KV
management, prefix caching, tensor parallelism, and OpenAI-compatible serving.

**Limitation:** Aggregate throughput alone did not describe user experience;
higher concurrency can increase throughput while TTFT and per-user speed
deteriorate.

**Historical baseline context:** approximately 8K prompts produced 352.2 tok/s
at concurrency 4, 724.1 tok/s at concurrency 8, and 624.4 tok/s at concurrency
16, with sharply worse high-concurrency TTFT.

**Evidence:** `SUPERINFER_KNOWLEDGE_BASE.md:95-146`.

## Stage 2 — SuperInfer Motivation

**Hypothesis:** Treat CPU/Grace memory as a warm KV capacity tier instead of
waiting until GPU HBM is exhausted.

**Design questions:** Which blocks are hot? Which are CPU-resident? How can
movement overlap compute? How should locality and movement cost affect
scheduling?

**Evidence:** `SUPERINFER_KNOWLEDGE_BASE.md:148-210`.

## Stage 3 — Why A Forward-Port Was Necessary

**Problem:** Original SuperInfer targeted an older vLLM architecture, roughly
0.6-era, while the target stack uses vLLM V1 and newer scheduler/KV interfaces.

**Decision:** Reproduce capability intent, not source-code structure.

```text
old SuperInfer implementation
        |
        | changed APIs and invariants
        v
modern vLLM V1 landing zones
        |
        v
functional/capability parity where safe
```

**Evidence:** `SUPERINFER_KNOWLEDGE_BASE.md:214-251`;
`docs/010_superinfer_delta_map.md`.

## Stage 4 — CPU-KV Tier And Transfer Pipeline

**Implementation:** `SimpleCPUOffloadConnector`, CPU block pools, D2H stores,
H2D reloads, independent queues, CUDA streams, compute events, and telemetry.

**Validation:** Exact reload tests eventually proved positive stores and loads
with zero output mismatches.

**Evidence:** `SUPERINFER_KNOWLEDGE_BASE.md:255-276,347-434`;
`docs/100_reasoning_tree_offload_results.md:176-200`.

## Stage 5 — Scheduler, Allocator, And Free-List Hardening

**Failures:** Pressure exposed grouped-allocation exhaustion, stale free-list
counters, duplicate free-list insertion, and rollback inconsistencies.

**Fixes:** Transactional allocation rollback, reference-count restoration,
stale-counter checks, duplicate-safe freeing, and atomic list operations.

**Evidence:** `docs/100_reasoning_tree_offload_results.md:358-395`;
`docs/050_implementation_plan.md`;
`docs/071_progress_decision_log.md`.

## Stage 6 — NUMA And Topology Work

**Implementation:** GPU-to-NUMA discovery, explicit worker binding, logical
local/remote CPU-KV pool planning, bandwidth cost hints, and remote-fallback
telemetry.

**Limitation:** Physical CPU page placement was not proven. Logical locality
labels are not equivalent to measured memory placement.

**Evidence:** `SUPERINFER_KNOWLEDGE_BASE.md:291-343`;
`docs/090_launch_recipe.md:221-241`.

## Stage 7 — Reload Correctness

**Problem:** Early reload runs proved stores but produced no loads or mismatches
because eviction did not force target prefixes out of GPU or because CPU-tier
capacity was insufficient.

**Fix:** Exact-token reload harness, concurrent eviction, larger CPU-KV profile,
group-aware lookup/admission visibility.

**Result:** 131K-token exact reload profile passed with zero output mismatches.

**Evidence:** `docs/100_reasoning_tree_offload_results.md:68-88`;
`docs/081_cpu_validation_status.md`;
`docs/071_progress_decision_log.md`.

## Stage 8 — Transfer-Quiescence And Hang Investigation

**Symptom:** A final client response could complete while asynchronous D2H
stores were still pending. EngineCore could become idle before transfer
completion metadata reached the scheduler.

**Fix:** Implemented `has_pending_push_work()` and made the harness require
queue and pending-transfer quiescence after each phase.

**Result:** Post-fix exact-token long-context soak completed 48/48 with stores,
loads, no EngineDead events, no CUDA OOM, no Xid errors, and quiescent queues.

**Evidence:** `/home/az04297/re-SuperInfer/vllm-superinfer-v4-integration/docs/superinfer/validation-status.md:232-250`;
`docs/100_reasoning_tree_offload_results.md:503-513`.

## Stage 9 — Production Lifecycle

**Implementation:** Startup warmup, scoped graceful stop, orphan EngineCore/
worker cleanup, read-only failure diagnostics, server-local exit evidence, and
optional user-level `deepseek-vllm.service` with pause/resume maintenance hold.

**Evidence:** `docs/110_systemd_service_guide.md`;
`docs/reasoning_tree_8202_canary.md`.

## Stage 10 — Newer v0.26 Integration Track

The separate integration checkout advances the architecture to vLLM v0.26 and
adds DSpark, `CPU_KV_ALLOCATION_MODE=empty`, native DMA experiments, stronger
quiescence gates, and high-risk proactive rotation.

It has real correctness and pressure evidence but incomplete DSpark rollback
and high-risk saturation proof. It should be presented as the next research
track, not as a replacement for the accepted reasoning-tree production result.

**Evidence:** `/home/az04297/re-SuperInfer/vllm-superinfer-v4-integration/docs/superinfer/README.md`;
`docs/superinfer/validation-status.md`.

## Timeline Summary

```text
serving objective
      ↓
modern vLLM baseline
      ↓
SuperInfer memory-tier hypothesis
      ↓
modern V1 forward-port
      ↓
CPU-KV stores/reloads
      ↓
allocator and free-list failures
      ↓
transactional scheduler/block fixes
      ↓
NUMA/topology modeling
      ↓
reload correctness
      ↓
quiescence/hang diagnosis
      ↓
heavy exact-token pressure validation
      ↓
production lifecycle and recovery
      ↓
newer v0.26 + DSpark/high-risk research track
```
