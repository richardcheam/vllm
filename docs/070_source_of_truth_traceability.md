# SuperInfer Source of Truth and Traceability

This document is the canonical map between:

- the SuperInfer paper semantics,
- the official SuperInfer release code,
- modern vLLM landing zones,
- and the current forward-port status in this branch.

Use it first when investigating bugs, deciding whether to follow old code, or
justifying intentional divergence.

## Authority Model

When sources differ, resolve in this order:

1. **Modern vLLM + DeepSeek-V4 correctness/safety constraints**
   (`docs/030_deepseekv4_cache_state_audit.md`).
2. **Paper-level semantic intent** (arXiv `2601.20309v2`, Sections 4.2 and 4.3).
3. **Official SuperInfer release implementation**
   (`/workspace/re-SuperInfer/SuperInfer`, branch `superinfer`).
4. **Current forward-port implementation details** (this repo).

Interpretation:

- Paper defines *what* the system means to do.
- Release code shows *what was actually shipped*.
- Modern constraints define *what is safe to port now*.

## Fast Debug Workflow

1. Identify the failing behavior area in the matrix below.
2. Read the paper semantic reference for intended behavior.
3. Check old release code to understand shipped heuristics and shortcuts.
4. Check modern landing-zone code and current status.
5. For block-first/native transfer work, consult
   `docs/090_block_first_native_design_spike.md` before editing worker/copy code.
6. If behavior differs, record a new divergence entry in
   `docs/071_progress_decision_log.md` before changing code.

## Traceability Matrix

| Area | Paper semantic source | Official SuperInfer release code | Modern landing zone | Current status |
|---|---|---|---|---|
| CLI/config knobs | Sec. 4.1 (system controls), Sec. 4.2/4.3 tuning knobs | `vllm/engine/arg_utils.py`, `vllm/config.py` | `vllm/engine/arg_utils.py`, `vllm/config/cache.py`, `vllm/config/scheduler.py`, `vllm/config/vllm.py` | Implemented; parsed and wired. `swap_cpu_memory_gb` now activates `SimpleCPUOffloadConnector` and carries transfer flags into connector extra config. |
| RotaSched proactive policy | Sec. 4.2.1-4.2.3 (active rotation, LVF) | `vllm/v1/core/scheduler.py` (`schedule_early`) | `vllm/v1/core/sched/scheduler.py`, `vllm/v1/simple_kv_offload/manager.py` | Partially integrated: `proactive_swap_budget` drives guarded decode-tail/small-decode-backlog proactive preemption and lazy offload scan depth; scheduler-side proactive movement can now preempt multiple guarded candidates until the configured free-block target is reached. The target accounts for immediate waiting-request block pressure above the static budget and is limited by available CPU swap capacity. Lazy offload batches budgeted movement by default; the single-request throttle remains explicit debug-only. Victim choice uses policy fallback by default and VLT scoring when VLT/SLO knobs are enabled. |
| VLT scoring definition | Sec. 4.2.2 (VLT equation and coefficients) | No exact equation implemented; heuristic timing in scheduler | `vllm/v1/core/sched/vlt.py`, `vllm/v1/core/sched/scheduler.py`, `vllm/v1/request.py`, `vllm/v1/simple_kv_offload/manager.py` | Partially active: helper now influences guarded proactive swap victim selection when VLT/SLO knobs are non-default, consumes connector-estimated swap bandwidth when available, and uses request timing hints for TTFT/TBT terms; full policy integration still pending. |
| Request rotary state | Sec. 4.2.1 (transient rotary state) | `RequestStatus.SWAPPED` plus timing fields in `vllm/v1/request.py` | `vllm/v1/request.py` (`RequestRotaryState`) | Metadata state machine implemented with transition guards and status sync. |
| CPU/GPU cache movement | Sec. 4.3.2 (rotation engine + CPU tier) | `vllm/v1/core/kv_cache_manager.py` swap lists + `vllm/v1/swapper/*` | `vllm/v1/simple_kv_offload/*`, `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py` | Using modern simple offload path as the first landing zone; debug single-request swap gate added. |
| Duplex transfer semantics | Sec. 4.3.2 (block-first, batched copies, full-duplex overlap) | `vllm/v1/swapper/native/swapper.cpp` (`swap_block_first`, `cudaMemcpyBatchAsync`) | Existing modern offload connectors/workers | Partially advanced: modern `DmaCopyBackend` uses `cuMemcpyBatchAsync`, separate load/store CUDA streams, and now independent load/store submission queues/threads. `pin_memory_fix`/`swapper_block_first` are propagated through metadata/logging. Native block-first layout and full old DuplexKV semantics are not forward-ported yet. |
| Scheduler/offload telemetry | Sec. 5 metrics and analysis sections | Engine observability in release fork | `vllm/v1/metrics/stats.py`, `vllm/v1/core/sched/scheduler.py`, `vllm/v1/simple_kv_offload/manager.py` | Implemented counters and tested roundtrip aggregation. |
| Engine overlap pipeline | Sec. 4.3.2, Fig. 15 | `vllm/v1/engine/core.py` custom overlap path | Modern `vllm/v1/engine/core.py` + async scheduling | Not ported; high-risk area deferred. |
| Prefix-cache safety fix | Mentioned in release behavior and ablation knobs | `vllm/v1/core/kv_cache_manager.py` | Modern prefix cache internals + connector flow | Initial guarded fix implemented: proactive swap candidates are excluded when allocated non-null blocks have `ref_cnt > 1`, keeping shared prefix-cache blocks pinned. Refcount-aware shared-prefix offload remains pending because modern ownership spans multiple request block lists and cache maps. |

## Known Intentional Divergences

### D-001: VLT equation is not yet scheduler-active

- Why: keep modern behavior stable while validating correctness and telemetry.
- Impact: VLT now only participates in guarded proactive preemption victim choice; baseline scheduling order remains unchanged.
- Exit criteria: extend VLT to full proactive policy and wider scheduling decisions with correctness tests.

### D-002: `swap_cpu_memory_gb` maps to modern connector, not old swapper thread

- Why: modern vLLM already provides offload interfaces and lifecycle hooks.
- Impact: allocator capacity is real and transfer compatibility flags are visible in connector metadata/telemetry; native DuplexKV block-first layout is not yet used.
- Exit criteria: confirm correctness first, then evaluate whether native engine is still required.

### D-006: Full-duplex transfer is implemented at submission-stream level only

- Why: the modern simple offload path already has separate CUDA streams and batched copy calls, but changing CPU KV layout or native kernels is higher risk for DeepSeek-V4.
- Impact: load/store submissions no longer share one Python queue/thread, but CPU block layout remains GPU-derived and not old SuperInfer block-first.
- Exit criteria: benchmark under pressure and decide whether block-first layout or native C++ swapper machinery is still needed.

### D-003: Rotary state is metadata-only for now

- Why: minimize behavior risk while preserving semantic traceability.
- Impact: state exists for observability/guarding but does not yet enforce scheduling policy.
- Exit criteria: hook rotary state transitions into active swap/resume path.

### D-004: Step-6 debug swap gate uses connector-level throttling

- Why: validate swap correctness incrementally without enabling broad proactive movement.
- Impact: when explicitly enabled, store/load planning is restricted to a single request per scheduler step. Proactive swap no longer enables this throttle automatically after guarded DeepSeek-V4 GPU validation.
- Exit criteria: keep as an explicit debugging override only; do not re-enable automatically for validated proactive paths.

### D-005: Proactive budget currently controls scan depth, not full LVF victim choice

- Why: activate bounded proactive movement with minimal scheduler risk while preserving modern request ordering.
- Impact: `proactive_swap_budget` now limits lazy offload blocks per step in `SimpleCPUOffloadScheduler` and enables guarded scheduler preemption for decode-tail plus small decode-backlog requests when free GPU blocks are below budget or the head waiting request's immediate block pressure. Scheduler-side proactive movement can evict multiple guarded candidates until the configured free-block target is reached, but only while available CPU swap capacity can hold each selected candidate. Lazy offload batches budgeted movement by default unless `debug_single_request_swap` is explicitly set. Victim ranking can use VLT weights, with connector-estimated swap bandwidth telemetry as input when present.
- Exit criteria: integrate request-level proactive victim selection across broader candidate sets and scheduling decisions with targeted preemption/resume correctness tests.

## Non-Negotiable Safety Rules for Porting

- Do not blindly copy old `v0.6.6.post1` assumptions into modern KV group logic.
- Treat DeepSeek-V4 MLA/indexer state as potentially unswappable until proven safe.
- Keep prefix-shared blocks conservative unless refcount semantics are audited.
- Require explicit tests for each behavior activation step.
