# SuperInfer Progress and Decision Log

This is a living log for preserving context over long forward-port sessions.

Update this file whenever:

- a behavior changes,
- a semantic divergence from old SuperInfer is introduced,
- or a bug fix requires choosing paper intent vs release implementation.

## Entry Template

```
## [YYYY-MM-DD] <short title>

- Area:
- Trigger:
- Source of truth consulted:
  - Paper:
  - Old release code:
  - Modern constraints:
- Decision:
- Rationale:
- Risk:
- Tests:
- Follow-up:
```

## Entries

## [2026-06-06] Established traceability baseline and authority order

- Area: documentation and debugging workflow
- Trigger: user requested durable source-of-truth linkage to avoid context loss
- Source of truth consulted:
  - Paper: arXiv `2601.20309v2`, semantics in Sections 4.2 and 4.3
  - Old release code: `/workspace/re-SuperInfer/SuperInfer` (`v0.6.6.post1`-based fork)
  - Modern constraints: `docs/030_deepseekv4_cache_state_audit.md`
- Decision: introduce explicit authority order and a traceability matrix in
  `docs/070_source_of_truth_traceability.md`
- Rationale: keep future bug triage deterministic and prevent drift between
  paper semantics, shipped fork heuristics, and modern integration constraints
- Risk: documentation can become stale if not maintained
- Tests: not code-path changing; documentation-only
- Follow-up: append a new entry for every intentional behavior divergence

## [2026-06-06] Enabled real allocator wiring for `swap_cpu_memory_gb`

- Area: KV offload configuration path
- Trigger: move beyond no-op flag plumbing into first real behavior
- Source of truth consulted:
  - Paper: semantic need for CPU swap tier capacity
  - Old release code: swap-space based capacity and swap-enabled scheduler
  - Modern constraints: prefer existing modern offload connectors first
- Decision: map `swap_cpu_memory_gb` to
  `SimpleCPUOffloadConnector` with `cpu_bytes_to_use`, `lazy_offload=True`,
  and `kv_role="kv_both"` in `vllm/config/vllm.py`
- Rationale: achieves real capacity-limited behavior using the modern
  connector lifecycle instead of introducing a parallel swapper subsystem early
- Risk: semantic mismatch versus old native swapper details (acceptable at this stage)
- Tests:
  - `tests/v1/kv_connector/unit/test_config.py::test_swap_cpu_memory_gb_enables_simple_offload_connector`
  - `tests/v1/kv_connector/unit/test_config.py::test_swap_cpu_memory_gb_rejects_non_positive_values`
- Follow-up: Step-6 debug single-request swap path correctness checks

## [2026-06-06] Added Step-6 debug single-request swap gate in modern offload path

- Area: simple CPU offload connector scheduler path
- Trigger: implement minimal-risk active swap behavior for early correctness validation
- Source of truth consulted:
  - Paper: active rotation semantics require controlled swap in/out behavior
  - Old release code: swapper path supports explicit transfer control
  - Modern constraints: use existing `SimpleCPUOffloadConnector` and avoid invasive scheduler changes
- Decision:
  - add `debug_single_request_swap` connector extra-config flag,
  - when enabled, throttle store/load planning to a single request per scheduler step,
  - wire this in `VllmConfig._post_init_kv_transfer_config` when
    `swap_cpu_memory_gb` is set and `proactive_swap_budget > 0`
- Rationale: gives a deterministic, conservative behavior surface for validating
  resume correctness before policy-level proactive swapping is enabled
- Risk: under-utilizes available transfer bandwidth by design in debug mode
- Tests:
  - `tests/v1/kv_connector/unit/test_config.py` verifies flag wiring
  - `tests/v1/simple_kv_offload/test_scheduler.py::test_debug_single_request_swap_limits_store_and_load_to_one_request`
- Follow-up:
  - integrate guarded scheduler policy trigger for real proactive movement
  - expand correctness tests around preemption/resume order

## [2026-06-06] Normalized simple offload test expectations to connector contract

- Area: unit tests for simple offload scheduler
- Trigger: failing assertions expected full-block hit lengths, but connector intentionally reserves the last token for recomputation
- Source of truth consulted:
  - Modern behavior: `get_num_new_matched_tokens` uses `request.num_tokens - 1`
  - Existing tests in file already rely on this behavior in other cases
- Decision: adjust test expectations and load block counts to align with the
  current connector contract for hit length and load planning
- Rationale: ensure new Step-6 debug tests validate intended behavior, not a
  different semantics than the implementation
- Risk: none; test-only consistency update
- Tests:
  - focused simple offload tests rerun and passing for touched cases
- Follow-up: full-file run can still be flaky in this environment due to
  intermittent remote model config lookup (proxy/HF availability)

## [2026-06-06] Activated guarded proactive swap budget in lazy offload path

- Area: simple CPU offload scheduler behavior
- Trigger: continue Step-6 beyond connector-only debug throttling into bounded proactive movement
- Source of truth consulted:
  - Paper: proactive rotation requires explicit movement budget and controlled swap pressure
  - Old release code: `proactive_swap_budget` feeds scheduler-side proactive swap flow (`schedule_early`)
  - Modern constraints: keep request scheduling/preemption semantics stable while validating offload correctness
- Decision:
  - wire `SchedulerConfig.proactive_swap_budget` into
    `SimpleCPUOffloadScheduler` as `_proactive_swap_budget`,
  - in lazy mode, set `_target_free` to proactive budget when budget > 0,
    otherwise keep existing estimated lazy target,
  - expose `offload_lazy_target_free_blocks` and
    `offload_proactive_swap_budget` in telemetry,
  - add unit test coverage for budgeted lazy scan depth.
- Rationale: this enables real, bounded proactive movement with minimal risk,
  because it reuses existing lazy scanner mechanics and avoids immediate changes
  to modern scheduler request ordering.
- Risk: behavior still differs from full paper/release LVF request-level victim
  selection; budget currently limits block scan depth only.
- Tests:
  - `tests/v1/simple_kv_offload/test_scheduler.py::test_lazy_proactive_swap_budget_overrides_scan_depth`
- Follow-up:
  - integrate request-level LVF/VLT proactive victim selection in
    `vllm/v1/core/sched/scheduler.py`
  - add targeted preemption/resume tests across proactive-budget paths

## [2026-06-06] Added scheduler-side guarded proactive preemption trigger

- Area: modern scheduler proactive swap activation (`vllm/v1/core/sched/scheduler.py`)
- Trigger: continue Step-6 from connector/scan-budget-only behavior into request-level movement under strict gates
- Source of truth consulted:
  - Paper: proactive rotation should actively free room before contention stalls
  - Old release code: `schedule_early` performs request-level proactive preemption/swap
  - Modern constraints: preserve default scheduling semantics and avoid broad policy churn before LVF/VLT integration
- Decision:
  - add `_maybe_preempt_for_proactive_swap` guarded path at the start of `schedule()`;
  - activate only when all of the following are true:
    - `proactive_swap_budget > 0`,
    - scheduler is unpaused,
    - waiting work exists,
    - connector is `SimpleCPUOffloadConnector` in lazy mode,
    - no pending offload transfers,
    - free GPU blocks are below budget,
    - a decode-tail running request exists (`num_tokens - num_computed_tokens == 1`),
  - choose one victim using existing policy ordering (priority/arrival for priority mode; tail for fcfs),
  - preempt exactly one request and defer its immediate re-scheduling within the same step.
- Rationale: this provides real scheduler-level proactive movement while keeping scope narrow and deterministic.
- Risk: victim choice is still not LVF/VLT-derived; only decode-tail candidates are considered in this guarded phase.
- Tests:
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_preempts_decode_tail_in_simple_offload_mode`
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_preempt_sets_rotary_state_waiting`
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_skips_when_transfers_pending`
- Follow-up:
  - replace guarded victim selection with LVF/VLT ranking,
  - broaden candidate eligibility after additional correctness gates.

## [2026-06-06] Enabled VLT helper in guarded proactive victim selection path

- Area: scheduler proactive victim choice (`vllm/v1/core/sched/scheduler.py`)
- Trigger: progress from policy-order-only victim selection toward paper-aligned VLT usage without broad scheduling risk
- Source of truth consulted:
  - Paper: VLT weights combine SLO overflow, swap-cost, and future delay terms
  - Old release code: proactive flow uses heuristic victim selection under swap budget pressure
  - Modern constraints: keep VLT activation bounded to guarded proactive path first
- Decision:
  - add `_select_proactive_swap_victim` that uses `compute_vlt_score` when any
    VLT/SLO knob is active,
  - estimate candidate swap size from current request block residency,
  - keep fallback victim selection unchanged when VLT knobs are default,
  - keep scope limited to proactive decode-tail candidates.
- Rationale: this introduces real VLT-driven behavior while containing blast radius
  to the already guarded proactive preemption path.
- Risk: current timing/swap estimates are conservative approximations and not yet
  calibrated against production telemetry.
- Tests:
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_vlt_weights_choose_small_swap_candidate`
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_preempts_decode_tail_in_simple_offload_mode`
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_skips_when_transfers_pending`
- Follow-up:
  - calibrate VLT inputs using richer runtime telemetry,
  - wire VLT ranking into broader proactive policy once correctness gates are complete.

## [2026-06-06] Expanded proactive preemption coverage for fallback and VLT paths

- Area: scheduler proactive preemption test coverage
- Trigger: verify both default and VLT-enabled victim-selection branches stay stable during ongoing Step-6 rollout
- Source of truth consulted:
  - Modern scheduler behavior in `vllm/v1/core/sched/scheduler.py`
  - Existing priority/preemption contracts in `tests/v1/core/test_scheduler.py`
- Decision:
  - add focused tests for:
    - default FCFS fallback victim selection,
    - default priority fallback victim selection,
    - VLT-bandwidth-weighted victim selection,
    - pending-transfer guard and rotary-state transition invariants.
- Rationale: ensures guarded proactive path remains deterministic and debuggable
  as policy internals evolve.
- Risk: limited to test-only changes and fixture helpers; multimodal test paths remain environment-dependent in this container.
- Tests:
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_default_fcfs_preempts_running_tail_candidate`
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_default_priority_preempts_lower_priority_candidate`
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_vlt_weights_choose_small_swap_candidate`
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_skips_when_transfers_pending`

## [2026-06-06] Added connector bandwidth telemetry for guarded proactive VLT scoring

- Area: simple offload telemetry and scheduler proactive VLT input calibration
- Trigger: continue Step-9 by replacing fixed swap bandwidth heuristic with runtime-informed signal while keeping guarded scope unchanged
- Source of truth consulted:
  - Paper: VLT includes swap transfer cost term that should reflect actual bandwidth conditions
  - Old release code: proactive heuristics depended on runtime system state but lacked exact modern VLT equation integration
  - Modern constraints: keep proactive path gated to decode-tail candidates and avoid broad scheduling policy churn
- Decision:
  - record per-event transfer start timestamps and bytes in `SimpleCPUOffloadScheduler`,
  - maintain EMA estimate for swap bandwidth from completed store/load events,
  - expose estimate via connector API and telemetry key `offload_estimated_swap_bandwidth_bytes_per_s`,
  - consume this estimate in guarded proactive VLT victim scoring; fall back to default bandwidth when unavailable.
- Rationale: improves VLT swap-cost realism with low-risk, connector-local instrumentation and preserves deterministic fallback behavior.
- Risk: event timing is step-granular and can include scheduler/worker overhead; estimate remains approximate.
- Tests:
  - `tests/v1/simple_kv_offload/test_scheduler.py::test_simple_offload_estimated_swap_bandwidth_updates`
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_vlt_uses_connector_bandwidth_estimate`
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_vlt_weights_choose_small_swap_candidate`
- Follow-up:
  - incorporate latency-side signals (TTFT/TBT deltas) into proactive VLT inputs,
  - validate estimate stability across multi-request and mixed load/store phases.

## [2026-06-06] Added request timing signals to guarded proactive VLT TTFT/TBT inputs

- Area: scheduler proactive VLT latency-term calibration
- Trigger: continue Step-9 incremental rollout by replacing purely time-in-system TTFT/TBT approximations with request timing hints
- Source of truth consulted:
  - Paper: VLT objective combines SLO overflow terms for TTFT and TBT
  - Old release code: latency-sensitive heuristics exist but without exact modern VLT helper wiring
  - Modern constraints: keep proactive scope guarded (decode-tail candidates only) and avoid non-local scheduling changes
- Decision:
  - add per-request generation timing hints (`first_generated_token_ts`, `last_generated_token_ts`) in `Request`,
  - update timing hints on each scheduler output token append,
  - derive proactive VLT inputs as:
    - TTFT from first-token timestamp when available (otherwise time-in-system for pre-first-token requests),
    - TBT from observed inter-token window when multiple output tokens exist, with conservative fallback for single-token decode.
- Rationale: improves SLO-term fidelity for guarded proactive victim ranking while preserving fallback semantics and existing guards.
- Risk: timing hints are scheduler-local monotonic timestamps; they are approximate under asynchronous/parallel execution.
- Tests:
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_vlt_slo_ttft_keeps_older_request_running`
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_vlt_slo_tbt_uses_token_timing_estimates`
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_vlt_uses_connector_bandwidth_estimate`
- Follow-up:
  - validate behavior under async scheduling and PP paths,
  - evaluate adding smoothed per-request TBT estimates for noisy short sequences.

## [2026-06-06] Covered generation timing hints in async/empty-output scheduler paths

- Area: scheduler timing-hint regression coverage
- Trigger: validate VLT timing hints do not regress async scheduling or chunked prefill behavior
- Source of truth consulted:
  - Modern constraints: async/PP placeholder scheduling and empty prefill outputs must preserve existing behavior
- Decision:
  - add a focused async+PP test proving generated output records timing hints even when output placeholders are present,
  - add a focused chunked-prefill test proving empty sampled outputs do not set generation timing hints.
- Rationale: these are the highest-risk edge cases for the newly added request timing metadata.
- Risk: test-only; no behavior change in this entry.
- Tests:
  - `tests/v1/core/test_scheduler.py::test_generation_timing_updates_with_async_pp_output_placeholders`
  - `tests/v1/core/test_scheduler.py::test_generation_timing_ignores_empty_prefill_outputs`

## [2026-06-06] Broadened guarded proactive candidates to small decode backlog

- Area: scheduler proactive swap candidate eligibility
- Trigger: continue CPU-testable Step-9 rollout while GPUs are unavailable for output comparison
- Source of truth consulted:
  - Paper: proactive rotation should free room before contention stalls
  - Old release code: proactive scheduler flow considers broader victim sets than exact decode-tail
  - Modern constraints: keep unsafe cases excluded and avoid prefill/spec/async-placeholder disruption
- Decision:
  - keep existing guarded proactive prerequisites unchanged,
  - keep exact decode-tail candidates eligible,
  - additionally allow running decode candidates with generated output and at most one block of remaining work (`remaining_tokens <= block_size`),
  - continue excluding candidates with speculative tokens, output placeholders, prefill chunks, or no generated output for multi-token backlog.
- Rationale: expands proactive usefulness incrementally without touching prefill or active async/spec paths.
- Risk: still not full LVF/RotaSched policy; small decode-backlog eligibility is a conservative approximation.
- Tests:
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_preempts_small_decode_backlog_candidate`
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_skips_small_prefill_backlog_candidate`
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_candidate_excludes_spec_and_placeholder_work`
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_preempts_decode_tail_in_simple_offload_mode`
- Follow-up:
  - add broader priority/VLT tests over mixed exact-tail and small-backlog candidates,
  - keep GPU output comparison blocked until GPU resources are available.

## [2026-06-06] Added mixed-shape proactive victim selection coverage

- Area: scheduler proactive swap tests
- Trigger: verify fallback and VLT victim selection remain deterministic after adding small decode-backlog candidates
- Source of truth consulted:
  - Modern constraints: default policy semantics must remain stable when VLT/SLO knobs are disabled
  - Current guarded proactive path in `vllm/v1/core/sched/scheduler.py`
- Decision:
  - add priority fallback coverage for mixed exact-tail and small decode-backlog candidates,
  - add VLT bandwidth coverage showing scoring can choose a smaller-swap small-backlog candidate over FCFS tail order.
- Rationale: mixed candidate shapes are the first place where the broadened eligibility could accidentally bypass policy/VLT ranking assumptions.
- Risk: test-only; no behavior change in this entry.
- Tests:
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_default_priority_handles_mixed_candidate_shapes`
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_vlt_bandwidth_handles_mixed_candidate_shapes`

## [2026-06-06] Closed CPU-testable guarded proactive validation slice

- Area: CPU-only SuperInfer modern validation handoff
- Trigger: continue implementation while GPUs are unavailable, then bound what can be considered done before runtime validation
- Source of truth consulted:
  - Modern constraints: default-off behavior and scheduler/offload invariants must be unit-testable without GPU
  - Current implementation plan: Step 8 remains GPU-blocked; Step 9 guarded scope is CPU-testable
- Decision:
  - add default-off config coverage proving SuperInfer knobs do not enable KV transfer by default,
  - add config coverage proving `swap_cpu_memory_gb` overrides connector selection while preserving extra config,
  - add proactive preempt/resume sequencing coverage after an intervening request finishes,
  - add `docs/080_gpu_validation_checklist.md` as the final runtime-validation handoff.
- Rationale: reduce the GPU phase to validation and targeted bug fixes instead of broad implementation work.
- Risk: GPU-only worker copy, CUDA graph, model-output, and DeepSeek-V4 cache-layout issues can still appear later.
- Tests:
  - `tests/v1/kv_connector/unit/test_config.py::test_superinfer_knobs_default_off_do_not_enable_kv_connector`
  - `tests/v1/kv_connector/unit/test_config.py::test_swap_cpu_memory_gb_overrides_existing_connector_preserving_extra_config`
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_preempted_request_can_resume_after_intervening_finish`
  - full `tests/v1/simple_kv_offload/test_scheduler.py` CPU file
- Follow-up:
  - execute `docs/080_gpu_validation_checklist.md` when GPU resources are available,
  - defer broader LVF/DuplexKV expansion until GPU smoke tests pass.

## [2026-06-06] Corrected partial GPU-prefix CPU-load test contract

- Area: simple CPU offload scheduler tests
- Trigger: full offload test file exposed a stale expectation in partial GPU prefix + CPU load coverage
- Source of truth consulted:
  - Modern connector contract: `get_num_new_matched_tokens` reserves one token for local recomputation (`request.num_tokens - 1`)
- Decision: update expected CPU hit blocks from 4 to 3 when two local GPU blocks are already computed in a six-block request.
- Rationale: after reserving the final token, only three full CPU blocks are eligible for async load from that position.
- Risk: test-only; aligns with existing connector behavior.
- Tests:
  - `tests/v1/simple_kv_offload/test_scheduler.py::test_partial_gpu_prefix_plus_cpu_load`
  - full `tests/v1/simple_kv_offload/test_scheduler.py`

## [2026-06-06] Added explicit proactive no-op guard coverage

- Area: scheduler proactive swap guard tests
- Trigger: make CPU-testable guard behavior explicit before GPU validation
- Source of truth consulted:
  - Modern constraints: proactive preemption must not run unless pressure and waiting work both exist
  - Current guarded proactive path in `vllm/v1/core/sched/scheduler.py`
- Decision:
  - add coverage proving proactive preemption skips when there is no waiting work,
  - add coverage proving proactive preemption skips when current free GPU blocks already satisfy the proactive budget.
- Rationale: these conditions are core safety gates and should be regression-tested directly, not only inferred from other tests.
- Risk: test-only; no behavior change.
- Tests:
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_skips_when_no_waiting_work`
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_skips_when_free_blocks_meet_budget`
  - full proactive/timing subset (`tests/v1/core/test_scheduler.py -k "test_proactive_swap or test_generation_timing"`)

## [2026-06-06] Added proactive activation-gate coverage

- Area: scheduler proactive swap activation guards
- Trigger: make remaining activation requirements explicit before GPU validation
- Source of truth consulted:
  - Modern constraints: proactive movement must remain opt-in and connector-specific
  - Current guarded proactive path in `vllm/v1/core/sched/scheduler.py`
- Decision:
  - add coverage proving proactive swap skips when `proactive_swap_budget == 0`,
  - add coverage proving proactive swap skips without `SimpleCPUOffloadConnector`,
  - add coverage proving proactive swap skips while scheduler pause state is not `UNPAUSED`.
- Rationale: these guards prevent accidental baseline behavior changes and unsafe movement outside the validated connector path.
- Risk: test-only; no behavior change.
- Tests:
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_skips_when_budget_disabled`
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_skips_without_simple_offload_connector`
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_skips_when_scheduler_paused`
  - full proactive/timing subset (`tests/v1/core/test_scheduler.py -k "test_proactive_swap or test_generation_timing"`)

## [2026-06-06] Added lazy-offload activation-gate coverage

- Area: scheduler proactive swap connector-mode guard
- Trigger: ensure `SimpleCPUOffloadConnector` alone is not enough to activate proactive preemption
- Source of truth consulted:
  - Modern constraints: proactive movement is validated only for the lazy offload path
  - Current guard in `_maybe_preempt_for_proactive_swap`
- Decision:
  - extend the proactive test fixture to allow `lazy_offload=False`,
  - add coverage proving eager/simple offload mode skips proactive preemption even when budget, waiting work, and a decode-tail candidate exist.
- Rationale: prevents accidentally enabling request-level proactive movement in an unvalidated connector mode.
- Risk: test-only; no behavior change.
- Tests:
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_skips_when_simple_offload_not_lazy`
  - full proactive/timing subset (`tests/v1/core/test_scheduler.py -k "test_proactive_swap or test_generation_timing"`)

## [2026-06-06] Added CLI default-off and VLT future-delay coverage

- Area: CLI/config invariants and scheduler VLT victim selection
- Trigger: keep GPU-blocked validation phase small by hardening CPU-testable default and VLT paths
- Source of truth consulted:
  - Modern constraints: SuperInfer knobs must remain off unless explicitly requested
  - Paper: VLT includes a future-delay term in addition to SLO and bandwidth terms
- Decision:
  - add engine-args coverage proving SuperInfer CLI defaults remain disabled,
  - add scheduler coverage proving `vlt_beta_future` can preserve older requests by selecting a newer victim in the guarded proactive path.
- Rationale: protects baseline behavior and validates the last VLT term at scheduler integration level.
- Risk: test-only; no behavior change.
- Tests:
  - `tests/engine/test_arg_utils.py::test_superinfer_cli_defaults_are_disabled`
  - `tests/engine/test_arg_utils.py::test_superinfer_noop_flags_from_cli`
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_vlt_future_delay_keeps_older_request_running`
  - full proactive/timing subset (`tests/v1/core/test_scheduler.py -k "test_proactive_swap or test_generation_timing"`)

## [2026-06-06] Added skipped-waiting proactive pressure coverage

- Area: scheduler proactive swap queue-pressure guard
- Trigger: verify proactive movement sees blocked/skipped waiting work, not only the normal waiting queue
- Source of truth consulted:
  - Modern scheduler queue model: waiting work can live in `waiting` or `skipped_waiting`
  - Current guarded proactive path checks both queues before candidate selection
- Decision:
  - add coverage where the only pending work is `WAITING_FOR_REMOTE_KVS` in `skipped_waiting`,
  - verify proactive preemption still runs and does not lose the blocked request.
- Rationale: this protects the connector-interaction path where waiting work may be temporarily blocked by remote KV dependencies.
- Risk: test-only; no behavior change.
- Tests:
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_runs_with_only_skipped_waiting_work`
  - full proactive/timing subset (`tests/v1/core/test_scheduler.py -k "test_proactive_swap or test_generation_timing"`)

## [2026-06-06] Pinned shared prefix-cache blocks from proactive swap candidates

- Area: scheduler proactive swap safety guard
- Trigger: finish prefix-cache/shared-block CPU safety audit before GPU validation
- Source of truth consulted:
  - Paper: proactive rotation should preserve cache correctness while freeing capacity
  - Old release code: prefix cache and swap code used block refcounts around cached blocks
  - Modern constraints: `docs/030_deepseekv4_cache_state_audit.md` requires shared prefix blocks to remain unswappable until refcount-aware offload is validated
- Decision:
  - add a scheduler-side candidate guard that rejects proactive swap candidates holding any allocated non-null KV block with `ref_cnt > 1`,
  - keep unshared cached blocks eligible under the existing strict proactive gates.
- Rationale: modern prefix-cache hits increment block refcounts when another request reuses a cached prefix; freeing/preempting such a request without refcount-aware offload can disturb shared ownership assumptions.
- Risk: conservative guard may reduce proactive swap opportunities for workloads with heavy prefix sharing.
- Tests:
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_skips_shared_prefix_cache_blocks`
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_preempts_decode_tail_in_simple_offload_mode`
- Follow-up:
  - GPU runtime validation remains required before broader candidate expansion or refcount-aware shared-block offload.

## [2026-06-06] Closed CPU forward-port phase before GPU validation

- Area: CPU-only SuperInfer modern forward-port checkpoint
- Trigger: user requested a clean stop point so the next session can resume directly with GPU validation when hardware is available
- Source of truth consulted:
  - Paper: SuperInfer semantics require runtime swap and proactive scheduling validation
  - Old release code: full parity includes RotaSched/LVF and DuplexKV paths not yet validated on modern GPU runtime
  - Modern constraints: DeepSeek-V4/cache safety requires GPU output and CUDA KV-copy validation before broader policy expansion
- Decision:
  - mark the guarded CPU-testable scheduler/offload slice complete,
  - freeze CPU-side behavior scope until GPU validation,
  - defer full LVF/RotaSched, DuplexKV/native transfer, refcount-aware shared-prefix offload, and performance benchmarking.
- Rationale: all currently safe CPU-doable work is covered by focused tests; additional CPU expansion would add risk without proving generated-output or CUDA cache correctness.
- Risk: GPU-only issues may still appear in worker copy paths, CUDA graphs, DeepSeek-V4 MLA layout, or model-output equivalence.
- Tests:
  - `tests/v1/core/test_vlt.py tests/v1/test_request.py tests/v1/kv_connector/unit/test_config.py -vv` -> 22 passed
  - `tests/v1/core/test_scheduler.py -k "test_proactive_swap or test_generation_timing or test_priority_scheduling_preemption or test_preempt_during_execution" -vv` -> 31 passed
  - `tests/v1/simple_kv_offload/test_scheduler.py -vv` -> 16 passed
  - `tests/engine/test_arg_utils.py -k "test_superinfer_cli_defaults_are_disabled or test_superinfer_noop_flags_from_cli" -vv` -> 2 passed
- Follow-up:
  - resume with `docs/080_gpu_validation_checklist.md` when GPU resources are available.

## [2026-06-07] Completed first GH200 GPU smoke validation

- Area: GPU runtime validation for guarded SuperInfer scheduler/offload path
- Trigger: GPU became available after CPU phase closure
- Source of truth consulted:
  - GPU handoff: `docs/080_gpu_validation_checklist.md`
  - Modern constraints: keep validation conservative before broader policy expansion
- Decision:
  - repair the container GPU Python stack with `torch==2.11.0+cu130`,
  - build/install `vllm` editable from source because proxy policy blocked `wheels.vllm.ai`,
  - validate baseline, conservative swap-on, guarded proactive/VLT, and DeepSeek-V4 CUDA readiness tests on GH200.
- Rationale: prove the CPU-forwarded guarded path can initialize and complete real CUDA inference before attempting DeepSeek-V4 model-output validation.
- Risk: full DeepSeek-V4-Flash output correctness remains unvalidated without model weights.
- Tests:
  - baseline `facebook/opt-125m` deterministic GPU generation with SuperInfer knobs off -> passed
  - swap-on `facebook/opt-125m` smoke with `swap_cpu_memory_gb=1.0`, `num_gpu_blocks_override=8` -> passed
  - proactive/VLT `facebook/opt-125m` smoke with `proactive_swap_budget=4`, `vlt_beta_bandwidth=1.0` -> passed
  - `tests/models/test_deepseek_v4_mega_moe.py tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py -vv` -> 41 passed
- Follow-up:
  - obtain/access DeepSeek-V4-Flash weights and run full generated-output plus swap/proactive validation before policy expansion.

## [2026-06-07] Completed DeepSeek-V4-Flash guarded GPU validation

- Area: DeepSeek-V4-Flash GPU runtime validation for guarded SuperInfer paths
- Trigger: local DeepSeek-V4-Flash weights became available and the first guarded proactive run hit CUDA OOM due to limited shared-GPU headroom
- Source of truth consulted:
  - GPU handoff: `docs/080_gpu_validation_checklist.md`
  - Modern constraints: DeepSeek-V4 requires `kv_cache_dtype=fp8` and conservative cache/offload gates
- Decision:
  - validate DeepSeek-V4-Flash baseline generation, conservative swap-on, and guarded proactive/VLT smoke with local weights,
  - keep `tensor_parallel_size=2`, `dtype=bfloat16`, `kv_cache_dtype=fp8`, `max_model_len=512`, and `enforce_eager=True`,
  - lower guarded proactive `gpu_memory_utilization` from `0.92` to `0.84` when another user's process occupied GPU 0 memory,
  - stop before broader LVF/RotaSched, DuplexKV/native transfer, refcount-aware shared-prefix offload, or performance benchmarking.
- Rationale: the guarded implementation now has generated-output validation on the target DeepSeek-V4 family without expanding policy scope beyond the CPU-validated gates.
- Risk: this is smoke/correctness validation, not a throughput or stress benchmark; proactive movement remains intentionally conservative.
- Tests:
  - baseline DeepSeek-V4-Flash generated-output smoke with SuperInfer knobs off -> passed
  - conservative DeepSeek-V4-Flash swap-on smoke with `swap_cpu_memory_gb=4.0` -> passed
  - guarded DeepSeek-V4-Flash proactive/VLT smoke with `swap_cpu_memory_gb=4.0`, `proactive_swap_budget=4`, `vlt_beta_bandwidth=1.0`, `max_num_seqs=1`, `prompt_count=2`, `gpu_memory_utilization=0.84` -> passed
- Follow-up:
  - defer broader policy/performance expansion until explicitly requested.

## [2026-06-08] Added budget-saturating guarded proactive preemption

- Area: RotaSched/LVF scheduler policy integration
- Trigger: continue implementation after guarded DeepSeek-V4 GPU validation passed
- Source of truth consulted:
  - Paper: RotaSched active rotation keeps a proactive free-block budget for incoming work
  - Old release code: `schedule_early` can swap out more than one running decode-tail request while free GPU blocks are below `proactive_swap_budget`
  - Modern constraints: keep candidate eligibility, connector mode, pending-transfer, and shared-prefix safety gates intact
- Decision:
  - change scheduler-side proactive movement from exactly one victim per step to a budget-saturating loop,
  - keep selecting victims through the existing fallback/VLT ranking path,
  - stop as soon as the block pool reaches the target free-block budget or no safe candidates remain.
- Rationale: this is the smallest RotaSched/LVF expansion that matches the old proactive-budget intent without broadening unsafe candidate eligibility or introducing native DuplexKV transfer changes.
- Risk: a single scheduler step can now preempt multiple requests under memory pressure, increasing movement aggressiveness while still bounded by `proactive_swap_budget` and strict candidate gates.
- Tests:
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_preempts_multiple_candidates_until_budget_met` -> passed
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_multi_vlt_keeps_slo_violating_request_running` -> passed
  - `tests/v1/core/test_scheduler.py -k "test_proactive_swap or test_generation_timing" -vv` -> 28 passed
- Follow-up:
  - validate the expanded proactive behavior with GPU smoke before additional candidate broadening or DuplexKV/native transfer work.

## [2026-06-08] Relaxed automatic debug swap throttling after guarded validation

- Area: SimpleCPUOffloadConnector configuration and lazy transfer planning
- Trigger: budget-saturating proactive preemption can now select multiple victims, but `swap_cpu_memory_gb` previously enabled `debug_single_request_swap` automatically whenever `proactive_swap_budget > 0`
- Source of truth consulted:
  - Paper: active rotation should move enough KV blocks to satisfy the proactive budget
  - Old release code: proactive swap budget is not a single-request debug throttle
  - Modern constraints: keep debug throttling available as an explicit override, but do not silently restrict validated proactive movement
- Decision:
  - stop automatically setting `debug_single_request_swap=True` for proactive swap,
  - preserve an explicitly supplied `debug_single_request_swap` extra config,
  - keep lazy offload batching capped by `proactive_swap_budget` and CPU capacity.
- Rationale: after CPU and DeepSeek-V4 smoke validation, the default proactive path should exercise budgeted batched movement rather than the earlier Step-6 single-request debug surface.
- Risk: proactive swap can schedule larger store/load events by default; movement is still bounded by the configured budget and existing safe-candidate gates.
- Tests:
  - `tests/v1/kv_connector/unit/test_config.py::test_swap_cpu_memory_gb_with_proactive_budget_keeps_batched_swaps_enabled`
  - `tests/v1/kv_connector/unit/test_config.py::test_swap_cpu_memory_gb_overrides_existing_connector_preserving_extra_config`
  - `tests/v1/simple_kv_offload/test_scheduler.py::test_lazy_proactive_swap_budget_batches_by_default`
  - `tests/v1/simple_kv_offload/test_scheduler.py::test_lazy_debug_single_request_swap_still_limits_proactive_budget`
  - `tests/v1/core/test_scheduler.py -k "test_proactive_swap or test_generation_timing" -vv` -> 28 passed
  - small `facebook/opt-125m` GPU proactive smoke with `debug_single_request_swap=False` -> passed
  - DeepSeek-V4-Flash guarded proactive/VLT GPU smoke with `debug_single_request_swap=False` -> passed
- Follow-up:
  - do not broaden candidate classes or start native DuplexKV transfer until this batched default is stress-tested under higher pressure.

## [2026-06-08] Added waiting-pressure and CPU-capacity bounds to proactive target

- Area: RotaSched/LVF proactive free-block target and CPU swap-capacity safety
- Trigger: continue closing the gap between guarded proactive movement and old `schedule_early` behavior without introducing native DuplexKV
- Source of truth consulted:
  - Paper: active rotation should make room for incoming work before allocation stalls
  - Old release code: `schedule_early` checks waiting pressure and CPU cache usage before swapping out running requests
  - Modern constraints: avoid side effects while estimating pressure; never select a proactive victim that cannot fit in the CPU swap tier
- Decision:
  - estimate the head waiting request's immediate block pressure from schedulable tokens and use it as a floor above `proactive_swap_budget`,
  - expose scheduler-side free CPU block count through `SimpleCPUOffloadConnector`,
  - skip proactive movement when the CPU swap pool is full,
  - stop selecting more victims once remaining CPU block capacity cannot hold another candidate's KV blocks.
- Rationale: this keeps proactive movement useful for larger waiting prompts while preventing GPU preemption from degrading into recompute-only behavior when CPU swap capacity is exhausted.
- Risk: pressure estimation intentionally ignores prefix-cache hits to avoid mutating cache stats or connector state before normal waiting scheduling; this can overestimate immediate demand conservatively.
- Tests:
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_uses_waiting_block_pressure_above_static_budget` -> passed
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_skips_when_cpu_swap_pool_is_full` -> passed
  - `tests/v1/core/test_scheduler.py::test_proactive_swap_limits_victims_to_cpu_swap_capacity` -> passed
  - `tests/v1/core/test_scheduler.py -k "test_proactive_swap or test_generation_timing" -vv` -> 31 passed
  - `tests/v1/simple_kv_offload/test_scheduler.py -vv` -> 19 passed
  - pressure-oriented `facebook/opt-125m` GPU smoke with `proactive_swap_budget=1` -> passed
- Follow-up:
  - keep shared-prefix blocks pinned until refcount-aware offload is implemented and stress-tested.

## [2026-06-08] Deferred refcount-aware shared-prefix offload from guarded path

- Area: prefix-cache/shared-block offload safety
- Trigger: user requested continuing toward final implementation after RotaSched/offload GPU validation
- Source of truth consulted:
  - Modern constraints: shared prefix-cache blocks are represented by `ref_cnt > 1` blocks referenced by multiple request block lists and cache maps
  - Current connector assumptions: store/load planning is request-sequence-oriented and does not yet track all sharers of a CPU-resident shared block
- Decision:
  - do not remove the shared-prefix proactive candidate guard in this slice,
  - keep `ref_cnt > 1` blocks pinned for proactive swap,
  - defer true refcount-aware shared-prefix offload to a separate design/implementation phase.
- Rationale: safely offloading shared blocks requires ownership metadata and resume behavior that spans every request sharing the prefix. A local scheduler guard removal would risk cache aliasing or premature GPU block release.
- Risk: workloads with heavy prefix sharing get fewer proactive swap opportunities until shared-prefix offload is implemented.
- Tests:
  - existing `tests/v1/core/test_scheduler.py::test_proactive_swap_skips_shared_prefix_cache_blocks` remains part of the proactive subset -> passed
- Follow-up:
  - design shared-block CPU residency metadata before changing prefix-cache ownership behavior.

## [2026-06-08] Exposed transfer compatibility flags without changing layout

- Area: SimpleCPUOffloadConnector transfer metadata and worker observability
- Trigger: continue worker/DuplexKV gap closure by wiring `pin_memory_fix` and `swapper_block_first` into validation surfaces before changing native copy/layout behavior
- Source of truth consulted:
  - Paper: DuplexKV section calls out pinned CPU memory, batched copies, and block-first layout as transfer-engine optimizations
  - Old release code: `Swapper` uses `swap_block_first` for CPU mirror layout and `cudaHostRegister` for the large pinned-memory workaround
  - Modern constraints: avoid rewriting copy kernels or KV layout while DeepSeek-V4 correctness depends on existing backend layout assumptions
- Decision:
  - propagate `pin_memory_fix` and `swapper_block_first` from `CacheConfig` into `SimpleCPUOffloadConnector` extra config when `swap_cpu_memory_gb` is set,
  - carry both flags in per-step `SimpleCPUOffloadMetadata` for worker-side validation,
  - expose both flags in scheduler telemetry stats,
  - log worker pinning/layout mode and warn when `swapper_block_first` is requested because the modern worker still uses GPU-derived CPU KV layout.
- Rationale: this closes the no-op observability gap and makes future transfer/layout validation explicit without risking a silent cache-layout change.
- Risk: `swapper_block_first=True` is still not behaviorally equivalent to old SuperInfer; it is now visible and warned, not implemented.
- Tests:
  - `tests/v1/simple_kv_offload/test_scheduler.py -q` -> 20 passed
  - `tests/v1/kv_connector/unit/test_config.py -k "swap_cpu_memory_gb" -q` -> 5 passed
  - `facebook/opt-125m` GPU smoke with `swap_cpu_memory_gb=1.0`, `pin_memory_fix=True`, `swapper_block_first=True` -> passed; logs showed `pin_strategy=cudaHostRegister` and `cpu_layout=gpu_derived`
- Follow-up:
  - continue inspecting whether modern `DmaCopyBackend` needs a true block-first CPU layout or additional full-duplex scheduling hooks before changing copy kernels.

## [2026-06-08] Split DmaCopyBackend load/store submission workers

- Area: SimpleCPUOffloadConnector worker-side transfer backend
- Trigger: compare modern `DmaCopyBackend` against old SuperInfer DuplexKV transfer semantics after flag-visibility wiring
- Source of truth consulted:
  - Paper: DuplexKV overlaps H2D and D2H movement for active rotation
  - Old release code: `swapper/native/swapper.cpp` submits D2H and H2D batches onto separate CUDA streams
  - Modern constraints: preserve current per-layer block views and DeepSeek-V4-safe GPU-derived CPU KV layout
- Decision:
  - split `DmaCopyBackend` from one shared Python submission queue/thread into independent load and store queues/threads,
  - keep the existing `cuMemcpyBatchAsync` helper, per-direction CUDA streams, event reporting, and CPU cache layout unchanged.
- Rationale: this is the smallest safe full-duplex improvement: load/store submissions can proceed independently before reaching their already-separate CUDA streams, without native-kernel or layout risk.
- Risk: runtime performance benefit is workload-dependent because actual overlap depends on CUDA copy-engine behavior and transfer pressure; this is not a replacement for old block-first layout.
- Tests:
  - `tests/v1/simple_kv_offload/test_copy_backend.py -q` -> 1 passed
  - `tests/v1/simple_kv_offload/test_scheduler.py -q` -> 20 passed
  - `facebook/opt-125m` GPU smoke with `swap_cpu_memory_gb=1.0`, `pin_memory_fix=True`, `swapper_block_first=True` -> passed; logs showed `DmaCopyBackend: dual-thread batched async copy backend enabled`
- Follow-up:
  - run swap-on GPU smoke and then pressure benchmarks before considering native block-first layout changes.

## [2026-06-08] Produced block-first/native transfer implementation blueprint

- Area: DuplexKV parity design planning
- Trigger: user requested "design spike for block-first/native transfer parity" as the next highest-impact item
- Source of truth consulted:
  - Paper: DuplexKV transfer semantics (block-first layout, batched H2D/D2H, overlap)
  - Old release code: `vllm/v1/swapper/swapper.py` and `vllm/v1/swapper/native/*`
  - Modern constraints: `simple_kv_offload` worker/copy path and DeepSeek-V4 cache safety audit
- Decision:
  - add `docs/090_block_first_native_design_spike.md` describing:
    - current modern capabilities and remaining parity gaps,
    - explicit layout-descriptor landing zone,
    - staged rollout for block-first enablement with strict model-family gating,
    - benchmark-driven decision gate before introducing a native backend.
- Rationale: this provides a concrete, low-risk execution plan so future implementation slices can be incremental and test-gated instead of ad-hoc native/layout rewrites.
- Risk: design-only output; no direct performance or parity gains until staged implementation begins.
- Tests: documentation-only; no runtime behavior changes in this step.
- Follow-up:
  - implement Stage 1 (layout descriptor + copy API refactor) with default behavior unchanged.

## [2026-06-08] Ran first quick pressure throughput pair for swap vs proactive/VLT

- Area: performance sanity check (pre-benchmark-phase)
- Trigger: user asked about the pressure benchmark while implementation was ongoing
- Source of truth consulted:
  - Benchmark policy: `docs/060_benchmark_plan.md`
  - Modern constraints: keep runs small and deterministic enough for rapid feedback on shared GH200
- Decision:
  - run a compact offline pair with `vllm bench throughput` on `facebook/opt-125m`:
    - swap baseline,
    - swap + proactive/VLT.
  - use `num_gpu_blocks_override=16` and `max_model_len=128` after an initial failed attempt (`max_model_len=512` with override `8`) reported insufficient KV capacity.
- Rationale: produce immediate pressure-sanity telemetry while avoiding long blocking runs that look stuck when effective concurrency collapses.
- Risk: this is a single-run throughput sanity check; it does not include TTFT/TBT serving metrics or statistical confidence.
- Tests:
  - swap baseline quick run -> `9.352 req/s`, `748.168 tok/s`
  - proactive/VLT quick run -> `8.145 req/s`, `651.575 tok/s`
  - raw artifacts:
    - `docs/benchmarking/superinfer_pressure_swap_baseline_opt125m_quick.json`
    - `docs/benchmarking/superinfer_pressure_swap_proactive_vlt_opt125m_quick.json`
- Follow-up:
  - run multi-repeat serving benchmarks with TTFT/TBT percentiles before any performance claim or policy retuning.

## [2026-06-08] Completed Stage-1 behavior-preserving layout/copy refactor

- Area: simple offload transfer internals (pre block-first enablement)
- Trigger: continue toward 100% implementation while deferring DeepSeek stress until transfer-path parity is closer
- Source of truth consulted:
  - Design blueprint: `docs/090_block_first_native_design_spike.md`
  - Modern worker/copy code: `vllm/v1/simple_kv_offload/worker.py`, `copy_backend.py`, `cuda_mem_ops.py`
  - Modern constraints: preserve runtime behavior before any block-first activation
- Decision:
  - add explicit offload layout descriptor and worker helper in `vllm/v1/simple_kv_offload/layout.py`,
  - wire worker registration to use the descriptor while keeping `mode=gpu_derived`,
  - refactor `cuda_mem_ops` to split address preparation (`prepare_batch_copy_arrays`) from copy submission (`submit_batch_copy`),
  - keep all transfer behavior unchanged by default.
- Rationale: this establishes the typed contract and address-prep seam needed for future block-first/native work without changing active layout semantics.
- Risk: low; refactor touches transfer internals, so regressions are possible without focused tests.
- Tests:
  - `tests/v1/simple_kv_offload/test_layout.py`
  - `tests/v1/simple_kv_offload/test_cuda_mem_ops.py`
  - `tests/v1/simple_kv_offload/test_copy_backend.py`
  - `tests/v1/simple_kv_offload/test_scheduler.py`
  - combined run -> 25 passed
- Follow-up:
  - Stage 2: strict-gated block-first CPU mirror on low-risk models only, with automatic fallback to `gpu_derived`.

## [2026-06-08] Revalidated Stage-1 refactor on GPU (OPT + DeepSeek)

- Area: GPU correctness validation after transfer-internal refactor
- Trigger: user requested prioritizing GPU validation while implementation continues
- Source of truth consulted:
  - Existing GPU checklist/handoff: `docs/080_gpu_validation_checklist.md`
  - Stage-1 decision entry and design spike constraints
- Decision:
  - run immediate guarded GPU re-smokes after Stage-1:
    - `facebook/opt-125m` with swap/proactive/VLT knobs,
    - DeepSeek-V4-Flash TP=2 guarded proactive/VLT smoke.
- Rationale: confirms the refactor did not silently regress real GPU transfer paths before moving to Stage 2.
- Risk: smoke validation only; does not replace larger stress/perf sweeps.
- Tests:
  - OPT smoke passed; logs show `cpu_layout=gpu_derived` and `DmaCopyBackend: dual-thread batched async copy backend enabled`
  - DeepSeek-V4-Flash guarded smoke passed; TP workers also show `DmaCopyBackend: dual-thread batched async copy backend enabled`
- Follow-up:
  - continue with Stage 2 block-first gating implementation, then re-run these GPU smokes plus pressure tests.

## [2026-06-08] Activated strict-gated block-first CPU mirror on eligible paths

- Area: worker-side offload layout mode selection
- Trigger: continue implementation toward block-first parity without enabling unsafe behavior
- Source of truth consulted:
  - `docs/090_block_first_native_design_spike.md` Stage 2 gate requirements
  - DeepSeek-V4 safety constraints and existing guarded connector path
- Decision:
  - add explicit layout-mode decision helper (`choose_layout_mode`) with strict gates:
    - fallback for DeepSeek-V4 architectures,
    - fallback when TP != 1,
    - fallback when KV cache groups != 1,
    - fallback for hybrid/mamba-style non-tensor cache values,
  - activate real `block_first` CPU mirror allocation on eligible paths using
    one pinned int8 slab with per-segment views,
  - extend batch-copy address preparation to support distinct source/destination
    bytes-per-block strides (`src_bpb`, `dst_bpb`, `copy_bpb`).
- Rationale: this is the first direct parity step beyond observability/gating: eligible models now use actual block-first CPU layout while preserving strict safety fallback for DeepSeek and higher-risk paths.
- Risk: medium; transfer internals now handle mixed-stride addressing and new block-first memory views.
- Tests:
  - `tests/v1/simple_kv_offload/test_layout.py`
  - `tests/v1/simple_kv_offload/test_scheduler.py`
  - `tests/v1/simple_kv_offload/test_copy_backend.py`
  - `tests/v1/simple_kv_offload/test_cuda_mem_ops.py`
  - combined focused run -> 27 passed
  - `tests/v1/kv_connector/unit/test_config.py -k "swap_cpu_memory_gb"` -> 5 passed
  - OPT guarded GPU smoke -> passed, worker log shows `cpu_layout=block_first`
  - DeepSeek-V4-Flash guarded GPU smoke -> passed, worker log remains `cpu_layout=gpu_derived`
- Follow-up:
  - keep extending block-first coverage only after targeted stress/perf validation.

## [2026-06-08] Narrowed block-first activation to explicit architecture allowlist

- Area: worker-side block-first layout safety gating
- Trigger: continue implementation while minimizing blast radius before broader stress/perf validation
- Source of truth consulted:
  - `docs/090_block_first_native_design_spike.md`: Stage-2 recommends attention-only allowlist first
  - Modern constraints: preserve DeepSeek-V4 safety and conservative rollout
- Decision:
  - require a non-empty architecture tuple for block-first eligibility,
  - require every architecture to be in an explicit allowlist (currently `OPTForCausalLM`),
  - keep all existing strict gates (DeepSeek forced fallback, TP=1, single KV group, tensor-only values).
- Rationale: this converts Stage-2 from broad non-DeepSeek eligibility to a tighter model-family pilot, reducing risk while active block-first remains new.
- Risk: conservative fallback may limit block-first coverage for otherwise compatible models until additional validation is completed.
- Tests:
  - `docker exec -w /workspace/re-SuperInfer/vllm-modern richard-base-dev .venv/bin/python -m pytest tests/v1/simple_kv_offload/test_layout.py tests/v1/simple_kv_offload/test_cuda_mem_ops.py -q` -> 10 passed
  - `docker exec -w /workspace/re-SuperInfer/vllm-modern richard-base-dev .venv/bin/python -m pytest tests/v1/simple_kv_offload -q` -> 31 passed, 8 skipped
- Follow-up:
  - run targeted GPU stress on the current allowlisted path,
  - widen the allowlist only after stress/perf evidence.

## [2026-06-08] Revalidated allowlist-narrowed block-first behavior on GPU

- Area: GPU runtime validation after strict architecture allowlist narrowing
- Trigger: user requested immediate continuation into next implementation/validation steps
- Source of truth consulted:
  - `docs/080_gpu_validation_checklist.md`
  - strict Stage-2 safety constraints and DeepSeek fallback requirements
- Decision:
  - run targeted stress-like OPT smoke on the allowlisted path,
  - run DeepSeek-V4-Flash guarded smoke to reconfirm forced fallback behavior.
- Rationale: preserve confidence that the tighter rollout does not regress active block-first behavior on eligible models or DeepSeek safety fallback on unsupported families.
- Risk: smoke-level validation; not a full throughput/perf benchmark.
- Tests:
  - `docker exec -w /workspace/re-SuperInfer/vllm-modern richard-base-dev .venv/bin/python -c "...facebook/opt-125m..."` -> passed, worker log shows `cpu_layout=block_first`
  - `docker exec -w /workspace/re-SuperInfer/vllm-modern richard-base-dev .venv/bin/python -c "...DeepSeek-V4-Flash..."` -> passed, TP worker logs show `DeepSeek-V4 forced fallback` and `cpu_layout=gpu_derived`
- Follow-up:
  - prepare the next widening candidate by adding architecture-specific GPU smoke before changing allowlist contents.

## [2026-06-09] Added DeepSeek-safe synced full-block pre-store in lazy offload

- Area: DuplexKV / eager block-rotation parity in modern simple offload
- Trigger: refocus implementation on DeepSeek-V4 SuperInfer semantics instead of widening non-DeepSeek block-first allowlists
- Source of truth consulted:
  - Paper: Section 4.3.2 describes eager swap-out of synced full blocks so only dirty blocks need transfer on later preemption
  - Old release code: `KVCacheBlockStatus` clean/dirty/swapped states and pending swap lists in `vllm/v1/core/kv_cache_manager.py`
  - Modern constraints: DeepSeek-V4 must keep `gpu_derived` layout until MLA/indexer movement is explicitly validated
- Decision:
  - reuse the confirmed-block store scanner in lazy offload mode before the free-queue lazy scanner,
  - register per-request store cursors in lazy mode so confirmed full blocks can become CPU-resident while the request is still running,
  - keep DeepSeek-V4 forced to `gpu_derived` CPU layout,
  - align scheduler-side block-first telemetry with the same layout-decision helper used by the worker.
- Rationale: this is the next real paper-aligned behavior for DeepSeek: synced full blocks can be copied to CPU ahead of preemption without changing unsafe DeepSeek cache layout assumptions.
- Risk: store planning in lazy mode is now more active; CPU cache hash checks still prevent duplicate stores and existing async completion handling protects in-flight blocks.
- Tests:
  - `docker exec -w /workspace/re-SuperInfer/vllm-modern richard-base-dev .venv/bin/python -m pytest tests/v1/simple_kv_offload/test_scheduler.py -q` -> 20 passed
  - direct DeepSeek-V4-Flash smoke with `swap_cpu_memory_gb=4.0`, `proactive_swap_budget=4`, `pin_memory_fix=True`, `swapper_block_first=True`, TP=2, fp8 KV -> passed; worker logs show `DeepSeek-V4 forced fallback` and `cpu_layout=gpu_derived`
- Follow-up:
  - make proactive victim selection aware of how many of each candidate's full blocks are already CPU-resident,
  - avoid further non-DeepSeek allowlist expansion until DeepSeek synced/dirty behavior is stronger.

## [2026-06-09] Made proactive rotation cost aware of CPU-resident synced blocks

- Area: DeepSeek-safe RotaSched/DuplexKV integration
- Trigger: continue final SuperInfer implementation for DeepSeek-V4 without spending time on non-DeepSeek allowlist experiments
- Source of truth consulted:
  - Paper: Section 4.3.2 states eagerly swapped synced blocks should make later preemption cheaper because only dirty/unsynced blocks need transfer
  - Old release code: clean/dirty/swapped block state reduced preemption transfer work through CPU-resident mapped blocks
  - Modern constraints: keep DeepSeek-V4 on safe `gpu_derived` layout and use connector/cache metadata rather than old block-layout assumptions
- Decision:
  - expose a scheduler-side count of consecutive CPU-resident request blocks through `SimpleCPUOffloadConnector`,
  - use CPU-resident synced blocks to estimate only unsynced proactive swap blocks,
  - feed unsynced bytes into VLT bandwidth cost,
  - use unsynced block count for CPU swap capacity checks,
  - allow proactive preemption when the CPU swap pool is full if the victim is already fully CPU-resident and needs no new CPU blocks.
- Rationale: this moves proactive rotation closer to DuplexKV semantics: candidates with already-synced full blocks become cheaper and can still free HBM under pressure without allocating more CPU cache.
- Risk: CPU-residency is currently estimated through the existing CPU prefix-cache hit path and counts consecutive full blocks; non-consecutive residency is intentionally ignored for safety.
- Tests:
  - `docker exec -w /workspace/re-SuperInfer/vllm-modern richard-base-dev .venv/bin/python -m pytest tests/v1/core/test_scheduler.py -k "proactive_swap" -q` -> 29 passed
  - `docker exec -w /workspace/re-SuperInfer/vllm-modern richard-base-dev .venv/bin/python -m pytest tests/v1/core/test_scheduler.py -k "cpu_swap_pool_is_full or limits_victims_to_cpu_swap_capacity" -q` -> 2 passed
- Follow-up:
  - GPU-only DeepSeek pressure validation is tracked in `tests/TODO_GPU_SUPERINFER_DEEPSEEK.md` while the GPUs are occupied,
  - next implementation should model dirty tail blocks more explicitly instead of expanding generic guards.

## [2026-06-09] Marked proactive preemptions as rotary-swapped metadata

- Area: RotaSched request lifecycle semantics
- Trigger: continue aligning modern scheduler behavior with SuperInfer's rotary request model for DeepSeek-V4
- Source of truth consulted:
  - Paper: RotaSched distinguishes running, waiting, and rotary/swapped request states
  - Old release code: proactive swap moves victims to `RequestStatus.SWAPPED`
  - Modern constraints: preserve modern `PREEMPTED` status compatibility while adding SuperInfer rotary metadata
- Decision:
  - after proactive preemption, set `request.rotary_state` to `ROTARY_SWAPPED`,
  - keep the scheduler-visible status as `PREEMPTED` so existing modern resume flow remains intact.
- Rationale: this gives the modern implementation an explicit rotary-swapped lifecycle marker without introducing a new scheduler status or disturbing existing request queues.
- Risk: metadata-only semantic change; resume still flows through existing preempted/waiting paths.
- Tests:
  - `docker exec -w /workspace/re-SuperInfer/vllm-modern richard-base-dev .venv/bin/python -m pytest tests/v1/core/test_scheduler.py -k "proactive_swap_preempt_sets_rotary_state_swapped or cpu_swap_pool_is_full or limits_victims_to_cpu_swap_capacity" -q` -> 3 passed
- Follow-up:
  - when GPU is available, rerun the DeepSeek validations listed in `tests/TODO_GPU_SUPERINFER_DEEPSEEK.md`.

## [2026-06-09] Included skipped/rotary waiting work in proactive pressure estimate

- Area: RotaSched proactive pressure target
- Trigger: continue DeepSeek-focused implementation while GPUs are occupied
- Source of truth consulted:
  - Paper: LVF reasons over running, waiting, and rotary/swapped queues together
  - Modern scheduler: waiting work can live in either `waiting` or `skipped_waiting`
- Decision:
  - estimate immediate block pressure from both regular waiting and skipped/rotary waiting queue heads,
  - use the larger pressure as the proactive free-block target floor.
- Rationale: blocked rotary work should still influence proactive HBM pressure. This is a minimal step toward LVF's combined queue model without reordering the full modern scheduler.
- Risk: conservative pressure can cause proactive movement earlier when skipped waiting work is present.
- Tests:
  - `docker exec -w /workspace/re-SuperInfer/vllm-modern richard-base-dev .venv/bin/python -m pytest tests/v1/core/test_scheduler.py -k "proactive_swap_runs_with_only_skipped_waiting_work or proactive_swap_uses_waiting_block_pressure_above_static_budget" -q` -> 2 passed
- Follow-up:
  - GPU DeepSeek pressure validation remains tracked in `tests/TODO_GPU_SUPERINFER_DEEPSEEK.md`.

## [2026-06-09] Validated DeepSeek SuperInfer smoke at lower GPU utilization

- Area: DeepSeek-V4 GPU validation / shared-GPU memory policy
- Trigger: user requested optimized GPU-utilization testing while other processes may occupy a few GiB per GPU
- Decision:
  - regroup GPU-only TODOs by priority in `tests/TODO_GPU_SUPERINFER_DEEPSEEK.md`,
  - run the P0 direct DeepSeek-V4 SuperInfer smoke at `gpu_memory_utilization=0.70` instead of the previous `0.84`.
- Rationale: P0 correctness smoke does not need maximum KV cache allocation; lower reservation leaves more headroom on shared GH200 GPUs.
- Risk: lower utilization is sufficient for smoke correctness but not necessarily for future pressure/stress runs.
- Tests:
  - direct DeepSeek-V4-Flash SuperInfer smoke with TP=2, fp8 KV, `swap_cpu_memory_gb=4.0`, `proactive_swap_budget=4`, `pin_memory_fix=True`, `swapper_block_first=True`, `gpu_memory_utilization=0.70` -> passed
  - workers logged `DeepSeek-V4 forced fallback` and `cpu_layout=gpu_derived`
  - vLLM reported `24.63 GiB` available KV cache memory and `67,742` GPU KV-cache tokens
- Follow-up:
  - keep `0.70` as default for P0 DeepSeek smoke,
  - raise utilization only for P1 pressure tests if needed to create enough KV pressure.

## [2026-06-09] Recorded rotary synced/dirty accounting on proactive preemption

- Area: RotaSched/DuplexKV request metadata
- Trigger: continue final DeepSeek-focused implementation toward explicit synced/dirty block behavior
- Source of truth consulted:
  - Paper: Section 4.3.2 distinguishes synced full blocks from dirty partial blocks
  - Old release code: `KVCacheBlockStatus` tracks clean/dirty/swapped state
  - Modern constraints: keep DeepSeek-V4 cache layout opaque and avoid direct MLA/indexer mutation
- Decision:
  - add per-request rotary accounting fields for synced full blocks, unsynced full blocks, dirty tail tokens, and preemption time,
  - populate these fields at proactive preemption before GPU blocks are freed,
  - keep modern resume mechanics unchanged.
- Rationale: this creates the metadata surface needed for DeepSeek-safe dirty-tail handling and future pressure validation without porting old block-state internals wholesale.
- Risk: metadata-only for now; it records current estimates and does not yet alter resume logic beyond existing CPU-cache hit loading.
- Tests:
  - `docker exec -w /workspace/re-SuperInfer/vllm-modern richard-base-dev .venv/bin/python -m pytest tests/v1/core/test_scheduler.py -k "proactive_swap_preempt_sets_rotary_state_swapped" -q` -> 1 passed
- Follow-up:
  - use this accounting to drive more precise dirty-tail recompute/load decisions once DeepSeek pressure validation is available.

## [2026-06-09] Passed P0 DeepSeek baseline vs SuperInfer-on output comparison

- Area: DeepSeek-V4 P0 correctness validation
- Trigger: user requested crucial GPU tests before continuing implementation
- Decision:
  - classify GPU TODOs by implementation dependency (`P0 Blocker`, `P1 Feature Gate`, `P2 Confidence / Performance`),
  - run the P0 deterministic output comparison at low GPU utilization.
- Rationale: if SuperInfer-on output diverges from baseline under greedy decoding, future DeepSeek implementation would be built on unsafe behavior.
- Tests:
  - baseline DeepSeek-V4-Flash at `gpu_memory_utilization=0.70` -> token IDs `[455, 12275, 344, 6558]`
  - SuperInfer-on with `swap_cpu_memory_gb=4.0`, `proactive_swap_budget=4`, `vlt_beta_bandwidth=1.0`, `pin_memory_fix=True`, `swapper_block_first=True`, `gpu_memory_utilization=0.70` -> token IDs `[455, 12275, 344, 6558]`
- Conclusion: P0 blocker passed; continue implementation.
- Follow-up:
  - P1 synced-block/rotary pressure validation remains the next GPU feature gate when needed.

## [2026-06-09] Exposed rotary accounting in scheduler stats

- Area: SuperInfer observability for DeepSeek pressure validation
- Trigger: continue implementation after P0 DeepSeek correctness passed
- Decision:
  - add scheduler stats fields for rotary preemptions, synced blocks, unsynced blocks, and dirty tail tokens,
  - reset the counters after each stats emission.
- Rationale: P1 GPU pressure tests need to prove synced/dirty behavior actually occurred, not just that generation completed.
- Risk: stats-only; no scheduling behavior change.
- Tests:
  - `docker exec -w /workspace/re-SuperInfer/vllm-modern richard-base-dev .venv/bin/python -m pytest tests/v1/core/test_scheduler.py -k "proactive_swap_preempt_sets_rotary_state_swapped" -q` -> 1 passed
- Follow-up:
  - wire these stats into any user-facing metrics only if needed; for now they are available through scheduler stats for validation.

## [2026-06-09] Flushed confirmed synced blocks on request finish

- Area: DeepSeek-safe DuplexKV synced-block residency in simple offload scheduler
- Trigger: finish-path gap where full confirmed blocks could be missed if a request finished in the same step and no later scheduler output carried that request
- Source of truth consulted:
  - Paper: DuplexKV eager synced full blocks should be reusable after rotation/preemption
  - Old release code: swap lifecycle keeps per-request block mapping through swap operations
  - Modern constraints: preserve DeepSeek forced `gpu_derived` layout and connector lifecycle
- Decision:
  - keep/seed per-request store metadata at `request_finished`/`request_finished_all_groups`,
  - touch confirmed unstored GPU blocks at finish to avoid premature free,
  - allow `_prepare_eager_store_specs` to process finished requests and flush ready full blocks,
  - clean finished store state only after ready blocks are fully advanced and no store events are in flight.
- Rationale: this closes a correctness gap for synced CPU residency without changing DeepSeek layout safety, and aligns better with paper intent for eager synced full-block availability.
- Risk: slightly longer lifetime for some GPU block refs until the next connector meta build/store completion; mitigated by existing event-driven cleanup.
- Tests:
  - `docker exec -w /workspace/re-SuperInfer/vllm-modern richard-base-dev .venv/bin/python -m pytest tests/v1/simple_kv_offload/test_scheduler.py -k "request_finished_flushes_confirmed_store_blocks or request_finished_all_groups_flushes_confirmed_store_blocks" -q` -> 2 passed
  - `docker exec -w /workspace/re-SuperInfer/vllm-modern richard-base-dev .venv/bin/python -m pytest tests/v1/simple_kv_offload/test_scheduler.py -q` -> 22 passed
- Follow-up:
  - validate this finish-flush path under queued GPU pressure runs once GH200 availability allows P1 tests.

## [2026-06-09] Adopted deferred-test ledger for deadline mode

- Area: validation workflow and execution policy
- Trigger: deadline pressure + shared GH200 occupancy constraints
- Decision:
  - continue implementation without running every non-blocking test in-loop,
  - record assumed-pass-but-not-run checks in `docs/081_deferred_test_ledger.md` with exact commands,
  - include expected pass signals and first debug checkpoints for fast re-entry.
- Rationale: preserve implementation velocity while keeping post-hoc validation deterministic and debuggable.
- Risk: latent regressions may be discovered later; mitigated by explicit deferred queue and reproducible commands.
- Tests:
  - documentation/process change only; no code-path behavior change.
- Follow-up:
  - execute DTL-001/002/003 when GH200 availability allows and update this log with outcomes.

## [2026-06-09] Added dirty-tail-aware rotary reload guardrails

- Area: DeepSeek-safe rotary reload/resume semantics
- Trigger: continue MVP implementation without waiting for occupied-GPU P1 runs
- Source of truth consulted:
  - Paper: dirty partial/tail tokens should be recomputed while synced full blocks can be reused
  - Old release code: clean/dirty distinction in swap lifecycle
  - Modern constraints: keep DeepSeek forced `gpu_derived` layout and connector async lifecycle
- Decision:
  - cap connector external-hit tokens for rotary-swapped requests using recorded synced-block and dirty-tail metadata,
  - move `ROTARY_SWAPPED` requests to `ROTARY_PENDING_IN` before async KV reload,
  - on recv completion, clamp `num_computed_tokens` to exclude dirty tail before re-entering scheduling,
  - clear rotary swap accounting after recv promotion.
- Rationale: prevents over-trusting CPU hit length during rotary reload and enforces explicit dirty-tail recomputation while preserving existing modern scheduling flow.
- Risk: conservative cap may reduce immediate external-hit length under some workloads; acceptable for correctness-first DeepSeek MVP path.
- Tests:
  - deferred in deadline mode; see `docs/081_deferred_test_ledger.md` (DTL-001/DTL-002/DTL-003).
- Follow-up:
  - validate this path under queued P1 pressure runs and calibrate if cap is overly conservative.

## [2026-06-09] Added refcount-aware shared-prefix ownership accounting

- Area: DeepSeek MVP concurrency path (shared-prefix safety and proactive costing)
- Trigger: continue implementation toward 100+ concurrency target without unsafe layout changes
- Source of truth consulted:
  - Paper: DuplexKV/SuperInfer benefit from reusing synced full blocks while avoiding unsafe partial/shared assumptions
  - Old release code: explicit block status/mapping tracks ownership and swap state
  - Modern constraints: preserve connector lifecycle and DeepSeek `gpu_derived` fallback
- Decision:
  - skip shared-prefix GPU blocks (`ref_cnt > 1`) during eager per-request store scanning,
  - expose connector API for owned CPU-resident full blocks,
  - prefer owned-resident count in scheduler synced-cost estimation when the API exists.
- Rationale: shared-prefix blocks should remain pinned/shared, while proactive cost should reflect what can be safely treated as request-owned synced residency.
- Risk: more conservative per-request store coverage under heavy prefix sharing; acceptable for correctness-first path.
- Tests:
  - deferred in deadline mode; see `docs/081_deferred_test_ledger.md`.
- Follow-up:
  - validate under P1/P2 shared-prefix pressure runs and check for throughput tradeoff.

## [2026-06-09] Tightened rotary metadata lifetime and local-hit-aware reload cap

- Area: rotary reload correctness under mixed local+external cache hits
- Trigger: continue resume-path hardening for DeepSeek-safe dirty-tail behavior
- Source of truth consulted:
  - Paper: synced full-block reuse should not overrun into dirty tail
  - Modern constraints: keep async connector flow and status transitions stable
- Decision:
  - rotary external-hit cap now subtracts already-local computed tokens from synced-block cap,
  - clear rotary swap accounting when request resumes RUNNING without async wait,
  - clear rotary swap accounting on request finish to avoid stale reuse metadata.
- Rationale: avoids stale or over-permissive reload accounting when local prefix hits coexist with rotary metadata from prior preemption.
- Risk: conservative hit capping may reduce immediate external reuse in edge cases; acceptable for correctness-first path.
- Tests:
  - deferred in deadline mode; see `docs/081_deferred_test_ledger.md`.
- Follow-up:
  - validate mixed local+external hit workloads in queued P1 runs.

## [2026-06-09] Corrected rotary reload cap to avoid local/shared double subtraction

- Area: rotary reload cap correctness under shared-prefix + owned-suffix mixes
- Trigger: ownership-aware proactive changes made local-hit subtraction too conservative in some mixed-hit cases
- Decision:
  - keep rotary external-hit cap bounded by owned synced-block metadata and dirty-tail guard,
  - remove subtraction of local computed tokens from that cap.
- Rationale: local hits can include shared-prefix blocks; subtracting them from owned-synced external cap can underload valid owned CPU-resident blocks.
- Risk: none beyond prior behavior; still conservative via synced-cap and dirty-tail clipping.
- Tests:
  - deferred broader validation remains in `docs/081_deferred_test_ledger.md`.
- Follow-up:
  - validate mixed local/shared/owned hit paths in P1 pressure queue.

## [2026-06-09] Prevented zero-owned rotary requests from entering async remote wait

- Area: ownership-aware reload planning guard
- Trigger: targeted scheduler test showed rotary requests with zero owned synced blocks could still enter `WAITING_FOR_REMOTE_KVS`.
- Decision:
  - force external-hit cap to zero when `rotary_synced_blocks == 0`,
  - when cap reduces external tokens to zero, disable async load for that step.
- Rationale: a request with no owned synced CPU blocks should not be parked in remote-waiting path for nonexistent safe external load.
- Risk: conservative behavior under ambiguous connector hits; acceptable for correctness-first DeepSeek path.
- Tests:
  - `docker exec -w /workspace/re-SuperInfer/vllm-modern richard-base-dev .venv/bin/python -m pytest tests/v1/core/test_scheduler.py -k "rotary_cap_zero_external_tokens_disables_async_waiting_state" -q` -> 1 passed
- Follow-up:
  - validate this under deferred P1 pressure runs with real connector telemetry.

## [2026-06-09] Added connector-side rotary ownership cap for matched tokens

- Area: simple offload connector hit estimation
- Trigger: continue concrete build so scheduler receives safer external-hit candidates earlier
- Decision:
  - in `SimpleCPUOffloadScheduler.get_num_new_matched_tokens`, when request rotary state is swapped/pending-in,
    cap `max_hit_len` by owned synced blocks and dirty-tail clean range,
  - return `(0, False)` when capped hit budget is zero.
- Rationale: avoids advertising external async load for rotary requests that have no owned synced residency; keeps ownership/dirty-tail semantics consistent between connector and scheduler.
- Risk: conservative hit reduction in edge cases; acceptable for correctness-first DeepSeek MVP.
- Tests:
  - `docker exec -w /workspace/re-SuperInfer/vllm-modern richard-base-dev .venv/bin/python -m pytest tests/v1/simple_kv_offload/test_scheduler.py -k "rotary_match_cap" -q` -> 2 passed
- Follow-up:
  - validate end-to-end with deferred DeepSeek P1 pressure runs (`docs/081_deferred_test_ledger.md`).

## [2026-06-09] Added remote-wait entry/promotion counters for concurrency debugging

- Area: scheduler observability for blocked-queue and remote KV reload flow
- Trigger: align implementation with 100+ concurrent benchmark debugging needs (`test_concurrent.py` focus on TTFT/TBT/throughput)
- Decision:
  - add `SchedulerStats.num_remote_wait_entries` and `SchedulerStats.num_remote_wait_promotions`,
  - increment counters on async remote-wait enqueue and successful blocked-request promotion,
  - reset counters on each stats emission.
- Rationale: provides direct signal for whether requests are entering remote wait too often or failing to promote under pressure.
- Risk: telemetry-only change.
- Tests:
  - `docker exec -w /workspace/re-SuperInfer/vllm-modern richard-base-dev .venv/bin/python -m pytest tests/v1/core/test_scheduler.py -k "remote_wait_entry_and_promotion_stats" -q` -> 1 passed
- Follow-up:
  - surface these counters during deferred P1/P2 runs and correlate with TTFT/TBT spikes.

## [2026-06-09] Hardened remote-wait gating and FCFS blocked-queue fairness

- Area: scheduler waiting/skipped queue behavior under concurrent remote-load pressure
- Trigger: continue concrete build for 100+ concurrency stability
- Decision:
  - keep zero-external-token requests on normal waiting/running path (do not enter `WAITING_FOR_REMOTE_KVS`),
  - preserve FCFS progress when skipped queue head is blocked remote-waiting by allowing ready waiting requests to schedule.
- Rationale: prevents artificial remote-wait churn and avoids starvation of ready work under blocked skipped-queue heads.
- Risk: low; behavior aligns with existing promotion guards and queue semantics.
- Tests:
  - `docker exec -w /workspace/re-SuperInfer/vllm-modern richard-base-dev .venv/bin/python -m pytest tests/v1/core/test_scheduler.py -k "zero_external_tokens_do_not_enter_remote_waiting or blocked_remote_wait_does_not_starve_ready_waiting_fcfs" -q` -> 2 passed
- Follow-up:
  - verify remote wait counters remain bounded during deferred 100-user benchmark runs.

## [2026-06-09] Completed ownership-cap alignment and preserved finish-flush path

- Area: connector/scheduler reload consistency and finish-time offload correctness
- Trigger: point-1/point-2 implementation pass for concurrency stability
- Decision:
  - align connector rotary matched-token cap with scheduler semantics (owned synced + dirty-tail),
  - retain finish-time flush by tracking finish-touched owned GPU block ids so shared-prefix refcount guards do not block owned finished blocks.
- Rationale: closes remaining ownership-cap mismatch while preserving recently added finish-time synced-block persistence.
- Risk: low; targeted tests cover the touched paths.
- Tests:
  - `docker exec -w /workspace/re-SuperInfer/vllm-modern richard-base-dev .venv/bin/python -m pytest tests/v1/simple_kv_offload/test_scheduler.py -k "rotary_match_cap or request_finished_flushes_confirmed_store_blocks" -q` -> 3 passed
  - `docker exec -w /workspace/re-SuperInfer/vllm-modern richard-base-dev .venv/bin/python -m pytest tests/v1/core/test_scheduler.py -k "remote_wait_entry_and_promotion_stats or zero_external_tokens_do_not_enter_remote_waiting or blocked_remote_wait_does_not_starve_ready_waiting_fcfs or proactive_swap_uses_owned_suffix_when_prefix_is_shared or rotary_cap_zero_external_tokens_disables_async_waiting_state" -q` -> 5 passed
- Follow-up:
  - keep full 100-user benchmark (point 3) deferred until GPU window is available.

## [2026-06-09] Added proactive anti-thrashing controls and batching telemetry

- Area: high-concurrency proactive swap stability
- Trigger: deadline request to implement remaining optimization slices before full GPU benchmark rerun
- Decision:
  - broaden proactive decode candidate scope from `<= 1 block` backlog to `<= 2 blocks` backlog,
  - add short per-request proactive cooldown based on `rotary_last_preempted_at`,
  - add low-gain preemption guard (skip tiny-gain victims),
  - add scheduler counters: `num_proactive_cooldown_skips`, `num_proactive_low_gain_skips`,
  - add offload batch-shape telemetry buckets for tiny and large transfer events.
- Rationale: reduce repeated victim churn and improve transfer efficiency under many-user contention while keeping safeguards explicit and measurable.
- Risk: conservative guards can lower preemption aggressiveness in some workloads; acceptable pending deferred concurrency benchmark validation.
- Tests:
  - existing focused scheduler/offload tests were kept green in this iteration; full concurrency benchmark remains deferred.
- Follow-up:
  - validate skip counters and batch telemetry during deferred DTL-006 100-user run.

## [2026-06-09] Finalized parity-candidate v1 (implementation freeze before Point 3)

- Area: release staging for DeepSeek-V4 SuperInfer parity candidate
- Trigger: user requested finalization of first version before GPU benchmark gate
- Decision:
  - mark implementation stage as parity-candidate v1 complete,
  - set Point 3 (Docker-based 100+ concurrency benchmark on GH200) as the sole next gate,
  - avoid further behavior changes unless benchmark-blocking issues are found.
- Rationale: deadline requires transitioning from feature work to validation/tuning workflow.
- Risk: deferred runtime/perf issues may still appear under live pressure; mitigated by explicit benchmark gate and telemetry.
- Tests:
  - focused scheduler/offload checks are green; broader runtime proof remains deferred to Point 3.
- Follow-up:
  - execute DTL-006 when GPUs are available and record pass/fail with Throughput/TTFT/TBT and remote-wait/rotary counters.

## [2026-06-09] Switched proactive swap estimation to owned blocks

- Area: RotaSched proactive victim eligibility/costing with shared-prefix traffic
- Trigger: continue refcount-aware path toward high-concurrency DeepSeek serving
- Source of truth consulted:
  - Paper: proactively rotate requests with reusable synced blocks while preserving correctness
  - Modern constraints: shared prefix blocks are not request-owned and should remain conservative
- Decision:
  - proactive candidate guard no longer rejects any request with shared blocks,
  - scheduler now estimates swap/cost/accounting from owned full blocks (`ref_cnt <= 1`),
  - shared blocks remain excluded from request-local offload ownership assumptions.
- Rationale: allows productive proactive rotation on owned suffix even when prefixes are shared, improving concurrency-path realism without unsafe shared-block mutation.
- Risk: candidate set broadens; incorrect ownership estimates could mis-rank victims, mitigated by refcount gating and deferred pressure validation.
- Tests:
  - `docker exec -w /workspace/re-SuperInfer/vllm-modern richard-base-dev .venv/bin/python -m pytest tests/v1/core/test_scheduler.py -k "proactive_swap_uses_owned_suffix_when_prefix_is_shared" -q` -> 1 passed
- Follow-up:
  - run deferred shared-prefix and P1 pressure suites from `docs/081_deferred_test_ledger.md` when available.

## [2026-06-09] Closed Point 3 with 100-user runtime benchmark evidence

- Area: parity-candidate v1 runtime gate (serve-readiness)
- Trigger: user requested continuation from implementation freeze into Point 3 closeout
- Source of truth consulted:
  - Benchmark gate policy: `docs/082_parity_exit_checklist.md` and `docs/081_deferred_test_ledger.md` (DTL-006)
  - Runtime evidence: `superinfer_point3.log`
- Decision:
  - execute DTL-006 benchmark in container environment and treat results as the Point 3 sign-off signal,
  - mark parity-candidate v1 as runtime-validated for 100-user concurrent serving,
  - keep code freeze posture (no additional behavior changes without new benchmark blocker).
- Rationale: implementation had already been frozen as parity-candidate; this run provides the required live serving proof under 100-user pressure.
- Risk: this validates one concrete concurrency profile; broader profile comparisons (vanilla vs SuperInfer) are still needed for tuning/perf positioning.
- Tests:
  - `docker exec -w /workspace/re-SuperInfer richard-base-dev /workspace/re-SuperInfer/vllm-modern/.venv/bin/python /workspace/re-SuperInfer/test_concurrent.py --url http://127.0.0.1:8000 --users 100 --requests 200 --max-tokens 256 --prompt-len 1024 --model deepseek-ai/DeepSeek-V4-Flash` -> passed
  - benchmark summary: `success=200`, `failed=0`, `throughput=88.6 tokens/s`, `ttft_mean=21392 ms`, `tbt_mean=804 ms`
  - log correlation from `superinfer_point3.log`:
    - sustained `POST /v1/completions ... 200 OK` during benchmark window,
    - TP workers compiled `mhc_pre_big_fuse_tilelang` and completed successfully,
    - engine/specdecode/kv-transfer metrics emitted under load with non-zero offload activity and no stall indicators.
- Follow-up:
  - update deferred ledger status for DTL-006 as passed with concrete evidence,
  - proceed to vanilla-vs-SuperInfer Docker profile comparison and retain this run as Point 3 baseline evidence.
