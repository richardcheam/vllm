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
