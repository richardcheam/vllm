# Feature Checklist

Status legend:

- `[x]` verified implemented and covered by source/test evidence.
- `[~]` partially implemented or runtime validation incomplete.
- `[ ]` not implemented or not yet validated.

This checklist is intentionally separate from the paper-parity claim. The
current target is a vLLM 0.26-compatible adaptation, not a complete clone of
the official v0.6.6-era native SuperInfer runtime.

## Environment and baseline

- [x] Run in `richard-base-dev-sysnice`.
- [x] Use the target `.venv`.
- [x] Verify target import path and vLLM version.
- [x] Verify Torch/CUDA/Triton versions.
- [x] Verify both GH200s and NVLink topology.
- [x] Verify `DeepSeek-V4-Flash-0731` and DSpark assets.
- [x] Run vanilla DSpark before runtime changes.
- [x] Preserve baseline logs and benchmark artifacts.
- [~] Repeat the full matrix after the latest scheduler watermark fix.
- [ ] Reproduce vanilla DSpark on a clean v0.27.x checkout before any baseline
  upgrade.

## Configuration and CLI

- [x] Add topology-aware enable/disable switch.
- [x] Add CPU KV capacity configuration.
- [x] Add local-pool fraction and bandwidth parameters.
- [x] Add proactive rotation and VLT/SLO parameters.
- [x] Add block-first and pinning controls.
- [x] Validate TP=2, PP=1, DSpark constraints at configuration/source level.
- [~] Prove disabled mode equivalence with a matched runtime comparison.
- [x] Document flags, profiles, and defaults.

## GH200 topology

- [x] Map logical to physical GPUs.
- [x] Discover GPU NUMA affinity.
- [x] Discover CPU NUMA capacity.
- [x] Backport per-worker and EngineCore NUMA binding controls.
- [x] Discover NVLink/P2P peers.
- [x] Respect `CUDA_VISIBLE_DEVICES` in the topology helper.
- [x] Allocate local pools first.
- [x] Track remote fallback in manager telemetry.
- [x] Provide discovery fallback and diagnostics.
- [ ] Verify actual page placement with NUMA counters during a live transfer.

## CPU KV pools

- [x] Derive capacity from v0.26 KV metadata.
- [x] Preserve packed layouts and block strides.
- [x] Separate CPU/GPU block namespaces.
- [x] Track ownership and references for normal offload paths.
- [x] Protect shared prefix blocks in normal proactive selection.
- [x] Handle CPU exhaustion through connector fallback/skip behavior.
- [x] Test repeated allocation/free cycles.
- [ ] Prove multi-group DeepSeek residency ownership under pressure.

## Block-first and transfer layout

- [x] Inspect DeepSeek-V4 KV groups.
- [~] Block-first addressing exists, DeepSeek-V4/TP=2 remains GPU-derived,
  contiguous fallback runs are coalesced, and descriptor-compatible DMA now
  supports separate source/destination strides; GPU numerical validation is
  still pending.
- [x] Preserve GPU-derived fallback.
- [ ] Validate K/V numerical equivalence for block-first mode.
- [x] Batch transfers by direction with bounded independent queues.
- [x] Use separate H2D/D2H streams.
- [x] Preserve compute-done event ordering.
- [~] Capture transfer metrics; effective bandwidth still needs a clean report.

## DuplexKV residency

- [~] Track residency through connector/prefix state with explicit per-block
  states, authoritative eager re-store reconciliation, confirmed final block
  handling, and request-record compaction; dirty-tail GPU validation remains
  pending.
- [x] Move stable blocks through lazy/eager store paths, including completed
  requests before the ordinary lazy watermark.
- [~] Protect confirmed data and classify dirty tails; DSpark dirty-tail and
  rollback semantics remain GPU-unvalidated.
- [x] Prevent premature block reuse with refs/events/flush ordering.
- [x] Wait for restore completion before resumed scheduling.
- [~] Lifecycle cleanup exists; cancellation during active transfer needs stress proof.
- [x] Drain transfer workers during shutdown path.
- [~] Propagate partial transfer failures to the worker; recovery semantics
  still need end-to-end validation.

## LVF/VLT rotation

- [~] Running/preempted metadata exists; explicit paper rotary state is absent.
- [x] Compute SLO-aware VLT.
- [~] Rank guarded candidates; not full waiting/running/rotary LVF.
- [x] Select guarded proactive victims when eligible.
- [x] Include free-HBM pressure and rotation budget without conflating them.
- [x] Include local/remote transfer cost estimates.
- [x] Protect prefill and speculative requests in normal mode.
- [x] Protect shared prefixes in normal mode.
- [x] Add cooldown behavior; starvation bounds need stress validation.
- [~] Preserve v0.26 admission/watermark behavior; full interaction needs matrix proof.

## DSpark compatibility

- [x] Preserve five-token minimum for this checkpoint.
- [x] Account for v0.26 lookahead configuration.
- [x] Protect draft execution from proactive rotation in normal mode.
- [ ] Validate rejection and KV rollback with SuperInfer transfers.
- [ ] Test restore before resumed decode.
- [x] Test/configure TP=2.
- [x] Keep PP=1 as required by DSpark.
- [ ] Measure DSpark acceptance rate.
- [~] High-risk override exists, but is experimental and unvalidated.

## Worker and executor overlap

- [x] Identify v0.26 runner hooks.
- [x] Keep copy submission off the compute critical path and submit H2D loads
  before forward.
- [x] Preserve CUDA graph/worker event boundaries at source level.
- [x] Keep buffers alive through completion.
- [x] Drain worker threads.
- [x] Provide synchronous fallback backend.
- [~] Measure exposed transfer stalls from live queue-depth telemetry.

## Validation and performance

- [x] Unit-test topology and address calculations.
- [x] Unit-test metadata and lifecycle transitions.
- [x] Unit-test scheduler scoring and offload victim helpers.
- [~] Test mixed prefill/decode pressure through matrix; needs stress classification.
- [ ] Test repeated preemption under DSpark pressure.
- [x] Test clean shutdown; active-transfer cancellation needs stress proof.
- [~] Test CPU exhaustion and remote fallback; source coverage exists, live
  pressure run is clean but has not exceeded GPU KV capacity yet.
- [~] Run saturation ladder; exhaustive matrix is in progress/recently completed.
- [x] Compare vanilla, native offload, and SuperInfer profiles.
- [~] High-risk serving candidate is launched and smoke-validated; long-context
  proactive-rotation correctness and rollback stress remain pending.
- [~] Maximum-throughput serving candidate selected as `.env.superinfer-serving`
  and basic smoke-validated; final long-context saturation selection remains
  pending.

## Missing cross-cutting features

- [ ] Persist DSpark acceptance, rejection, rollback, and restore counters.
- [ ] Persist scheduler proactive-skip/victim/rotation counters in benchmark artifacts.
- [~] Export SuperInfer connector telemetry through `/metrics`; live export is
  validated, but nonzero transfer attribution remains pending.
- [ ] Add output-equivalence harness for vanilla versus each offload profile.
- [~] Add repeated-prefix CPU store/load output-equivalence harness; exact output
  equality and store evidence pass, while hybrid CPU lookup/load proof remains
  pending. The harness now has a fail-closed `validation.json` metric gate.
- [ ] Add fault-injection tests for transfer failure and worker failure.
- [ ] Add a separate 1M-token/CPU-capacity exhaustion profile.
- [ ] Add a profile-level correctness gate to the matrix orchestrator.
- [ ] Evaluate v0.27.x partial-tail, MLA layout, and DeepSeek kernel backports
  individually before changing the project baseline.

## Parity Boundary

- [ ] Native C++/CUDA/ZMQ DuplexKV swap path.
- [ ] Full paper RotaSched/LVF waiting/running/rotary queue semantics.
- [ ] Paper-level residency, retry, and failure-recovery semantics.
- [ ] Full cross-iteration executor overlap and native control-plane behavior.

Upstream v0.27.x features may improve performance without completing these
parity items. Track accelerator validation and parity validation independently.
