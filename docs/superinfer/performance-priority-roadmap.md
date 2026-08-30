# Performance Priority Roadmap

This roadmap orders work by expected performance impact on the fixed target:
two GH200s, TP=2, PP=1, DeepSeek-V4-Flash-0731, and DSpark. It deliberately
puts measurement and correctness gates before broad paper-parity work.

## Priority 0: Close The Current Measurement Loop

**Status:** completed for the four-profile run; native offload is invalid.

- Finish the four-profile matrix: vanilla, native offload, normal SuperInfer,
  and high-risk SuperInfer.
- Require identical robust unique-prefix warmup and measurement workloads.
- Persist TTFT/TBT summaries, GPU snapshots, Prometheus metrics, and log
  diagnostics.
- Reject a profile report when requests fail, model names differ, or the server
  reports an engine-dead condition.
- Compare the median and tail behavior, not only the best repeat.
- Keep the native-offload control marked invalid until its upstream store-job
  assertion is isolated or fixed independently of SuperInfer.

**Why first:** The previous matrix showed only a small SuperInfer advantage in
one run and a regression in another. Without stable evidence, parameter tuning
can optimize noise.

**Exit gate:** three successful repeats for each valid profile, valid model
identity, latency summaries, no unclassified engine failures, and a generated
matrix report. A failed control must be classified and excluded rather than
converted to zero throughput.

## Priority 1: Make SuperInfer Telemetry Complete

**Status:** exporter and live `/metrics` capture validated; nonzero transfer
attribution remains pending.

- Expose SuperInfer store/load bytes, event counts, bandwidth, locality, pending
  queue depth, CPU pool usage, and proactive decisions through benchmark artifacts.
- Add counters for proactive victims, skipped candidates, DSpark skips, dirty or
  unsynced blocks, and CPU allocation failures.
- Verify in a pressure run that the metrics are emitted by the scheduler-side
  connector and not lost during worker aggregation.

**Why second:** The current logs prove topology initialization, but the
  Prometheus snapshots do not expose the custom SuperInfer transfer counters.
  This prevents attributing a performance delta to actual KV movement.

**Exit gate:** every SuperInfer pressure run reports nonzero and internally
consistent transfer/residency metrics when pressure requires movement.

## Priority 2: Tune The Active Safe Path

**Status:** deferred until CPU reload and transfer attribution are proven.

Sweep one variable at a time, then a small joint matrix:

- `SWAP_CPU_MEMORY_GB`: 8, 16, 32, 64.
- `PROACTIVE_SWAP_BUDGET`: 0, 256, 600, 1200, 2400.
- `MAX_NUM_SEQS`: 16, 24, 32, 48.
- `MAX_NUM_BATCHED_TOKENS`: 8192, 16384, 32768.
- `GPU_MEMORY_UTILIZATION`: 0.90, 0.92, 0.94.
- DSpark tokens: 5, then larger checkpoint-valid values.

CPU capacity is a separate opt-in axis. `.env.superinfer-service128` provides
the unified one-server validation/service profile with 128 GiB total (64 GiB per
TP rank), while `.env.superinfer-cpu128` remains the isolated pressure profile.
The observed host has about 1.2 TiB RAM, so 128 GiB is a reasonable starting
allocation, but leave headroom for the OS, model processes, other services, and
NUMA locality. `CPU_KV_ALLOCATION_MODE=empty` avoids initialization zero-fill
but does not change transfer semantics. Capacity extension can improve
working-set retention; it must not be interpreted as a throughput gain without
pressure measurements.

Keep `SUPERINFER_HIGH_RISK_MODE=0` and `SWAPPER_BLOCK_FIRST=0` for this phase.

**Why third:** These settings can improve performance without changing the
  correctness boundary or enabling unvalidated DSpark rotation.

**Exit gate:** normal SuperInfer is stable and beats native offload or has a
  clear measured tradeoff under the intended pressure regime.

Do not begin this sweep before the reload gate reports positive H2D deltas. A
policy sweep cannot create a breakthrough while the connector still stores KV
but recomputes the corresponding prefix.

## Priority 3: Validate And Tune High-Risk DSpark Rotation

**Status:** selected for the maximum-throughput serving candidate; stress
validation remains incomplete.

- Compare `.env.superinfer-high-risk` against normal SuperInfer.
- Use `.env.superinfer-serving` for the 128 GiB/NUMA/topology service candidate.
- Confirm DSpark rejection and rollback output equivalence.
- Confirm restored requests resume with identical output.
- Stress repeated preemption, cancellation during transfer, and shutdown.
- Measure whether high-risk mode actually increases CPU residency/rotation or
  only adds overhead.

**Why fourth:** This has the largest possible upside for combining DSpark with
  SuperInfer, but it can corrupt KV state or invalidate speculative bookkeeping.

**Exit gate:** no output divergence, no deadlocks, no leaked blocks, and a
  repeatable throughput/latency improvement.

## Priority 4: Improve Transfer Throughput And Overlap

**Status:** first duplex-safety slice implemented; overlap remains incomplete.

- Measure effective local H2D/D2H bandwidth.
- Verify CPU page placement on NUMA-local nodes.
- Tune transfer batch size and queue behavior.
- Check whether the current dual-thread backend is actually overlapping loads
  and stores under pressure.
- Reduce small transfer events and metadata churn.

Implementation sequence:

1. Independent load/store admission and bounded queues.
2. Per-transfer compute events with safe lifetime.
3. Pre-forward load submission after ordering validation.
4. Contiguous fallback-copy coalescing.
5. Descriptor-driven heterogeneous DMA for DeepSeek layouts; source stride-aware
   batch parameters are now implemented, but real DeepSeek validation is pending.

The bounded queue/error-propagation, pre-forward H2D submission, contiguous
fallback-copy, authoritative eager-residency reconciliation, and stride-aware
heterogeneous DMA slices are source-implemented with CPU/GPU-mock unit
coverage; GPU overlap and real worker-to-scheduler queue pressure remain
runtime validation work.

**Why fifth:** The paper's largest performance gain comes from efficient large
  DuplexKV transfers. This work matters after telemetry proves transfers are the
  bottleneck.

**Exit gate:** measured transfer bandwidth and reduced exposed transfer stalls.

## Priority 5: Explicit Residency State Machine

**Status:** partial.

- Add explicit GPU-only, CPU-only, synced, dirty, load-in-flight, and
  store-in-flight state tracking.
- Track dirty tails and confirmed full blocks per request/group.
- Make cancellation and failure transitions explicit.

**Why fifth/sixth:** This enables deeper parity and safer high-risk rotation,
  but it should follow evidence that state ambiguity is limiting performance or
  correctness.

## Priority 6: DeepSeek Block-First Layout

**Status:** guarded fallback; not active for DeepSeek-V4 TP=2.

- Validate actual DeepSeek KV group/tensor layouts.
- Add numerical block-address tests.
- Benchmark block-first versus GPU-derived copies.
- Enable only after K/V equivalence and DSpark compatibility pass.

**Why later:** It may provide the largest transfer improvement, but DeepSeek-V4
  hybrid/compressed MLA layouts make this the highest-risk invasive change.

## Priority 7: Full Official/Paper Parity

**Status:** long-term.

- Paper-level waiting/running/rotary queues.
- Full RotaSched/LVF behavior.
- Native C++/ZMQ DuplexKV path.
- Full cross-iteration executor overlap.
- Complete failure/retry/admission semantics.

**Why last:** These mechanisms are valuable for parity, but they are invasive
  and not necessarily the fastest route to better measured performance on the
  current vLLM 0.26/DSpark/GH200 target.

## Decision Rule

Do not start a lower-priority item until the higher-priority exit gate is met.
In particular:

- Do not claim high-risk correctness from reload correctness alone; keep the
  normal-mode rollback profile available during service.
- Do not port native DuplexKV before measuring current transfer bottlenecks.
- Do not enable DeepSeek block-first before output and K/V equivalence tests.
- Do not claim a SuperInfer gain from a run with no confirmed CPU KV movement.
