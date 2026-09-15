# 30-Minute Presentation Storyboard

Suggested titles:

1. **Topology-Aware LLM Inference on Dual GH200: Modernizing SuperInfer for vLLM**
2. **Extending GPU KV Capacity with Grace Memory on DeepSeek-V4**
3. **From GPU Pressure to CPU-KV Offload: A Modern vLLM Forward-Port**
4. **Engineering Reliable Long-Context Serving on Dual GH200**

## Slide 1 — Title

- **Message:** This is a systems engineering story, not just a benchmark.
- **Show:** title, model, dual GH200, vLLM/SuperInfer subtitle.
- **Say:** GPU HBM is the hot tier; the project investigates a warm CPU-KV tier.
- **Do not claim:** full SuperInfer parity.
- **Time:** 1 minute.

## Slide 2 — Problem And Objective

- **Message:** Long-context concurrent serving stresses GPU KV capacity and latency.
- **Show:** HBM versus Grace memory imbalance diagram.
- **Evidence:** `00_PROJECT_OVERVIEW.md`, `02_BASELINE_AND_MOTIVATION.md`.
- **Time:** 1.5 minutes.

## Slide 3 — Hardware And Model

- **Message:** Two GH200s provide enormous compute and memory, but HBM is still finite.
- **Show:** GPU/NUMA/NVLink table.
- **Evidence:** `00_PROJECT_OVERVIEW.md`, `07_NUMA_AND_MEMORY_BEHAVIOUR.md`.
- **Time:** 1 minute.

## Slide 4 — Baseline Serving

- **Message:** Standard vLLM already solves much of serving; the bottleneck is capacity/latency under pressure.
- **Show:** baseline vLLM pipeline and historical concurrency table.
- **Evidence:** `02_BASELINE_AND_MOTIVATION.md`.
- **Time:** 1.5 minutes.

## Slide 5 — Why Aggregate Throughput Is Not Enough

- **Message:** Higher aggregate throughput can hide severe per-user latency degradation.
- **Show:** concurrency 4/8/16 table.
- **Say:** report throughput, TTFT, TPOT, success rate together.
- **Time:** 1 minute.

## Slide 6 — SuperInfer Idea

- **Message:** Treat CPU/Grace memory as a warm capacity tier.
- **Show:** GPU HBM ↔ Grace CPU memory diagram.
- **Evidence:** `03_SUPERINFER_ORIGINAL_IDEA.md`.
- **Time:** 1.5 minutes.

## Slide 7 — Why Forward-Porting Was Necessary

- **Message:** Old SuperInfer APIs could not be copied into modern vLLM V1.
- **Show:** old architecture → modern V1 landing zones.
- **Evidence:** `03_SUPERINFER_ORIGINAL_IDEA.md`, `01_STORY_AND_TIMELINE.md`.
- **Time:** 1.5 minutes.

## Slide 8 — Two Implementation Tracks

- **Message:** Separate the validated production track from the newer v0.26 research track.
- **Show:** two-column comparison.
- **Say:** reasoning tree has strongest accepted pressure evidence; v0.26 adds DSpark/high-risk research.
- **Time:** 1.5 minutes.

## Slide 9 — Forward-Port Architecture

- **Message:** Modern connector, scheduler, worker, queues, streams, and CPU-KV blocks form one pipeline.
- **Show:** architecture diagram.
- **Evidence:** `04_FORWARD_PORT_ARCHITECTURE.md`.
- **Time:** 2 minutes.

## Slide 10 — D2H/H2D Movement

- **Message:** Stores are compute-gated; loads occur before a request needs restored KV.
- **Show:** two directional timelines.
- **Evidence:** `04_FORWARD_PORT_ARCHITECTURE.md`, `SUPERINFER_KNOWLEDGE_BASE.md:347-385`.
- **Time:** 1.5 minutes.

## Slide 11 — NUMA And Topology

- **Message:** Topology-aware planning is implemented, but physical page locality remains an open measurement.
- **Show:** GPU0/NUMA0 and GPU1/NUMA1 map.
- **Do not claim:** measured NUMA performance improvement.
- **Evidence:** `07_NUMA_AND_MEMORY_BEHAVIOUR.md`.
- **Time:** 1.5 minutes.

## Slide 12 — Failure-Driven Engineering

- **Message:** Pressure exposed allocator/free-list invariants and lifecycle gaps.
- **Show:** symptom → diagnosis → fix → regression diagram.
- **Evidence:** `06_CORRECTNESS_AND_RELIABILITY.md`, `11_SERVER_HANG_POSTMORTEM.md`.
- **Time:** 2 minutes.

## Slide 13 — Reload Correctness

- **Message:** CPU-KV restore is real and deterministic for the accepted profile.
- **Show:** exact 131K table with zero mismatches.
- **Evidence:** `10_BENCHMARK_RESULTS.md`.
- **Time:** 1 minute.

## Slide 14 — Full GPU-KV Pressure

- **Message:** The accepted system reaches 100% GPU KV usage and still completes requests.
- **Show:** pressure ladder table.
- **Evidence:** `10_BENCHMARK_RESULTS.md`, `docs/100_reasoning_tree_offload_results.md`.
- **Time:** 2 minutes.

## Slide 15 — Strongest Result

- **Message:** 524K prompt / 10K output / 24 users / 48 requests passed.
- **Show:** large-number result card with stores, loads, quiescence, health.
- **Do not claim:** throughput improvement.
- **Time:** 1.5 minutes.

## Slide 16 — Newer v0.26/DSpark Track

- **Message:** Newer integration expands the frontier but has incomplete high-risk rollback proof.
- **Show:** v0.26 feature/status matrix.
- **Evidence:** `05_SUPERINFER_PARITY_MATRIX.md`.
- **Time:** 1.5 minutes.

## Slide 17 — Benchmark Methodology

- **Message:** Exact token lengths, success counts, transfer counters, and quiescence make results interpretable.
- **Show:** pressure/reload method flow.
- **Evidence:** `08_BENCHMARK_METHODOLOGY.md`, `09_METRIC_DEFINITIONS.md`.
- **Time:** 1.5 minutes.

Add one visual vocabulary box to this slide:

```text
1 bootstrap + N sequential pressure rounds
```

Use `17_PRESSURE_TEST_VOCABULARY.md` to define the distinction between one
round, two-round soak, five-round extended soak, and the separate 524K/10K
aggressive test.

## Slide 18 — Production Operations

- **Message:** The implementation includes repeatable startup, scoped shutdown, diagnostics, and optional recovery.
- **Show:** systemd → supervisor → start.sh → Docker → vLLM.
- **Evidence:** `12_DEPLOYMENT_AND_OPERATIONS.md`, `docs/110_systemd_service_guide.md`.
- **Time:** 1.5 minutes.

## Slide 19 — What Is Proven / Not Proven

- **Message:** Capability-level parity is more credible than an invented percentage.
- **Show:** parity matrix excerpt.
- **Evidence:** `05_SUPERINFER_PARITY_MATRIX.md`.
- **Time:** 1.5 minutes.

## Slide 20 — Current State And Next Steps

- **Message:** Correctness and pressure stability are ahead of causal performance attribution and full parity.
- **Show:** completed / partial / future columns.
- **Evidence:** `13_CURRENT_STATE_AND_NEXT_STEPS.md`.
- **Time:** 1.5 minutes.

## Slide 21 — Claims And Limitations

- **Message:** The strongest result is narrowly defined and defensible.
- **Show:** claims table with safe wording.
- **Evidence:** `15_CLAIMS_EVIDENCE_TABLE.md`.
- **Time:** 1 minute.

## Slide 22 — Closing

- **Message:** The project demonstrates a reliable path from GPU-only serving toward a measured GPU–Grace memory hierarchy.
- **Show:** final architecture and next experiment priorities.
- **Say:** the next high-value measurements are physical locality, transfer exposure, matched performance, DSpark rollback, and startup phases.
- **Time:** 1 minute.

## Presentation Flow

```text
WHY → BASELINE → BOTTLENECK → SUPERINFER IDEA
    → FORWARD PORT → ARCHITECTURE → FAILURES/FIXES
    → CORRECTNESS → PRESSURE RESULTS → NEWER TRACK
    → METHODOLOGY → OPERATIONS → CLAIM BOUNDARY → NEXT WORK
```
