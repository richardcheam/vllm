# Presentation Knowledge Base

This directory is the evidence-backed source for a roughly 30-minute Beamer
presentation on topology-aware DeepSeek serving and SuperInfer-inspired CPU-KV
offload on dual NVIDIA GH200 GPUs.

## Reading Order

1. `00_PROJECT_OVERVIEW.md`
2. `01_STORY_AND_TIMELINE.md`
3. `02_BASELINE_AND_MOTIVATION.md`
4. `03_SUPERINFER_ORIGINAL_IDEA.md`
5. `04_FORWARD_PORT_ARCHITECTURE.md`
6. `05_SUPERINFER_PARITY_MATRIX.md`
7. `06_CORRECTNESS_AND_RELIABILITY.md`
8. `07_NUMA_AND_MEMORY_BEHAVIOUR.md`
9. `08_BENCHMARK_METHODOLOGY.md`
10. `09_METRIC_DEFINITIONS.md`
11. `10_BENCHMARK_RESULTS.md`
12. `11_SERVER_HANG_POSTMORTEM.md`
13. `12_DEPLOYMENT_AND_OPERATIONS.md`
14. `13_CURRENT_STATE_AND_NEXT_STEPS.md`
15. `14_PRESENTATION_STORYBOARD.md`
16. `15_CLAIMS_EVIDENCE_TABLE.md`
17. `16_PRESENTATION_DATA_GAPS.md`
18. `17_PRESSURE_TEST_VOCABULARY.md`

## Project Summary

The project serves `DeepSeek-V4-Flash-0731` on two NVIDIA GH200 144 GB HBM3e
GPUs using tensor parallelism. The core systems problem is that GPU HBM is the
hot, latency-sensitive tier while Grace/CPU memory provides a much larger warm
capacity tier. The engineering work forward-ports selected SuperInfer ideas to
modern vLLM V1 interfaces: CPU-KV storage, asynchronous D2H/H2D movement,
bounded queues, topology metadata, guarded proactive movement, capacity-aware
scheduling, and correctness/lifecycle instrumentation.

The strongest accepted evidence comes from the
`vllm-modern-reasoning` checkout: exact-token pressure reached 100% GPU KV
usage, performed real CPU-KV stores and loads, completed all requests, and
left the server healthy. A separate newer `vllm-superinfer-v4-integration`
checkout targets vLLM v0.26.0 and adds DSpark/high-risk and native-DMA research,
but its high-risk rollback and final saturation claims remain incomplete.

## Current Status

- CPU-KV D2H stores: verified in accepted pressure and reload runs.
- CPU-KV H2D reloads: verified with exact output correctness.
- Full-pressure correctness: verified for the documented DS4-0731 profile.
- Full SuperInfer parity: not proven.
- Physical NUMA page placement: not verified.
- Causal throughput improvement: not claimed by the validated production tree.
- v0.26 DSpark high-risk proactive rotation: newer research track, not final production evidence.
- User-level systemd recovery design: implemented and installed disabled/inactive for controlled handoff.

The exact meaning of “round,” “soak,” “extended soak,” “bootstrap,” and the
524K/10K aggressive test is defined in `17_PRESSURE_TEST_VOCABULARY.md`.

## Evidence Rules

- Every benchmark number must include workload, prompt/output lengths, concurrency, request count, and configuration.
- Exact-token pressure and deterministic reload are separate validation questions.
- A successful response does not by itself prove H2D reload; transfer counters are required.
- Logical NUMA locality is not the same as measured physical page placement.
- Historical, directional, smoke, verified, and future-work claims are labeled separately.

## Source Checkouts

Primary validated track:

```text
/home/az04297/re-SuperInfer/vllm-modern-reasoning
```

Newer comparison/research track:

```text
/home/az04297/re-SuperInfer/vllm-superinfer-v4-integration
```

Initial context documents supplied for reconciliation:

```text
/home/az04297/SUPERINFER_KNOWLEDGE_BASE.md
/home/az04297/ask.md
```
