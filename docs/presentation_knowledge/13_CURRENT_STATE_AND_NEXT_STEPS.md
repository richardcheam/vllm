# Current State And Next Steps

## COMPLETED / VERIFIED

- Modern vLLM V1 CPU-KV connector path exists.
- Real D2H stores and H2D reloads are observed.
- Exact reload profile has zero output mismatches.
- Accepted exact-token pressure reaches 100% GPU KV usage.
- 524K/10K aggressive profile completes 48/48.
- Allocator/free-list pressure fixes have regression coverage.
- Transfer quiescence is explicitly validated.
- Production startup, shutdown, diagnostics, and service-control paths exist.

## FUNCTIONAL BUT NEEDS MORE VALIDATION

- Logical GH200 local/remote topology planning.
- VLT-like proactive victim scoring.
- Native/default DMA parity in the newer v0.26 track.
- Large CPU-KV allocation modes.
- Server recovery under every external termination mode.
- Startup cache reuse and phase-level startup optimization.

## NOT YET VERIFIED

- Physical CPU page placement on intended NUMA nodes.
- Copy/compute overlap ratio.
- Scheduler-visible transfer stall.
- General throughput or latency improvement versus a matched baseline.
- Full high-risk DSpark rejection/rollback/restore correctness.
- Native block-first DeepSeek layout.

## NOT PORTED / FUTURE RESEARCH

- Native C++/CUDA/ZMQ DuplexKV.
- Complete paper-level RotaSched/LVF semantics.
- MoE expert offload as a separate capacity tier.
- Predictive prefetching based on measured future demand.
- Residency-authoritative scheduling if transfer exposure justifies it.

## Highest-Value Next Work

1. Measure physical NUMA page locality for CPU-KV allocations.
2. Measure transfer overlap and critical-path exposure.
3. Complete a matched baseline comparison with identical workloads and transfer volume.
4. Validate DSpark rejection/rollback under sustained high-risk rotation in the v0.26 track.
5. Build cold/warm/second-warm startup timing tables.
6. Verify persistent compile-cache behavior across container recreation.
7. Scale CPU-KV capacity gradually beyond 128 GiB only with memory and locality gates.

## What Is No Longer The Immediate Priority

- Copying old SuperInfer source line by line.
- Adding unvalidated block-first behavior for DeepSeek-V4/TP=2.
- Treating raw GPU memory or aggregate throughput as sufficient evidence.
- Broad host watchdog changes before server-local recovery is exercised.

## Sources

- `docs/100_reasoning_tree_offload_results.md:90-120,503-513`.
- `SUPERINFER_KNOWLEDGE_BASE.md:804-991`.
- `/home/az04297/re-SuperInfer/vllm-superinfer-v4-integration/docs/startup_optimization_audit.md`.
