# Benchmark Methodology

## Principle

Every result must identify the workload that generated it. A throughput number
without prompt length, output budget, concurrency, and success count is not
interpretable.

## Workload Dimensions

Record model/checkpoint, vLLM revision, hardware/container, TP/PP/EP, actual
tokenized prompt length, output budget, concurrency, request count, prefix
caching, KV dtype/block size, GPU utilization, CPU-KV capacity, backend/profile,
warmup, and deadlines.

## Validation Modes

| Mode | Question |
|---|---|
| Baseline/concurrency | How does normal serving scale as users increase? |
| Reload | Do stored CPU prefixes restore exactly? |
| Pressure | Does CPU-KV movement preserve correctness at full GPU-KV pressure? |
| Soak | Does repeated pressure remain healthy over time? |
| Transfer backend | Does native/default DMA differ under matched workload? |
| NUMA | Is physical locality measured, and does it affect performance? |
| Startup | Which initialization phase consumes time and memory? |
| Lifecycle | Do shutdown, restart, cancellation, and transfer drain behave safely? |

## Exact Token Lengths

The pressure harness builds prompts with the local tokenizer and verifies the
observed prompt length. This avoids confusing a nominal “327K” request with a
shorter character-truncated prompt.

```text
requested tokenizer length
        ↓
generated exact-length prompt
        ↓
API usage verification
        ↓
accept/reject run
```

## Valid Run Criteria

A pressure run is accepted only when all intended requests succeed, exact
prompt lengths are observed, expected CPU movement occurs, GPU KV reaches the
intended pressure, stale-capacity time passes the gate, pending transfers
drain, final health succeeds, and no unrelated GPU occupancy contaminates the
run.

## Invalid Run Criteria

Reject or relabel runs when prompt lengths were estimated, client deadlines
caused cancellation, transfer volumes differ in a backend comparison, a profile
crashes during startup, a control backend hits an upstream assertion, GPU memory
is occupied by another service, or the run was manually interrupted before the
acceptance gate.

## Pressure Method

```text
load tokenizer → wait for health → bootstrap
        ↓
build unique exact-token prompts
        ↓
run bounded concurrent rounds
        ↓
sample /metrics every 2 seconds
        ↓
check requests, KV pressure, stores/loads, stale waits, quiescence, health
```

## Reload Method

```text
store exact prefixes → concurrent distinct eviction
        ↓
repeat target prefix requests
        ↓
check exact prompt length and output equality
        ↓
use pressure metrics for direct transfer counters
```

Reload correctness and pressure transfer evidence are intentionally separate.

## Sources

- `scripts/run_heavy_offload_soak.py`.
- `scripts/validate_offload_reload.py`.
- `docs/heavy_offload_staging.md:34-130`.
- `docs/100_reasoning_tree_offload_results.md:217-250`.
- `SUPERINFER_KNOWLEDGE_BASE.md:535-568`.
