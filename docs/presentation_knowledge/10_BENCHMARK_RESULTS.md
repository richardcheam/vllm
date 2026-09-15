# Benchmark Results

For exact definitions of a bootstrap request, workload warmup, pressure round,
multi-round soak, extended five-round soak, and the separate 524K/10K aggressive
test, see `17_PRESSURE_TEST_VOCABULARY.md`.

## Experiment A — Historical Baseline Scaling

Historical approximately-8K prompt baseline:

| Concurrency | Aggregate throughput | Approx. per-user | TTFT p50 |
|---:|---:|---:|---:|
| 4 | 352.2 tok/s | 88.0 tok/s/user | 1,594 ms |
| 8 | 724.1 tok/s | 90.5 tok/s/user | 1,717 ms |
| 16 | 624.4 tok/s | 39.0 tok/s/user | 13,253 ms |

This is motivation, not a matched comparison to later pressure runs.

## Experiment B — Reload Correctness

Protocol:

```text
4 exact 131,072-token target prompts
        ↓
8 distinct eviction prompts at concurrency 24
        ↓
repeat eviction + target reload 3 times
        ↓
compare deterministic output and API prompt-token counts
```

| Parameter | Result |
|---|---:|
| Prompt length | 131,072 exact tokenizer tokens |
| Store requests | 4 |
| Evictions/repeat | 8 |
| Repeats | 3 |
| Output mismatches | 0 |
| Prompt-length mismatches | 0 |
| Failures | 0 |
| Final health | true |

The client stores the first four target outputs, then compares each later reload
output against the corresponding original text. It does not collect transfer
counters itself. Direct D2H/H2D evidence must come from pressure-run metrics or
a separate metrics capture taken during this sequence.

## Experiment C — Full GPU-KV Pressure

| Run | Requests | Prompt | Output max | Stores/loads | GPU KV max | Final pending | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| One round | 24/24 | 327,680 | 256 | 18/2 | 100% | 0 | PASS |
| Two rounds | 48/48 | 327,680 | 256 | 36/4 | 100% | 0 | PASS |
| Three rounds | 72/72 | 327,680 | 256 | 53/6 | 100% | 0 | PASS |
| Five rounds | 120/120 | 327,680 | 256 | 90/10 | 100% | 0 | PASS |
| Aggressive | 48/48 | 524,288 | 10,000 | 26/2 | 100% | 0 | PASS |

The five-round run took approximately 56.8 minutes. The aggressive run took
approximately 81.8 minutes. These are maintenance-window tests.

## Strongest Accepted Result

```text
Exact prompt:        524,288 tokens
Maximum output:      10,000 tokens
Concurrency:         24
Requests:            48 over 2 rounds
GPU KV max:          100%
Minimum free blocks: 0
Stores/loads:        26 / 2
Store bytes:         71,981,445,888
Load bytes:          3,014,390,016
Stale capacity:      0 seconds
Final pending age:   0 ms
Final health:        true
```

## Experiment D — Transfer Backend

The newer v0.26 integration contains directional native-DMA evidence. One exact
reload probe had approximately 11.9% lower native mean reload time, but only
approximately 0.6% median improvement, with differing store volume. This is
directional, not a causal performance claim.

The validated reasoning-tree track does not claim a native/default throughput
win.

## Experiment E — NUMA

Topology discovery and worker binding are verified. Physical page placement and
controlled local-vs-remote performance are not verified. No NUMA speedup claim
is safe.

## Experiment F — Soak/Reliability

The accepted five-round reasoning-tree run completed 120/120 requests with
100% GPU KV pressure, 90 stores, 10 loads, zero stale-capacity time, and zero
pending transfers. The post-fix v0.26 integration long-context soak completed
48/48 with approximately 28.6 GB stores and 10.2 GB loads.

## Experiment G — Startup

One reasoning-tree startup log reports approximately 74.85 GiB model load per
worker, approximately 69 seconds of graph capture, approximately 40.79 GiB
available GPU KV memory, and approximately 2.74M GPU KV tokens. The newer
startup audit says cache persistence and AOT reuse remain unverified.

## Invalid Results

- Manually interrupted timeout runs are not EngineCore crash evidence.
- Native-offload control runs that hit upstream assertions are invalid controls.
- Character-truncated nominal prompt lengths are not exact-token results.
- Differing transfer volumes prevent causal backend speedup claims.
- A successful response without positive load counters does not prove reload.

## Sources

- `docs/100_reasoning_tree_offload_results.md:13-88,90-103`.
- `docs/080_gpu_validation_checklist.md`.
- `SUPERINFER_KNOWLEDGE_BASE.md:122-146,438-470`.
- `/home/az04297/re-SuperInfer/vllm-superinfer-v4-integration/docs/superinfer/validation-status.md`.
