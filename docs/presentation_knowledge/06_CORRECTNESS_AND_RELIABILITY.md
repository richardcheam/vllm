# Correctness And Reliability

## Reload Correctness

The deterministic reload profile used exact tokenizer lengths and three
explicit phases. The client does not start or stop the server.

### Test Construction

The accepted settings were:

```text
target prompt length:   exactly 131,072 tokenizer tokens
target store prompts:   4
eviction prompts:       8 distinct prompts
eviction/reload repeats: 3
store concurrency:      1
eviction concurrency:   24
reload concurrency:     4
maximum output:         64 tokens
temperature:            0.0
seed:                   20260813
```

Each target prompt was built from a unique prefix such as:

```text
Reload validation prompt 0.
```

followed by deterministic tokenizer filler until exactly 131,072 token IDs were
reached. Eviction prompts used a separate index range beginning at `10000`, so
they did not reuse the target prefix sequences.

### Phase Sequence

```text
Store phase:
  send target prompts 0..3 sequentially
        ↓
Repeat 1..3:
  send 8 distinct eviction prompts concurrently
        ↓
  resend target prompts 0..3 concurrently
        ↓
  compare reload text against original store text
```

The eviction phase is what attempts to remove the target KV blocks from GPU
residency. The reload phase then asks for the same target sequences, allowing
the CPU-KV lookup/load path to be exercised.

| Field | Result |
|---|---:|
| Prompt length | exactly 131,072 tokens |
| Store requests | 4 |
| Eviction requests/repeat | 8 |
| Repeats | 3 |
| Store concurrency | 1 |
| Eviction concurrency | 24 |
| Reload concurrency | 4 |
| Maximum output | 64 tokens |
| Output mismatches | 0 |
| Prompt-length mismatches | 0 |
| Request failures | 0 |
| Final server healthy | true |

### What Zero Mismatches Proves

For every reload response, the harness compares the returned completion text
with the corresponding text from the original store phase. It also checks
`usage.prompt_tokens` for every phase result.

Therefore the accepted result proves:

- the target prompts were exactly 131,072 tokens as observed by the API;
- all store, eviction, and reload requests returned successfully;
- reloading the target prefixes produced identical deterministic output;
- the server remained healthy after the sequence.

The reload client explicitly records:

```text
transfer_counters_collected: false
```

So zero mismatches alone does **not** prove that the reload used CPU-KV rather
than recomputation. Direct transfer evidence comes from the pressure harness'
Prometheus deltas, or from a separate metrics capture taken during reload.

For the broader accepted evidence, pressure runs recorded positive store/load
events and bytes, while the reload run established exact deterministic output.
These are complementary proofs, not one combined counter from the reload client.

## Pressure Correctness

Accepted pressure runs required:

- all intended requests complete successfully;
- exact prompt token lengths;
- real store/load activity when required;
- GPU KV reaches the intended pressure condition;
- stale-capacity time remains within the gate;
- pending transfer queues drain;
- final `/health` succeeds.

Strongest accepted result:

```text
prompt:                  exactly 524,288 tokens
maximum output:          10,000 tokens
concurrency:             24 users
requests:                48 across 2 rounds
GPU KV usage max:        100%
minimum free GPU blocks: 0
stores/loads:            26 / 2
final pending transfers: 0
final transfer age:      0 ms
server healthy:          true
```

This is correctness/stability evidence for one aggressive profile, not a
general performance claim.

## Failure: Allocator And Free-List Pressure

Early pressure runs reached full GPU KV pressure and then EngineCore failed.
Later runs performed stores/loads but exposed linked-list and free-block
counter inconsistencies.

The pressure workload exposed paths where a late grouped allocation failure
could leave partially updated request/block state. Separate failures involved
stale free-list counters, duplicate insertion, and unsafe list movement.

Fixes included:

- grouped allocation preflight and rollback;
- request-table and reference-count restoration;
- stale free-list counter validation;
- duplicate-safe block freeing;
- atomic `popleft_n()` behavior including fake head/tail links.

Focused core/scheduler tests passed, followed by accepted exact-token pressure
runs reaching full GPU KV utilization with no EngineCore death.

**Evidence:** `docs/100_reasoning_tree_offload_results.md:358-395`;
`vllm/v1/core/kv_cache_utils.py`; `vllm/v1/core/block_pool.py`;
`vllm/v1/core/kv_cache_coordinator.py`; `vllm/v1/core/kv_cache_manager.py`.

## Failure: Transfer Quiescence / Idle Engine

An asynchronous D2H store could remain pending after the final client response.
The engine could treat the request set as empty before worker completion
metadata arrived.

The fix added `has_pending_push_work()` and made the harness require queue and
pending-transfer quiescence after each phase.

Post-fix exact-token long-context soak completed 48/48 with stores, loads, no
EngineDead events, no CUDA OOM, no Xid errors, and quiescent queues.

**Evidence:** `/home/az04297/re-SuperInfer/vllm-superinfer-v4-integration/docs/superinfer/validation-status.md:232-250`.

## Lifecycle Reliability

The production tree now includes:

- API-parent-first `SIGTERM` shutdown;
- descendant and orphan EngineCore/worker discovery;
- process-local distributed/CUDA/CPU-KV/shared-memory cleanup;
- bounded `SIGKILL` escalation;
- startup warmup before `READY`;
- failure-only read-only diagnostics;
- server-local API PID/exit-code/signal records;
- optional user systemd supervision with development maintenance hold.

The `resource_tracker` messages are treated as IPC shutdown bookkeeping. They
are investigated with repeated `/dev/shm` snapshots, not by deleting live
shared-memory objects.

## Current Confidence

The strongest claim is:

> The validated profile survives repeated exact-token GPU-KV pressure and
> deterministic reload, with observed CPU-KV movement, drained transfers, and
> healthy post-run service state.

Do not upgrade this to “production proven for all workloads” or “no possible
memory leak.”

## Sources

- `docs/100_reasoning_tree_offload_results.md:13-29,68-103,358-395`.
- `docs/reasoning_tree_8202_canary.md`.
- `docs/110_systemd_service_guide.md`.
- `/home/az04297/re-SuperInfer/vllm-superinfer-v4-integration/docs/superinfer/validation-status.md:115-157,232-250`.
