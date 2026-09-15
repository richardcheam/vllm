# Reasoning-Tree SuperInfer Offload Results

Date: 2026-09-03

## Executive Summary

The reasoning-tree vLLM forward-port was validated in the
`richard-base-dev-sysnice` container with the DeepSeek-V4-Flash-0731 checkpoint.
The validated configuration uses modern vLLM V1 interfaces, the
`SimpleCPUOffloadConnector`, 128 GiB total CPU KV capacity, FP8 GPU KV cache,
NUMA binding, and guarded proactive swapping.

The final pressure validation reached 100% GPU KV utilization, performed real
CPU-KV stores and loads, completed every request, and left the server healthy:

| Test | Result | Requests | Stores / loads | GPU KV max | Stale wait | Pending transfers |
|---|---:|---:|---:|---:|---:|---:|
| One pressure round | PASS | 24/24 | 18 / 2 | 100% | 0 s | 0 |
| Two pressure rounds | PASS | 48/48 | 36 / 4 | 100% | 0 s | 0 |
| Three pressure rounds | PASS | 72/72 | 53 / 6 | 100% | 0 s | 0 |
| Five pressure rounds, extended timeout | PASS | 120/120 | 90 / 10 | 100% | 0 s | 0 |
| 524K prompt / 10K output, two rounds | PASS | 48/48 | 26 / 2 | 100% | 0 s | 0 |
| Deterministic reload | PASS | all phases passed | not collected by client | n/a | n/a | n/a |

The two-round pressure result is the primary acceptance evidence. It supports
the narrower claim that the candidate survives repeated full GPU KV pressure
with CPU-KV movement and without the previously observed EngineCore death or
stale-capacity condition. It is not a claim of complete SuperInfer parity or a
performance improvement.

## Detailed Runtime Results

The pressure harness sampled `/metrics` every two seconds. Values below are
run-level deltas or extrema from those samples; transfer byte values are raw
bytes, not token counts.

| Run | Samples | GPU KV blocks | Free blocks min / final | GPU KV usage max | Waiting max | Running max | Store events | Load events | Store bytes | Load bytes | Final pending |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pressure_320k_retry2` | 337 | 41642 | 0 / 37723 | 100% | 23 | 21 | 18 | 2 | 109055498112 | 1010756736 | 0 |
| `pressure_320k_soak2` | 671 | 41642 | 0 / 37724 | 100% | 23 | 21 | 36 | 4 | 218110996224 | 2021513472 | 0 |
| `pressure_320k_soak3` | 1007 | 41642 | 0 / 37724 | 100% | 23 | 21 | 53 | 6 | 312391145088 | 3285748224 | 0 |
| `pressure_320k_soak5_timeout2400` | 1677 | 41639 | 0 / 36414 | 100% | 23 | 21 | 90 | 10 | 545088170880 | 5053783680 | 0 |
| `pressure_524k_agentic10k_2round` | 2414 | 41639 | 0 / 35299 | 100% | 23 | 13 | 26 | 2 | 71981445888 | 3014390016 | 0 |

The corresponding local NUMA transfer byte deltas matched the total transfer
bytes in both accepted runs. Remote transfer bytes and remote fallback counts
were zero. Final pending transfer age was `0 ms` in both runs, and the server
health endpoint remained available after completion.

The three-round and extended five-round runs showed the same behavior. The
five-round run used a `2400`-second per-request timeout and completed all 120
requests. This is important because an earlier five-round attempt used a
`900`-second timeout and was manually interrupted after many client-side
timeouts; that earlier run was not an engine-crash result and is not acceptance
evidence.

The five-round extended run completed in approximately `3409` seconds, or
`56.8` minutes, with a `2400`-second request timeout and `3600`-second stage
timeout. This is an important operational limitation: the profile is stable,
but a full five-round test is not a short smoke test and should not be run on a
user-facing service without a maintenance window.

The final aggressive run used `524288` prompt tokens (`2^19`, aligned to 2048
configured 256-token blocks) and a `10000`-token maximum output. It completed
in approximately `4906` seconds, or `81.8` minutes, with no request failures.
This is an extreme offload/capacity test rather than a short readiness check.

The reload validator reported:

| Parameter | Result |
|---|---:|
| Prompt length requested | `131072` tokens |
| Observed prompt length range | `131072` to `131072` tokens |
| Store requests | `4` |
| Eviction requests per repeat | `8` |
| Evict/reload repeats | `3` |
| Store concurrency | `1` |
| Eviction concurrency | `24` |
| Reload concurrency | `4` |
| Maximum output | `64` tokens |
| Output mismatches | `0` |
| Prompt-length mismatches | `0` |
| Request failures | `0` |
| Server healthy after completion | `true` |

The reload client does not collect transfer counters. Its result proves
deterministic response and exact tokenizer-length behavior; the pressure
harness is the source for direct store/load transfer evidence.

## Claim Boundary

| Claim | Status | Evidence |
|---|---|---|
| Server starts in the NUMA-capable validation container | Proven | Startup completed and three warmup requests passed |
| Short requests generate valid responses | Proven | Startup warmup and all pressure responses returned successfully |
| CPU-KV offload is actually activated | Proven | Positive store/load event and byte deltas during 327K pressure |
| Offload survives full GPU KV pressure | Proven for this profile | Two 24-request rounds reached 100% GPU KV usage and completed 48/48 |
| Stale-capacity condition occurred during final validation | Not observed | `max_stale_capacity_seconds=0` |
| Allocator/free-list crash is fixed | Validated for this profile | Final pressure runs passed after the fixes |
| 524K prompt with 10K output survives repeated pressure | Proven for this profile | Two rounds completed `48/48` with 100% GPU KV usage and drained transfers |
| Reload output remains deterministic | Proven for the reload profile | Exact 131072-token reload run had zero mismatches |
| Full SuperInfer feature parity | Not proven | DSpark/empty-allocation/native block-first remain unsupported or unported |
| Throughput or latency improvement | Not measured | No controlled vanilla-vs-SuperInfer comparison was run |

## Approach

1. Port the safest useful SuperInfer behavior onto modern vLLM V1 interfaces
   instead of copying the old swapper implementation.
2. Use `SimpleCPUOffloadConnector` for CPU-KV storage and modern worker
   lifecycle hooks.
3. Keep DeepSeek on the safe GPU-derived KV layout. Block-first remains opt-in
   and is disabled for the validated DeepSeek profile.
4. Add guarded proactive movement for waiting-work pressure, with refcount-aware
   ownership checks and CPU-capacity limits.
5. Make capacity telemetry distinguish ordinary queueing from true full-sequence
   capacity waits and expose pending transfer age.
6. Exercise pressure with exact tokenizer-built prompt lengths rather than
   estimating logical capacity from raw GPU block counts.
7. Keep reload validation separate from pressure validation so output correctness
   is not confused with transfer-counter evidence.

## Validated Configuration

```text
container: richard-base-dev-sysnice
checkout: /workspace/re-SuperInfer/vllm-modern-reasoning
model: DeepSeek-V4-Flash-0731
tensor parallel: 2
gpu memory utilization: 0.92
max model length: 1048576
max sequences: 24
max batched tokens: 32768
block size: 256
GPU KV dtype: fp8
CPU KV capacity: 128 GiB total, approximately 64 GiB per TP rank
proactive swap budget: 2400 blocks
NUMA binding: nodes 0 and 1
high-risk mode: disabled
block-first mode: disabled
speculative decoding: disabled in this tree
```

### Model And Runtime Specification

The server log identified the model as `DeepseekV4ForCausalLM` with the
DeepSeek-V4 FP8 quantization path. The relevant runtime properties were:

| Area | Effective setting |
|---|---|
| GPUs | 2x NVIDIA GH200 144 GB HBM3e |
| vLLM build | `0.20.2.dev0+g132765e35.d20260607` |
| Model | DeepSeek-V4-Flash-0731 local snapshot |
| Model dtype | `bfloat16` |
| Quantization | `deepseek_v4_fp8` |
| Tensor parallelism | `2` |
| Pipeline parallelism | `1` |
| Expert parallelism | enabled |
| Custom all-reduce | disabled |
| Tokenizer mode | `deepseek_v4` |
| Reasoning parser | `deepseek_v4` |
| Tool parser | `deepseek_v4` |
| Auto tool choice | enabled |
| Prefix caching | enabled |
| Chunked prefill | enabled |
| Max model length | `1048576` tokens |
| Max sequences | `24` |
| Max batched tokens | `32768` |
| GPU KV dtype | `fp8` |
| KV block size | `256` tokens |
| CUDA graph mode | `FULL_AND_PIECEWISE` |
| CUDA graph capture limit | `128` |
| FP4 indexer cache | disabled |
| Async scheduling | enabled by the effective engine configuration |
| Speculative decoding | disabled; no `--speculative-config` was passed |

### CPU-KV And Scheduler Specification

| Area | Effective setting |
|---|---|
| Connector | `SimpleCPUOffloadConnector` |
| Connector mode | lazy offload |
| CPU KV capacity | `128 GiB` total |
| CPU KV per TP rank | approximately `64 GiB` |
| CPU allocation mode | current-tree default allocation; reference `empty` mode is unavailable |
| Proactive swap budget | `2400` blocks |
| VLT alpha | `3` |
| VLT bandwidth coefficient | `1` |
| VLT future coefficient | `1` |
| SLO TTFT | `5` |
| SLO TBT | `0.1` |
| Capacity wait timeout | `30` seconds |
| High-risk mode | disabled |
| Swapper block-first | disabled |
| Pin-memory fix | enabled |
| GH200 topology tuning | enabled |
| NUMA binding | enabled, nodes `0` and `1` |
| Local CPU pool fraction | `0.75` |
| Local swap bandwidth hint | `966367641600` bytes/s |
| Remote swap bandwidth hint | `300647710720` bytes/s |

The effective command is generated by `start.sh` and recorded in the `.cmd`
sidecar for each launch. It uses the checked-out source explicitly:

```text
.venv/bin/python -m vllm.entrypoints.cli.main serve
```

The startup script sends three short sequential API warmup requests after the
health endpoint becomes ready, and reports `READY` only after all enabled
requests succeed. They prime the tokenizer, model request path, CUDA execution,
and short decode path for initial users without adding large-context KV
pressure. Warmup requests are operational only and are excluded from pressure
acceptance counts. Launch, readiness, and warmup failures invoke the scoped
`stop.sh` cleanup path so a failed detached launch does not remain on the port.

The production checkout intentionally does not include the older multi-stage
daily observability/warmup wrappers. Metrics are collected directly by the
pressure harness and can be inspected from the vLLM `/metrics` endpoint during
normal serving.

## Harness Specification

The pressure client is `scripts/run_heavy_offload_soak.py`. It performs these
steps:

1. Load the tokenizer from the explicit local snapshot.
2. Wait for `/health`.
3. Send one tiny bootstrap request so scheduler capacity gauges are populated.
4. Read `vllm:kv_cache_total_blocks` from Prometheus. The value is recorded for
   diagnostics only; it is not multiplied by block size to infer logical token
   capacity.
5. Build unique prompts with exactly the requested tokenizer length.
6. Run bounded concurrent pressure stages.
7. Sample Prometheus metrics every two seconds.
8. Verify request success, exact prompt-token usage, stale capacity, pending
   transfers, observed transfer activity, and final server health.

The accepted pressure shape was:

| Parameter | Value |
|---|---:|
| Concurrent users | `24` |
| Requests per round | `24` |
| Pressure rounds | `1` or `2` |
| Prompt length | exactly `327680` tokenizer tokens |
| Maximum output | `256` tokens |
| Request deadline | `900` seconds |
| Stage deadline | `1800` seconds for one round, `2400` seconds for the two-round soak |
| Workload warmup requests | `0`; only the tiny bootstrap request ran |

The separate reload client is `scripts/validate_offload_reload.py`. It uses
four exact `131072`-token store prompts, eight distinct eviction prompts, three
evict/reload repeats, store concurrency `1`, eviction concurrency `24`, reload
concurrency `4`, and maximum output `64` tokens.

## Metric Definitions

| Metric | Meaning |
|---|---|
| `kv_cache_total_blocks` | Usable GPU KV blocks reported by the scheduler; raw layout capacity, not logical token capacity |
| `kv_cache_free_blocks_min` | Lowest observed number of free GPU KV blocks during the run |
| `kv_cache_usage_max` | Highest observed GPU KV usage fraction |
| `offload_store_events` | Delta of completed CPU-KV store events over the sampled run |
| `offload_load_events` | Delta of completed CPU-KV load events over the sampled run |
| `offload_store_bytes` | Delta of CPU-KV bytes moved GPU to CPU |
| `offload_load_bytes` | Delta of CPU-KV bytes moved CPU to GPU |
| `local_swap_out_bytes` | GPU-to-CPU bytes assigned to the local NUMA pool |
| `local_swap_in_bytes` | CPU-to-GPU bytes assigned to the local NUMA pool |
| `waiting_max` | Maximum aggregate scheduler waiting count observed |
| `running_max` | Maximum aggregate scheduler running count observed |
| `max_stale_capacity_seconds` | Longest sampled interval with `running=0`, true `capacity` waiting, and no progress |
| `pending_transfers_final` | Final pending store events plus pending load requests |
| `pending_transfer_age_ms_final` | Final oldest pending transfer age |
| `server_healthy_final` | Whether `/health` remained available after all pressure stages |

The `capacity` waiting reason is intentionally narrower than ordinary queue
waiting. It represents requests blocked by the full-sequence KV admission
check, not every request waiting because of token budget, sequence limits,
chunked prefill, or transient connector constraints.

## Prompt Prefix And Token Semantics

A prompt prefix is the beginning of the actual prompt token sequence, not a
request number. The harness constructs text containing a request-specific
identifier, tokenizes it with the same local tokenizer used by the server, and
repeats deterministic filler text until the requested token length is reached.
It sends the resulting token IDs directly in the completion request.

For example, pressure prompts have a shape similar to:

```text
Staging offload request 17 seed 20260825. <repeated deterministic filler>
```

The request ID makes each pressure prompt unique. This deliberately minimizes
prefix-cache reuse and creates new KV data for offload pressure. The reload
client instead rebuilds the same four original token sequences after sending
different eviction prompts, allowing the CPU-resident prefixes to be found and
loaded back.

The pressure acceptance check requires the server-reported `prompt_tokens` to
equal the requested length. The accepted pressure runs used exactly `327680`
prompt tokens; reload used exactly `131072` prompt tokens.

## GPU VRAM Accounting

`nvidia-smi` process memory is broader than the logical KV-cache metrics. The
startup-to-active-request increase observed during validation is expected to
include runtime allocations such as:

- model weights and CUDA context state;
- NCCL communication buffers;
- CUDA graph pools;
- DeepGEMM/MoE and attention workspaces;
- temporary activation tensors;
- allocator reservations and cached blocks;
- populated and referenced KV blocks.

Therefore, the `0.92` GPU memory utilization setting is a vLLM budgeting input,
not a hard ceiling on the value shown by `nvidia-smi`. A higher process-memory
value after requests begin is not by itself a leak. The stronger leak signal is
monotonic growth across drained runs while GPU KV usage, pending transfers, and
request ownership have returned to baseline.

Interpret VRAM and KV metrics together:

| Observation | Interpretation |
|---|---|
| `kv_cache_used_blocks` rises and free blocks fall | Logical GPU KV occupancy is rising |
| `offload_store_bytes` rises | GPU-to-CPU KV movement is active |
| `offload_load_bytes` rises | CPU-to-GPU KV reload is active |
| `nvidia-smi` rises while KV metrics are stable | Workspace, graph, communication, or allocator reservation is likely |
| memory remains high after a drained request set | Could be CUDA allocator reuse; investigate only if it grows monotonically across runs |

The accepted pressure runs reached zero free logical KV blocks, drained pending
transfers to zero, and left the server healthy. Those observations are more
useful for offload correctness than the absolute `nvidia-smi` number alone.

## Problems Found

### NUMA permission failure

The ordinary `richard-base-dev` container rejected `get_mempolicy` and
`set_mempolicy` with `Operation not permitted`. EngineCore exited before model
startup completed. The sysnice container permits NUMA binding and is the
validated runtime target.

### Insufficient pressure is not offload validation

The 196K and 262K workloads completed successfully but did not activate CPU-KV
offload because they remained above the proactive free-block watermark. A clean
result with zero transfer counters is a stability control, not proof of
offload.

### Allocator/free-list failure at full pressure

The first 320K pressure attempts reached full GPU KV usage and failed while
CPU-KV stores/loads were active. The observed failures were:

```text
ValueError: Cannot get 1 free blocks from the pool
AssertionError in FreeKVCacheBlockQueue.popleft_n()
```

The failures killed EngineCore and left worker processes holding GPU memory
after the API server exited.

## Validation Chronology

| Phase | Workload | Outcome | Interpretation |
|---|---|---|---|
| Control | 196K prompt pressure | 48/48 completed, no transfers | Stable, but below offload trigger |
| Control | 262K prompt pressure | 48/48 completed, no transfers | Stable, but below offload trigger |
| Pre-fix stress | 327K prompt pressure | EngineCore died at block exhaustion | Exposed the allocator/free-list bug |
| Post-first-fix stress | 327K prompt pressure | Stores/loads occurred, then EngineCore died | Exposed linked-list/counter inconsistency |
| Final pressure | 327K, one round | 24/24 passed | Offload and allocator fix validated |
| Final soak | 327K, two rounds | 48/48 passed | Repeated full-pressure validation |
| Extended soak | 327K, five rounds, 2400s request timeout | 120/120 passed | Longer-duration pressure validation |
| Final aggressive soak | 524K prompt, 10K output, two rounds | 48/48 passed | Heaviest completed prompt/output profile |
| Reload | 131K exact prompts, 3 repeats | PASS, zero mismatches | Deterministic reload validation |

## Fixes

- Added bounded full-sequence capacity handling and truthful waiting-reason
  metrics.
- Added pending load/store/transfer age telemetry.
- Added process-local distributed, CUDA, host allocator, shared-memory, and
  offload cleanup on worker shutdown.
- Made grouped KV allocation recoverable and transactional, restoring request
  tables, block references, and per-manager state after a late allocation miss.
- Hardened the free-block queue against stale counters and duplicate free-list
  insertion.
- Updated `stop.sh` to send `SIGTERM` to the API parent first, allowing vLLM's
  process-local engine, distributed, CUDA, shared-memory, and CPU-KV cleanup to
  run before signaling the captured process tree.
- The stop path also finds orphaned `VLLM::EngineCore` and `VLLM::Worker_*`
  processes by executable name and reasoning-tree working directory. It uses a
  30-second API-parent grace period, then a 120-second process-tree grace period
  by default, with `SIGKILL` only as a last resort.
- The shell does not perform a separate transfer-counter drain: the worker-side
  CPU-KV shutdown synchronizes in-flight CUDA copy events. PID/PGID state files
  are not used because the detached Docker launcher does not create matching
  state for this checkout; cleanup remains scoped by container, port, command,
  and working directory.
- Changed launchers to use the checked-out module invocation:
  `.venv/bin/python -m vllm.entrypoints.cli.main serve`.
- Added exact-length pressure and reload clients with bounded deadlines and
  failure summaries.

## Evidence

Raw runtime artifacts are intentionally excluded from Git. They are large and
diagnostic bundles may contain sensitive container metadata. The sanitized
review record is:

```text
pressure_320k_retry2:
  24/24 successful
  18 store events, 2 load events
  109055498112 store bytes, 1010756736 load bytes
  100% GPU KV usage, 0 seconds stale capacity
  0 pending transfers, 0 ms final transfer age, server healthy

pressure_320k_soak2:
  48/48 successful
  36 store events, 4 load events
  218110996224 store bytes, 2021513472 load bytes
  100% GPU KV usage, 0 seconds stale capacity
  0 pending transfers, 0 ms final transfer age, server healthy

pressure_320k_soak3:
  72/72 successful
  53 store events, 6 load events
  312391145088 store bytes, 3285748224 load bytes
  100% GPU KV usage, 0 seconds stale capacity
  0 pending transfers, 0 ms final transfer age, server healthy

pressure_320k_soak5_timeout2400:
  120/120 successful
  90 store events, 10 load events
  545088170880 store bytes, 5053783680 load bytes
  100% GPU KV usage, 0 seconds stale capacity
  0 pending transfers, 0 ms final transfer age, server healthy

pressure_524k_agentic10k_2round:
  48/48 successful
  26 store events, 2 load events
  71981445888 store bytes, 3014390016 load bytes
  100% GPU KV usage, 0 seconds stale capacity
  0 pending transfers, 0 ms final transfer age, server healthy

reload_validation:
  exact prompt length: 131072 tokens
  store requests: 4
  eviction requests: 8
  reload repeats: 3
  output mismatches: 0
  prompt-length mismatches: 0
  failures: 0
  server healthy after completion: true
```

## Reproduction Commands

Start the validated server:

```bash
CONTAINER_NAME=richard-base-dev-sysnice PORT=8202 \
  bash /home/az04297/re-SuperInfer/vllm-modern-reasoning/start.sh
```

Run one bounded pressure round:

```bash
docker exec -i richard-base-dev-sysnice bash -lc '
  cd /workspace/re-SuperInfer/vllm-modern-reasoning
  .venv/bin/python scripts/run_heavy_offload_soak.py \
    --url http://127.0.0.1:8202 \
    --model deepseek-ai/DeepSeek-V4-Flash-0731 \
    --tokenizer-path /workspace/re-SuperInfer/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062 \
    --users 24 \
    --pressure-requests 24 \
    --pressure-rounds 1 \
    --pressure-prompt-tokens 327680 \
    --expected-pressure-prompt-tokens 327680 \
    --max-prompt-tokens 327680 \
    --max-tokens 256 \
    --request-timeout-seconds 900 \
    --stage-timeout-seconds 1800 \
    --output-dir heavy_offload_artifacts/pressure_320k
'
```

For the extended duration validation, keep the same profile and change only the
round count and client deadlines:

```text
--pressure-rounds 5
--request-timeout-seconds 2400
--stage-timeout-seconds 3600
```

The accepted five-round run used 120 total requests, `327680` prompt tokens,
and `256` maximum output tokens. For agentic coding evaluation, run a separate
generation-heavy variant with `--max-tokens 4096`; do not compare its transfer
volume directly with the 256-token baseline without recording the changed
generation budget.

Stop the server and clean orphan workers if needed:

```bash
CONTAINER_NAME=richard-base-dev-sysnice PORT=8202 \
  bash /home/az04297/re-SuperInfer/vllm-modern-reasoning/stop.sh
```

## Remaining Limitations

- The current tree does not implement native DSpark speculative decoding.
- The current tree does not implement the reference branch's
  `CPU_KV_ALLOCATION_MODE=empty` option.
- DeepSeek uses the GPU-derived CPU KV layout; native block-first and full old
  DuplexKV semantics remain unported.
- The reload client validates exact prompt lengths and deterministic output, but
  does not collect direct transfer counters. Transfer activity must be checked
  in pressure-run Prometheus snapshots.
- The validated profile demonstrates correctness and bounded pressure behavior,
  not throughput, TTFT, TBT, or vanilla-vs-SuperInfer performance gains.
- Broader concurrency profiles, multi-node behavior, and longer-duration soak
  remain open.

## Review Conclusion

The reasoning-tree candidate is ready for selective Git review as a correctness-
and observability-focused change. The evidence supports the claim that modern
CPU-KV offload survives repeated full GPU KV pressure in the sysnice container
without the previously observed EngineCore death or stale-capacity condition.
It should not yet be presented as complete SuperInfer feature parity or as a
performance release.
