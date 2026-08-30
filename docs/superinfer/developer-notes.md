# SuperInfer Developer Notes

This is a code-oriented guide to the current v0.26-compatible SuperInfer port.
It describes what is implemented, where to inspect it, and how to reason about
performance. It is not a claim of full paper/native SuperInfer parity.

## Runtime Flow

1. `scripts/launch_superinfer.sh` resolves the profile into vLLM CLI flags.
2. `vllm/config/vllm.py:_post_init_kv_transfer_config()` converts
   `SWAP_CPU_MEMORY_GB` into `SimpleCPUOffloadConnector` configuration.
3. `vllm/v1/core/sched/scheduler.py:schedule()` admits requests, allocates GPU
   blocks, and optionally performs guarded proactive rotation.
4. `SimpleCPUOffloadConnector.build_connector_meta()` calls
   `SimpleCPUOffloadScheduler.build_connector_meta()`.
5. The manager builds flat GPU/CPU block transfer lists and event IDs.
6. The v1 worker calls `start_load_kv()` before forward, submitting H2D loads.
7. The model executes. D2H stores are submitted later by `get_finished()` after
   a per-store compute-done CUDA event.
8. `update_connector_output()` consumes load/store completions and updates CPU
   residency, refs, cache visibility, and telemetry.
9. `PrometheusStatLogger` receives connector stats and exposes
   `vllm:simple_cpu_offload_*` metrics.

## File Map

### Configuration

- `vllm/config/cache.py`: CPU swap capacity, NUMA, pinning, block-first, and
  high-risk settings.
- `vllm/config/scheduler.py`: proactive budget and VLT/SLO settings.
- `vllm/config/vllm.py`: translates top-level settings into connector extras.
- `vllm/engine/arg_utils.py`: CLI flag definitions.
- `.env.superinfer`: production-oriented normal profile.
- `.env.superinfer-pressure`: bounded pressure profile.
- `.env.superinfer-service128`: unified one-server reload plus inference profile
  with 128 GiB total CPU KV capacity, or 64 GiB per TP rank.
- `.env.superinfer-serving`: maximum-throughput high-risk serving profile using
  the same 128 GiB CPU KV tier and validated GPU-derived DeepSeek layout.
- `.env.superinfer-cpu128`: opt-in 128 GiB total CPU KV capacity with empty
  allocation; it does not alter the default/reload/pressure profiles.

### Scheduling and Residency

- `vllm/v1/core/sched/scheduler.py`:
  - `_maybe_preempt_for_proactive_swap()` is the guarded rotation entry point.
  - `_is_proactive_swap_candidate()` protects DSpark/prefill/shared-prefix
    requests in normal mode.
  - `_select_proactive_swap_victim()` applies VLT when configured.
- `vllm/v1/simple_kv_offload/manager.py`:
  - `get_num_new_matched_tokens()` finds CPU prefix hits.
  - `update_state_after_alloc()` builds H2D load mappings.
  - `_prepare_lazy_store_specs()` scans evictable GPU blocks.
  - `_prepare_eager_store_specs()` stores confirmed request-owned blocks.
  - `build_connector_meta()` assigns transfer event IDs.
  - `_process_store_event()` makes completed CPU blocks cache-visible.
  - `_reconcile_request_residency()` detects CPU-cache eviction.
  - `_drop_residency_request()` compacts finished request records while retaining
    records required by in-flight transfers.
  - `ResidencyRecord` and `_residency_stats()` expose lifecycle state.

### Transfer Engine

- `vllm/v1/simple_kv_offload/worker.py`:
  - `start_load_kv()` submits H2D before forward.
  - `get_finished()` submits compute-gated D2H stores and polls events.
  - `build_connector_worker_meta()` reports completions and queue depth.
  - CPU KV allocation mode and allocated/pinned capacity are reported after
    registration.
- `vllm/v1/simple_kv_offload/copy_backend.py`:
  - `DmaCopyBackend` owns independent bounded load/store queues and threads.
  - `CopyBackendError` is latched and propagated to the worker.
  - `contiguous_block_runs()` coalesces fallback copies.
- `vllm/v1/simple_kv_offload/cuda_mem_ops.py`:
  - `build_params()` creates batch-DMA pointer/size arrays.
  - `BatchMemcpyParams` carries independent source stride, destination stride,
    and payload bytes.
  - `copy_blocks()` submits `cuMemcpyBatchAsync`/`hipMemcpyBatchAsync`.
- `vllm/v1/simple_kv_offload/native_copy.py`:
  - Optional C++/CUDA bridge for native batch submission.
  - Keeps current heterogeneous descriptors and Python event lifetime.
  - Disabled by default; enable with `--native-copy-backend`.
- `vllm/v1/simple_kv_offload/layout.py`:
  - `choose_layout_mode()` applies safety gates.
  - `build_gpu_cache_views()` creates storage segments and descriptors.
  - DeepSeek-V4 block-first remains disabled in normal mode.

### Topology and NUMA

- `vllm/v1/simple_kv_offload/topology.py`: GH200 GPU/NUMA/NVLink discovery and
  local/remote pool planning.
- `vllm/utils/numa_utils.py`: per-worker and EngineCore NUMA subprocess binding.
- `vllm/v1/executor/multiproc_executor.py`: worker binding hook.
- `vllm/v1/engine/utils.py`: EngineCore binding hook.
- `.env.superinfer`: `NUMA_BIND=1`, `NUMA_BIND_NODES=0,1`.

## Performance Reasoning

### If store/load counters are zero

Inspect in this order:

1. `offload_lazy_target_free_blocks` versus GPU KV usage.
2. Whether requests are active concurrently enough to exceed GPU KV capacity.
3. Whether completed blocks enter the GPU free queue before request cleanup.
4. Whether `SchedulerOutput.finished_req_ids` reaches the manager.
5. Whether block hashes are present and CPU capacity is available.
6. Whether `_prepare_lazy_store_specs()` sees eligible hashed blocks.

Completed requests now bypass the ordinary lazy watermark once, using their
confirmed final blocks as store candidates. If transfer counters remain zero
after a completed-request pressure run, inspect block hashes and the GPU free
queue rather than increasing the benchmark request count.

Lazy finished requests with final block IDs are retained until the next
`finished_req_ids` metadata pass. Cleaning them immediately in
`request_finished()` loses the only store state needed to publish final
confirmed blocks to CPU.

Residency counts are request-scoped diagnostics. Finished records are removed
after cleanup; only records associated with in-flight transfers are retained.
Large historical residency counts therefore indicate a lifecycle leak, not
useful CPU capacity.

Do not tune VLT or transfer queue depth until actual stores occur.

### If stores occur but loads do not

Inspect:

1. CPU cache hit lookup in `get_num_new_matched_tokens()`.
2. `update_state_after_alloc()` group/block back-tracing.
3. `load_event` assignment in `build_connector_meta()`.
4. `start_load_kv()` invocation before forward.
5. `finished_recving` completion polling.

If `offload_residency_cpu_only_blocks` is nonzero but load events remain zero,
stored hashes exist but `get_num_new_matched_tokens()` did not create a
`_reqs_to_load` entry. Inspect request block hashes, computed-token alignment,
and the pending-hit handoff before changing the DMA backend.

Use these counters to locate the exact break:

- `offload_cpu_lookup_requests`
- `offload_cpu_lookup_hits`
- `offload_cpu_lookup_hit_tokens`
- `offload_load_requests_created`
- `offload_load_events_assigned`
- `offload_cpu_cached_keys`

For DeepSeek-V4, do not assume a single MLA cache group. The live CPU
coordinator reports group block sizes `[256, 64, 64, 4, 8]`, with a `256`-token
scheduler block and `4`-token request-hash granularity. A zero aggregate hit
may therefore be caused by one group preventing hybrid hit reconciliation.
The manager logs one bounded diagnostic per scheduler: `cache geometry`,
`first CPU lookup`, and `CPU lookup result`. Compare the per-group hit list with
the cached key-group counts before changing hash normalization or load
admission.

The current DeepSeek-V4 evidence is:

- CPU stores succeed and CPU-only residency is populated.
- Repeated-prefix output equality passes.
- CPU lookup requests occur, but the last observed run had zero lookup hits and
  zero load requests.
- The 20260815 run showed `cached_key_groups={0:85}` during the first completed
  lookup, while the runtime has five DeepSeek groups. This confirms the CPU
  store path was publishing only primary group-0 keys; GPU alias hashes were not
  being mirrored to the CPU cache.
- This is not yet proof of a hash mismatch; the next run must capture the
  per-group diagnostic above.

The first aggressive reload implementation slice now retries hybrid lookup
per-group when aggregate reconciliation returns zero. It admits only the
common prefix rounded to the `256`-token scheduler boundary, so all DeepSeek
groups are restored together. This is intentionally not a partial-group load;
the next GPU run decides whether the remaining issue is key visibility or a
specific group layout.

The next targeted fix mirrors every GPU block-hash alias to the corresponding
CPU block at store completion. Store-completion diagnostics now report the
cached key-group map after each event. The expected post-store map must contain
all relevant group IDs, not only group `0`.

For lazy stores, aliases are now derived when request GPU blocks are recorded
and indexed by GPU block ID before the free-queue scan. This matters because a
free DeepSeek block may expose only its primary group-0 hash through
`node.block_hash`; requiring that field alone skips the other hybrid groups.
Store completion now mirrors this indexed alias set into the CPU cache. The
next run should show group IDs beyond `0` even for the bulk lazy-store events;
only then can CPU lookup admission be evaluated.

`offload_cpu_cached_keys` is a gauge of the CPU prefix-cache key map. Compare it
with `offload_residency_cpu_only_blocks`: CPU-only records without cached keys
indicate an ownership/reconciliation bug, while cached keys with zero hits
indicate a request-hash or hybrid lookup problem.

The post-store diagnostic also reports `direct_group_first_hits` and
`direct_group_hash_indices`. For the current DeepSeek-V4 geometry the expected
indices are `[63, 15, 15, 0, 1]` relative to a `4`-token hash list. Interpret
them as follows:

- All false with cached keys: request hashes and stored keys use different
  boundaries or representations.
- Some true: the corresponding groups are visible; the false groups block the
  common all-group load.
- All true but aggregate hit zero: hybrid coordinator reconciliation is the
  remaining bug.

### If transfer latency is high

Inspect:

1. `offload_load_queue_depth` and `offload_store_queue_depth`.
2. Batch versus fallback path in `DmaCopyBackend._try_build_params()`.
3. Number and shape of `contiguous_block_runs()`.
4. Source/destination strides in `BatchMemcpyParams`.
5. NUMA binding and actual page placement.
6. CUDA/Nsight overlap between copy streams and model kernels.

### If correctness fails

Check event and residency transitions before changing scheduling:

1. Store has a unique compute-done event.
2. CPU block remains pinned until transfer completion.
3. GPU block is not reused before D2H completion.
4. Load is complete before resumed request execution.
5. Dirty speculative tails are not published as stable CPU KV.
6. Reset/cancellation releases abandoned transfer refs.

## Optimization Order

1. Prove real CPU movement with `.env.superinfer-pressure`.
2. Validate NUMA binding and page placement.
3. Measure H2D/D2H overlap.
4. Make residency state authoritative for all groups and tails.
5. Backport reliable partial-tail and single-copy MLA mechanisms.
6. Make `OffloadLayoutDescriptor` drive DeepSeek heterogeneous DMA.
7. Add DSpark-safe rotation at accepted-token boundaries.
8. Tune budgets only after the implementation path is measured.

## Reload Validation

Use `scripts/run_reload_validation.sh` after starting a SuperInfer profile. It
runs identical deterministic prefixes across three phases:

1. `store`: populate CPU KV through completed-request stores.
2. `evict`: send distinct prefixes concurrently so their live batch exceeds GPU
   KV capacity and removes the target prefixes from GPU.
3. `reload`: reuse the original prefixes and require CPU H2D loads.

The harness captures metrics before/after and writes `reload.json` and
`validation.json`. The validation gate compares metric deltas, not absolute
counters, because the server may have served earlier traffic. Accept only when
store events/bytes and load events/bytes all increase and `output_mismatches`
is zero.

Run the complete lifecycle inside `richard-base-dev-sysnice` when the GPU is
free:

```bash
docker exec -w /workspace/re-SuperInfer/vllm-superinfer-v4 \
  richard-base-dev-sysnice bash -lc \
  './scripts/run_reload_validation.sh --restart .env.superinfer-reload'
```

`--restart` stops any existing profile process, launches a clean server, skips
the unrelated production warmup through `SKIP_LAUNCH_WARMUP=1`, runs
store/evict/reload, and stops the server on exit. The launch readiness probe
bypasses the container HTTP proxy. If the server is already running, omit
`--restart`; the harness then runs only the validation client and captures the
existing process.

For a reliable reload gate, use `.env.superinfer-reload`. It gives the CPU tier
enough capacity to retain target prefixes while the eviction phase removes them
from GPU KV:

```bash
scripts/run_reload_validation.sh --restart .env.superinfer-reload
```

Do not use a reload artifact with only exact output equality as a restore proof.
Exact equality can pass when the repeated request recomputes on GPU;
`validation.json` must also report positive H2D load deltas.

## What Is Not Implemented

- Native C++/CUDA/ZMQ DuplexKV swap path.
- Full paper RotaSched/LVF queue semantics.
- Complete native retry/control-plane behavior.
- Fully validated DeepSeek block-first layout.
- GPU proof of CPU KV movement and copy/compute overlap.

These are separate parity or validation goals, not prerequisites for every
performance optimization.
