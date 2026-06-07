# SuperInfer Delta Map

SuperInfer is a vLLM `v0.6.6.post1` fork. Its delta from upstream `v0.6.6.post1` is about 60 files, 4586 insertions, and 785 deletions.

| Concept | File(s) | Function/class | Behavior | Dependencies | Notes |
|---|---|---|---|---|---|
| CLI/config flags | `vllm/engine/arg_utils.py`, `vllm/config.py` | `EngineArgs`, `CacheConfig`, `SchedulerConfig` | Adds `--proactive-swap-budget`, `--swapper-block-first`, `--pin-memory-fix`, `--prefix-cache-fix`. | Old monolithic config classes. | Must be re-added in modern split config as no-op first. |
| RotaSched/LVF policy | `vllm/v1/core/scheduler.py` | `Scheduler`, `schedule_early`, `schedule_single_*` | Proactively swaps running requests when free GPU blocks fall below budget and CPU swap usage allows it. | Old V1 scheduler list/deque state, `RequestStatus.SWAPPED`. | Current code is heuristic and not a complete VLT implementation despite LVF naming. |
| Request timing state | `vllm/v1/request.py` | `Request` | Adds `in_running_since`, `in_waiting_since`, lookahead state, cached token counter. | Old scheduler mutates these fields directly. | Modern request has different state and spec-token accounting. Port carefully. |
| CPU/GPU block pairing | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager` | Creates GPU block pool plus CPU block pool; each GPU block has a mapped CPU block. | Modified `KVCacheBlock` metadata. | Modern KV groups make one-to-one mapping more complex, especially DeepSeek-V4. |
| Block state tracking | `vllm/v1/core/kv_cache_utils.py` | `KVCacheBlockStatus`, `KVCacheBlock` | Adds clean/dirty/swapped/half/full/empty states and CPU-block refcount propagation. | Old block pool implementation. | Modern block pool/coordinator must be extended rather than replaced. |
| Pending swap lists | `vllm/v1/core/kv_cache_manager.py` | `pending_blocks_to_swap_in/out` | Scheduler accumulates block-id pairs for transfer. | Executor swap IPC. | Modern scheduler outputs need a typed extension or existing offload interfaces. |
| DuplexKV Python wrapper | `vllm/v1/swapper/swapper.py`, `_native.py` | `Swapper` | Allocates pinned CPU KV mirror and starts native C++ swap thread. | Torch C++ extension, ZMQ, msgpack, CUDA. | Modern vLLM already has offload paths; use them as landing zones first. |
| DuplexKV native copy | `vllm/v1/swapper/native/*` | `swap_block_first`, `swap`, `swap_naive` | Uses CUDA streams, block-first CPU layout, and `cudaMemcpyBatchAsync` for batched H2D/D2H. | CUDA runtime, GH200. | Correct only after modern cache layout and DeepSeek-V4 cache state are audited. |
| Engine pipeline overlap | `vllm/v1/engine/core.py` | `EngineCore.step` | Overlaps scheduling/swap with model execution from previous iteration. | Executor process wrappers, ZMQ IPC. | Modern engine core is much larger and already supports async scheduling/parallel features. High risk to copy. |
| Executor process swap control | `vllm/v1/executor/uniproc_executor.py`, `multiproc_executor.py` | `UniprocExecutorProcess`, `MultiprocExecutor` | Co-locates swap thread with model worker and uses IPC for model execution and swap completion. | Old executor topology. | Modern executor has changed; defer until a safe worker/offload extension point is selected. |
| Prefix-cache fix | `vllm/v1/core/kv_cache_manager.py` | `append_slots`, `allocate_slots` | Delays/adjusts prefix cache full-block caching and marks full blocks clean. | Old prefix cache internals. | Modern prefix cache already changed; no direct port. Audit before behavior. |

## Why Not Copy Directly

- Modern scheduler path moved to `vllm/v1/core/sched/`.
- Modern KV cache supports multiple KV cache groups, hybrid specs, Mamba, DeepSeek-V4 MLA, and offload connectors.
- DeepSeek-V4 cache layout is not a plain dense K/V cache.
- Modern vLLM already has native KV offload infrastructure that did not exist in the old SuperInfer base.
