# Modern vLLM Landing Zones

| SuperInfer old location | Modern landing zone | Porting difficulty | Implementation plan |
|---|---|---|---|
| `vllm/engine/arg_utils.py` flags | `vllm/engine/arg_utils.py`, `vllm/config/cache.py`, `vllm/config/scheduler.py` | Low | Add parsed no-op fields first and test CLI/config propagation. |
| `vllm/v1/core/scheduler.py` | `vllm/v1/core/sched/scheduler.py`, `request_queue.py`, `output.py`, `interface.py` | High | Start with telemetry and pure VLT helper; do not alter admission/preemption until DeepSeek-V4 audit is complete. |
| `vllm/v1/core/kv_cache_manager.py` | `vllm/v1/core/kv_cache_manager.py`, `kv_cache_coordinator.py`, `single_type_kv_cache_manager.py`, `block_pool.py`, `kv_cache_interface.py` | High | Add metadata-only helpers first. Avoid moving blocks until all KV cache groups and DeepSeek-V4 layouts are understood. |
| `vllm/v1/core/kv_cache_utils.py` | `vllm/v1/core/kv_cache_utils.py`, `block_pool.py` | Medium | If needed, add block state as separate SuperInfer metadata rather than mutating core block semantics initially. |
| `vllm/v1/swapper/*` | `vllm/v1/kv_offload/`, `vllm/v1/simple_kv_offload/`, worker offload hooks | High | Prefer extending existing offload interfaces over adding a parallel swapper subsystem. |
| `vllm/v1/engine/core.py` pipeline | `vllm/v1/engine/core.py`, async scheduler, executor interfaces | High | Defer. Modern engine has async scheduling; first measure no-op overhead and use existing pipeline points. |
| `vllm/v1/executor/*` swap thread | `vllm/v1/executor/*`, `vllm/v1/worker/gpu/*`, `kv_offload/worker/*` | High | Keep disabled until single-request swap correctness is proven. |
| Old attention layout assumptions | `vllm/v1/attention/backends/mla/*`, `vllm/v1/attention/ops/deepseek_v4_ops/*` | Very high | Do not change layout initially. Block-first layout is a later DuplexKV optimization. |

## Modern DeepSeek-V4 Search Results

- Model registry includes `DeepseekV4ForCausalLM` and `DeepSeekV4MTPModel`.
- DeepSeek-V4 cache sizing lives in `vllm/v1/kv_cache_interface.py` and `vllm/v1/core/kv_cache_utils.py`.
- DeepSeek-V4 attention ops live in `vllm/v1/attention/ops/deepseek_v4_ops/`.
- MLA sparse attention and indexer logic lives under `vllm/v1/attention/backends/mla/`.
- MTP plumbing exists in `vllm/v1/spec_decode/` and `vllm/v1/worker/gpu/spec_decode/`.

## Recommended First Landing Zone

The first code change should only add no-op configuration flags. That gives users the requested CLI surface while guaranteeing vanilla DeepSeek-V4 behavior is unchanged.
