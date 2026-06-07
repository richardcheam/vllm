# DeepSeek-V4 Cache State Audit

## Relevant Files

| Area | Files |
|---|---|
| Model | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`, `vllm/model_executor/models/deepseek_mtp.py` |
| Registry | `vllm/model_executor/models/registry.py` |
| KV specs/layout | `vllm/v1/kv_cache_interface.py`, `vllm/v1/core/kv_cache_utils.py` |
| KV manager | `vllm/v1/core/kv_cache_manager.py`, `kv_cache_coordinator.py`, `single_type_kv_cache_manager.py`, `block_pool.py` |
| DeepSeek-V4 attention ops | `vllm/v1/attention/ops/deepseek_v4_ops/*` |
| MLA backends | `vllm/v1/attention/backends/mla/*` |
| MTP/spec decode | `vllm/v1/spec_decode/*`, `vllm/v1/worker/gpu/spec_decode/*`, `vllm/v1/worker/gpu/model_runner.py` |
| Prefix cache | `vllm/v1/core/kv_cache_utils.py`, `kv_cache_manager.py`, `block_pool.py` |

## Findings

1. DeepSeek-V4 does not use a simple dense K/V cache model. `MLAAttentionSpec` and `SlidingWindowMLASpec` have DeepSeek-V4-only sizing where `fp8_ds_mla` stores `448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token`.
2. DeepSeek-V4 can have multiple KV cache groups with different block sizes/compression ratios. Modern `KVCacheBlocks` is a tuple of per-group block sequences, not a single flat block table.
3. DeepSeek-V4 has indexer-related state. `DeepseekV4IndexerBackend` supports `use_fp4_indexer_cache`; indexer cache may need to move with main MLA cache or be marked unswappable initially.
4. Prefix cache owns full blocks through block hashes and reference counts. Prefix-owned or shared blocks must not be blindly evicted or copied without refcount-aware state tracking. The guarded proactive swap path now treats allocated non-null blocks with `ref_cnt > 1` as pinned and excludes their requests from proactive victim selection.
5. MTP speculative decoding can add lookahead/speculative state and target hidden state plumbing. Initial swap correctness should disable MTP or avoid swapping requests with active speculative state.
6. CUDA graph capture may assume stable cache tensor addresses. Swapping by copying contents into existing GPU cache blocks is safer than reallocating cache tensors, but any block table mutation must happen outside captured graph assumptions.
7. Existing modern KV offload infrastructure should be studied before adding a separate SuperInfer swapper.

## Answers

1. To resume a DeepSeek-V4 request after swap-out, at minimum the block table, all per-group MLA cache pages, request token/progress state, prefix-cache ownership/refcounts, and any indexer/cache metadata must be consistent.
2. K/V tensors alone are not enough for a general DeepSeek-V4 request because compressed MLA layout, fp8 scales, sparse/indexer metadata, and possibly fp4 indexer cache may be involved.
3. Prefix cache can own shared blocks. Initially, prefix-shared blocks are treated as pinned/unswappable: scheduler proactive candidates are rejected when any allocated non-null block has `ref_cnt > 1`.
4. MTP creates additional speculative execution state. Initially treat active MTP/speculative requests as unswappable or require MTP disabled for correctness tests.
5. CUDA graphs should not see changing cache tensor allocations. Content copies into stable cache buffers are the only plausible initial approach.
6. Safest first swap scope is a single non-prefix-shared, non-MTP request where all relevant KV cache groups are copied as opaque pages and restored into stable GPU block IDs before execution resumes.
7. Initially never swap prefix-shared blocks, MTP-active requests, indexer-cache-dependent blocks without paired metadata movement, or requests using un-audited hybrid/Mamba groups.
