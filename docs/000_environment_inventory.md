# Environment Inventory

## Repositories

| Repo | Path | Role | Revision |
|---|---|---|---|
| SuperInfer | `/workspace/re-SuperInfer/SuperInfer` | Official SuperInfer fork used for archaeology | `superinfer`, `790d1fad feat(superinfer): KV-cache swap path, LVF scheduler, and AE harness` |
| SuperInfer-Old | `/workspace/re-SuperInfer/SuperInfer-Old` | Failed prior DeepSeek-V4 attempt; inspect only for pitfalls | `superinfer`, `790d1fad-dirty` |
| vLLM modern | `/workspace/re-SuperInfer/vllm-modern` | Implementation target | detached `v0.20.1`, `132765e35 Revert "[DSv4] Use cvt PTX for FP32->FP4 conversion (#41015)"` |

## Container

- Container: `richard-base-dev`
- Repo mount inside container: `/workspace/re-SuperInfer`
- Python: `Python 3.12.3`
- uv: `uv 0.9.8`
- Project venv: `/workspace/re-SuperInfer/vllm-modern/.venv`

## Base Versions

- SuperInfer README states it is built as a fork of `vLLM v0.6.6.post1`.
- Upstream tag `v0.6.6.post1` exists after adding/fetching upstream in the SuperInfer checkout.
- Modern target is `vLLM v0.20.1`, which contains DeepSeek-V4 support in `vllm/model_executor/models/deepseek_v4.py`, DeepSeek-V4 MTP support, and DeepSeek-V4-specific V1 attention/cache code.
- SuperInfer paper source available locally at `/workspace/re-SuperInfer/2601.20309v2.pdf` and used as semantic reference.

## Important Local State

- `SuperInfer-Old` has uncommitted local files:
  - modified `vllm/model_executor/models/registry.py`
  - untracked `vllm/model_executor/models/deepseek_v4.py`
  - untracked `2601.20309.pdf`
- Treat `SuperInfer-Old` as a failed-attempt artifact, not an implementation source.

## Modern vLLM Architecture Notes

- This is the vLLM V1 engine path for the target serving use case.
- Scheduler moved from old `vllm/v1/core/scheduler.py` to `vllm/v1/core/sched/scheduler.py` plus related modules under `vllm/v1/core/sched/`.
- KV cache management is split across `vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/kv_cache_coordinator.py`, `vllm/v1/core/block_pool.py`, `vllm/v1/core/single_type_kv_cache_manager.py`, and `vllm/v1/kv_cache_interface.py`.
- Modern vLLM already includes KV offload infrastructure under `vllm/v1/kv_offload/` and `vllm/v1/simple_kv_offload/`.
- Modern DeepSeek-V4 support includes model files, MTP model files, DeepSeek-V4 attention ops, MLA sparse attention, fp8 DeepSeek MLA cache layout, and optional fp4 indexer cache.

## CUDA/PyTorch Assumptions

- The official SuperInfer artifact targeted GH200 with CUDA 12.8 and PyTorch 2.5.1.
- The modern vLLM checkout should use its own pinned dependency flow. Per repository instructions, Python commands must go through `uv` and `.venv/bin/python`.
