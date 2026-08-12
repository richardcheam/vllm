# SuperInfer GPU Validation Checklist

This checklist tracks GPU runtime validation. The current branch has CPU/unit
coverage for the guarded scheduler and offload paths; guarded CPU/GPU
correctness validation has completed on GH200.

## Preconditions

- Use the same container/workspace used for CPU validation where possible.
- Confirm GPU visibility with `nvidia-smi` inside the runtime container.
- Keep SuperInfer knobs disabled for the baseline pass.
- Use deterministic decoding first: temperature `0`, fixed seed, small prompt set.

## Phase 1: Baseline Modern Behavior

Goal: prove this branch does not change outputs when SuperInfer knobs are off.

- Run a small model such as `facebook/opt-125m` with default SuperInfer knobs.
- Compare generated token IDs/text against a vanilla modern vLLM checkout or a
  known-good baseline run from the same branch with all SuperInfer knobs unset.
- Expected result: no output divergence attributable to SuperInfer config.

## Phase 2: Conservative Swap-On Smoke Test

Goal: prove guarded swap/proactive paths run without corrupting outputs.

- Enable `swap_cpu_memory_gb` with a small CPU budget appropriate for the host.
- Keep `proactive_swap_budget` small.
- Use a small prompt set that creates at least two concurrent requests.
- Expected result: valid completions, no scheduler assertion failures, no KV load
  failures, and no request stuck in waiting/preempted state.

## Phase 3: VLT/SLO Smoke Test

Goal: prove VLT-enabled guarded victim selection remains runtime-safe.

- Enable conservative VLT knobs, for example `vlt_beta_bandwidth > 0` and one
  SLO knob.
- Repeat the small-model prompt set.
- Expected result: valid completions and telemetry showing offload/proactive
  activity when pressure exists.

## Phase 4: DeepSeek-V4-Flash Readiness

Goal: validate model-family safety before larger DeepSeek-V4 runs.

- Start with the smallest available DeepSeek-V4-Flash-compatible setup.
- Keep proactive budget conservative.
- Monitor KV connector metadata, request status transitions, and output validity.
- Expected result: no MLA/cache-layout mismatch, no rotary-state inconsistency,
  and no model-output corruption.

## Commands To Record

For every GPU run, record:

- exact command and config flags,
- model name and revision,
- prompt set,
- generated outputs or token IDs,
- relevant logs around KV transfer/proactive preemption,
- pass/fail conclusion and any follow-up patches.

## Known CPU-Only Validation Already Completed

- VLT helper and ranking tests.
- Request rotary state transition tests.
- SuperInfer config/default-off tests.
- Simple CPU offload scheduler tests.
- DmaCopyBackend CPU routing test for independent load/store submission workers.
- Simple CPU offload transfer flag propagation tests for `pin_memory_fix` and
  `swapper_block_first`.
- Guarded proactive scheduler tests, including fallback/VLT victim selection,
  small decode-backlog candidates, unsafe-candidate guards, and resume sequencing.

## Validation Log

### 2026-06-07 GH200 Smoke Pass

- Environment:
  - Container: `richard-base-dev`.
  - Repo: `/workspace/re-SuperInfer/vllm-modern`.
  - GPU: 2x NVIDIA GH200 144G HBM3e visible via `nvidia-smi`.
  - Python stack repaired for GPU validation: `torch==2.11.0+cu130`, CUDA `13.0`, `vllm._C` import OK.
  - `vllm` installed editable from source because `wheels.vllm.ai` was blocked by proxy policy; PyTorch wheels were installed through the provided proxy.
- Phase 1 baseline, SuperInfer knobs off:
  - Model: `facebook/opt-125m`.
  - Config: `dtype=float16`, `seed=123`, `max_model_len=128`, `gpu_memory_utilization=0.2`, `enforce_eager=True`, `temperature=0.0`, `max_tokens=8`.
  - Result: passed; two prompts completed with deterministic token IDs and no runtime errors.
- Phase 2 conservative swap-on smoke:
  - Model: `facebook/opt-125m`.
  - Config: `swap_cpu_memory_gb=1.0`, `num_gpu_blocks_override=8`, `max_num_seqs=2`, `max_model_len=128`, `enforce_eager=True`.
  - Result: passed; `SimpleCPUOffloadConnector` initialized in lazy mode on worker and scheduler, 1 GiB CPU block pool allocated, two pressured requests completed without KV load failures or stuck states.
- Phase 3 guarded proactive/VLT smoke:
  - Model: `facebook/opt-125m`.
  - Config: `swap_cpu_memory_gb=1.0`, `proactive_swap_budget=4`, `vlt_beta_bandwidth=1.0`, `num_gpu_blocks_override=8`, `max_num_seqs=2`, `max_model_len=128`, `enforce_eager=True`.
  - Result: passed; lazy target free blocks followed proactive budget (`4`), debug single-request swap gate enabled, three pressured requests completed.
- Phase 4 DeepSeek-V4 readiness and generated-output smoke:
  - Local model snapshot: `/workspace/re-SuperInfer/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136`.
  - CUDA readiness tests passed: `tests/models/test_deepseek_v4_mega_moe.py`, `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`, and `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` -> 41 passed.
  - Baseline DeepSeek-V4 generated-output smoke passed with `tensor_parallel_size=2`, `dtype=bfloat16`, `kv_cache_dtype=fp8`, `max_model_len=512`, `max_tokens=4`, `gpu_memory_utilization=0.92`, `enforce_eager=True`, and SuperInfer knobs off.
  - Conservative DeepSeek-V4 swap-on smoke passed with `swap_cpu_memory_gb=4.0`, `max_num_seqs=2`, and `prompt_count=2`; `SimpleCPUOffloadConnector` initialized on scheduler and workers with 2 GiB per TP rank.
  - Guarded DeepSeek-V4 proactive/VLT smoke passed with `swap_cpu_memory_gb=4.0`, `proactive_swap_budget=4`, `vlt_beta_bandwidth=1.0`, `max_num_seqs=1`, `prompt_count=2`, and lowered `gpu_memory_utilization=0.84` to leave headroom for an unrelated GPU process.
  - Test harness fixes applied: provide `compilation_config.static_forward_context` in the MegaMoE unit fixture and avoid gated `meta-llama/Meta-Llama-3-8B` in the DeepSeek-V4 indexer test.

### 2026-06-08 Batched Proactive Follow-Up

- Scope: validate the post-MVP RotaSched expansion where proactive preemption can free multiple guarded candidates and lazy offload no longer enables `debug_single_request_swap` automatically.
- Small-model smoke:
  - Model: `facebook/opt-125m`.
  - Config: `swap_cpu_memory_gb=1.0`, `proactive_swap_budget=4`, `vlt_beta_bandwidth=1.0`, `num_gpu_blocks_override=8`, `max_num_seqs=2`, `max_model_len=128`, `gpu_memory_utilization=0.2`, `enforce_eager=True`.
  - Result: passed; `SimpleCPUOffloadConnector` initialized on worker and scheduler with `debug_single_request_swap=False`, three prompts completed.
- DeepSeek-V4-Flash smoke:
  - Model snapshot: `/workspace/re-SuperInfer/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136`.
  - Config: `tensor_parallel_size=2`, `dtype=bfloat16`, `kv_cache_dtype=fp8`, `max_model_len=512`, `max_tokens=4`, `gpu_memory_utilization=0.84`, `swap_cpu_memory_gb=4.0`, `proactive_swap_budget=4`, `vlt_beta_bandwidth=1.0`, `max_num_seqs=1`, `prompt_count=2`.
  - Result: passed; scheduler and both TP workers initialized `SimpleCPUOffloadConnector` with `debug_single_request_swap=False`, CPU block pools allocated, and both prompts completed.
- Pressure-target follow-up:
  - Model: `facebook/opt-125m`.
  - Config: `swap_cpu_memory_gb=1.0`, `proactive_swap_budget=1`, `vlt_beta_bandwidth=1.0`, `num_gpu_blocks_override=8`, `max_num_seqs=3`, `max_model_len=128`, `gpu_memory_utilization=0.2`, `enforce_eager=True`.
  - Result: passed; four prompts completed while exercising a larger waiting prompt than the static proactive budget.

### 2026-06-08 Transfer Flag Visibility Smoke

- Scope: validate that `pin_memory_fix` and `swapper_block_first` now reach runtime connector/worker logging without changing CPU KV layout.
- Model: `facebook/opt-125m` on GPU 0.
- Config: `swap_cpu_memory_gb=1.0`, `pin_memory_fix=True`, `swapper_block_first=True`, `num_gpu_blocks_override=8`, `max_num_seqs=2`, `max_model_len=128`, `gpu_memory_utilization=0.2`, `enforce_eager=True`.
- Result: passed; two prompts completed. Logs showed scheduler and worker `SimpleCPUOffloadConnector` initialized with both flags set, worker `pin_strategy=cudaHostRegister`, worker `cpu_layout=gpu_derived`, and the expected `swapper_block_first` warning.

### 2026-06-08 Dual Submission Backend Smoke

- Scope: validate worker initialization and request completion after splitting `DmaCopyBackend` into independent load/store submission queues and threads.
- Model: `facebook/opt-125m` on GPU 0.
- Config: `swap_cpu_memory_gb=1.0`, `pin_memory_fix=True`, `swapper_block_first=True`, `num_gpu_blocks_override=8`, `max_num_seqs=2`, `max_model_len=128`, `gpu_memory_utilization=0.2`, `enforce_eager=True`.
- Result: passed; two prompts completed. Logs showed `DmaCopyBackend: dual-thread batched async copy backend enabled` while preserving `cpu_layout=gpu_derived` and the expected `swapper_block_first` warning.

### 2026-06-08 Post Stage-1 Refactor GPU Re-Validation

- Scope: validate the Stage-1 behavior-preserving layout/copy refactor on real GPU paths before any block-first enablement.
- Small-model guarded smoke (`facebook/opt-125m`, GPU 0):
  - Config: `swap_cpu_memory_gb=1.0`, `pin_memory_fix=True`, `swapper_block_first=True`, `proactive_swap_budget=4`, `vlt_beta_bandwidth=1.0`, `max_model_len=128`, `num_gpu_blocks_override=16`, `max_num_seqs=2`, `gpu_memory_utilization=0.2`, `enforce_eager=True`.
  - Result: passed; prompts completed. Logs confirmed `cpu_layout=gpu_derived` and `DmaCopyBackend: dual-thread batched async copy backend enabled`.
- DeepSeek-V4-Flash guarded smoke (TP=2, GPUs 0/1):
  - Model snapshot: `/workspace/re-SuperInfer/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136`.
  - Config: `swap_cpu_memory_gb=4.0`, `proactive_swap_budget=4`, `vlt_beta_bandwidth=1.0`, `max_model_len=512`, `max_tokens=4`, `max_num_seqs=1`, `prompt_count=2`, `tensor_parallel_size=2`, `gpu_memory_utilization=0.84`, `kv_cache_dtype=fp8`, `enforce_eager=True`.
- Result: passed; both prompts completed. Worker logs showed `DmaCopyBackend: dual-thread batched async copy backend enabled` on TP workers, with no new transfer/layout failures.

### 2026-06-08 Stage-2 Gating GPU Re-Validation

- Scope: validate strict `swapper_block_first` eligibility gating and fallback logging on real GPU paths.
- Small-model guarded smoke (`facebook/opt-125m`, GPU 0):
  - Config: `swap_cpu_memory_gb=1.0`, `pin_memory_fix=True`, `swapper_block_first=True`, `proactive_swap_budget=4`, `vlt_beta_bandwidth=1.0`, `max_model_len=128`, `num_gpu_blocks_override=16`, `max_num_seqs=2`, `gpu_memory_utilization=0.2`, `enforce_eager=True`.
  - Result: passed; logs include explicit fallback reason `block_first eligible but not yet implemented`, active `cpu_layout=gpu_derived`, and `DmaCopyBackend: dual-thread batched async copy backend enabled`.
- DeepSeek-V4-Flash guarded smoke (TP=2, GPUs 0/1):
  - Config unchanged from prior guarded run (`swap_cpu_memory_gb=4.0`, `proactive_swap_budget=4`, `vlt_beta_bandwidth=1.0`, `tensor_parallel_size=2`, `kv_cache_dtype=fp8`, `max_model_len=512`, `max_tokens=4`, `max_num_seqs=1`, `prompt_count=2`, `gpu_memory_utilization=0.84`, `enforce_eager=True`).
  - Result: passed; no new transfer regressions after gating changes.

### 2026-06-08 Allowlist-Narrowing GPU Validation Follow-Up

- Scope: validate the stricter Stage-2 architecture allowlist (currently `OPTForCausalLM` only) with active block-first on eligible paths and guarded fallback on DeepSeek-V4.
- Small-model stress smoke (`facebook/opt-125m`, GPU 0):
  - Config: `swap_cpu_memory_gb=1.0`, `pin_memory_fix=True`, `swapper_block_first=True`, `proactive_swap_budget=4`, `vlt_beta_bandwidth=1.0`, `max_model_len=128`, `num_gpu_blocks_override=16`, `max_num_seqs=3`, `gpu_memory_utilization=0.2`, `enforce_eager=True`.
  - Run shape: 6 prompts, two generate passes in one engine instance.
  - Result: passed; logs show active `cpu_layout=block_first` and `DmaCopyBackend: dual-thread batched async copy backend enabled`.
  - Note: `reset_prefix_cache()` returned `False` because in-flight blocks were still referenced; this did not affect run completion or output validity.
- DeepSeek-V4-Flash guarded smoke (TP=2, GPUs 0/1):
  - Config: `swap_cpu_memory_gb=4.0`, `pin_memory_fix=True`, `swapper_block_first=True`, `proactive_swap_budget=4`, `vlt_beta_bandwidth=1.0`, `tensor_parallel_size=2`, `kv_cache_dtype=fp8`, `max_model_len=512`, `max_tokens=4`, `max_num_seqs=1`, `gpu_memory_utilization=0.84`, `enforce_eager=True`.
  - Result: passed; both TP workers logged `swapper_block_first requested but using gpu_derived layout (DeepSeek-V4 forced fallback)` and `cpu_layout=gpu_derived`, with successful completions.

### Remaining GPU Work

- Keep broader LVF/RotaSched, native DuplexKV/block-first transfer, refcount-aware shared-prefix offload, and performance benchmarking deferred until explicitly started.
- If the shared GH200 is busy, lower `gpu_memory_utilization` before rerunning DeepSeek-V4-Flash smoke; a prior `0.92` proactive run hit Triton CUDA OOM while another process occupied GPU 0 memory, and the same validation passed at `0.84`.

### 2026-06-09 DeepSeek Synced Full-Block Pre-Store Smoke

- Scope: validate the lazy-offload change that opportunistically stores confirmed full blocks while requests are still running, matching the paper's synced-block direction without enabling unsafe DeepSeek block-first layout.
- Model snapshot: `/workspace/re-SuperInfer/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136`.
- Config: `tensor_parallel_size=2`, `dtype=bfloat16`, `kv_cache_dtype=fp8`, `max_model_len=512`, `max_tokens=4`, `max_num_seqs=1`, `gpu_memory_utilization=0.84`, `swap_cpu_memory_gb=4.0`, `proactive_swap_budget=4`, `vlt_beta_bandwidth=1.0`, `pin_memory_fix=True`, `swapper_block_first=True`, `enforce_eager=True`.
- Result: passed; two prompts completed. Both TP workers kept the DeepSeek safety fallback with `swapper_block_first requested but using gpu_derived layout (DeepSeek-V4 forced fallback)` and `cpu_layout=gpu_derived`.

### 2026-06-09 Low-Utilization DeepSeek SuperInfer Smoke

- Scope: validate the current DeepSeek-V4 SuperInfer path after synced-block victim-cost and rotary-state changes while minimizing shared-GPU memory reservation.
- Model snapshot: `/workspace/re-SuperInfer/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136`.
- Config: `tensor_parallel_size=2`, `dtype=bfloat16`, `kv_cache_dtype=fp8`, `max_model_len=512`, `max_tokens=4`, `max_num_seqs=1`, `gpu_memory_utilization=0.70`, `swap_cpu_memory_gb=4.0`, `proactive_swap_budget=4`, `vlt_beta_bandwidth=1.0`, `pin_memory_fix=True`, `swapper_block_first=True`, `enforce_eager=True`.
- Result: passed; two prompts completed. Workers kept `DeepSeek-V4 forced fallback` with `cpu_layout=gpu_derived`. vLLM reported `24.63 GiB` available KV cache memory and `67,742` GPU KV-cache tokens, so the P0 correctness smoke does not require the previous `0.84` utilization.

### 2026-06-09 P0 Baseline vs SuperInfer-On DeepSeek Output Comparison

- Scope: P0 blocker validation for deterministic DeepSeek output parity with SuperInfer knobs enabled.
- Prompt: `DeepSeek deterministic comparison prompt for SuperInfer.`
- Shared config: `tensor_parallel_size=2`, `dtype=bfloat16`, `kv_cache_dtype=fp8`, `max_model_len=512`, `max_tokens=4`, `max_num_seqs=1`, `gpu_memory_utilization=0.70`, `enforce_eager=True`, greedy decoding.
- Baseline result: token IDs `[455, 12275, 344, 6558]`, text `' The prompt is designed'`.
- SuperInfer-on config additions: `swap_cpu_memory_gb=4.0`, `proactive_swap_budget=4`, `vlt_beta_bandwidth=1.0`, `pin_memory_fix=True`, `swapper_block_first=True`.
- SuperInfer-on result: token IDs `[455, 12275, 344, 6558]`, text `' The prompt is designed'`.
- Conclusion: passed; P0 deterministic output parity holds for this low-utilization DeepSeek smoke.
