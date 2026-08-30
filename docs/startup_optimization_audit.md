# Startup Optimization Audit

This document records the startup investigation for the SuperInfer vLLM
checkout. The current phase is measurement and observability only. No startup
optimization is enabled by this change.

## Execution Environment

All runtime, cache, and future GPU validation work for this checkout must run
inside the Docker container `richard-base-dev-sysnice`. The repository path and
interpreter inside that container are:

```text
repository=/workspace/re-SuperInfer/vllm-superinfer-v4
python=/workspace/re-SuperInfer/vllm-superinfer-v4/.venv/bin/python
HOME=/home/richard
```

Host-side paths such as `/data1/home/az04297/re-SuperInfer/vllm-superinfer-v4`
are not the runtime paths used by the service. Run source-only diagnostics from
the host as follows:

```bash
docker exec -w /workspace/re-SuperInfer/vllm-superinfer-v4 \
  richard-base-dev-sysnice bash -lc \
  './.venv/bin/python scripts/report_startup_diagnostics.py --json'
```

The target checkout uses the container's compatible Torch installation through
the repository-local `.venv`. Do not substitute a host Python interpreter or
install a replacement Torch build for startup measurements.

## Scope And Invariants

The goal is to reduce application startup time without changing steady-state
behavior. The following must remain unchanged unless a separately measured
experiment proves equivalence:

- KV-cache capacity, context length, GPU memory utilization, and concurrency.
- CUDA graph mode, graph coverage, torch.compile settings, and optimized kernels.
- Tensor, pipeline, expert, and context parallelism.
- Prefix caching, speculative decoding, NUMA placement, and model quality.
- Existing production profiles and SuperInfer transfer/offload behavior.

Do not run, stop, restart, or benchmark GPU workloads while the GPUs are in
use by another service.

## Current Findings

- Inside `richard-base-dev-sysnice`, the active vLLM cache resolves to
  `/home/richard/.cache/vllm` because `VLLM_CACHE_ROOT` is not explicitly
  configured.
- The cache contains a large `torch_compile_cache` tree. The diagnostic report
  currently identifies its filesystem as the container's `overlay`, not as a
  separately verified persistent mount. It may therefore disappear when the
  container is recreated.
- The Hugging Face cache is separately persisted through the Docker volume
  `az04297_richard_huggingface_cache`.
- Cache ownership and permissions currently allow the `richard` runtime user
  to read and write the cache.
- `VLLM_USE_AOT_COMPILE` defaults to enabled for supported Torch versions when
  compile caching is enabled. The active DeepSeek/DSpark path uses breakable
  CUDA graphs, which currently forces `CompilationMode.NONE`; AOT reuse has
  not been demonstrated for that path.
- `VLLM_ENABLE_STARTUP_PLAN` can persist a fingerprinted KV-memory result and
  reuse it only when the fingerprint and free-memory safety gate match. It
  does not skip model loading or CUDA graph capture, so it remains opt-in.
- Explicit KV sizing, alternative loader strategies, prefetch behavior, MoE
  cold-start settings, and NUMA loading changes have not been enabled.

## Existing Timing Evidence

Prior startup logs show the following approximate ranges, but they are not a
controlled baseline for the current task:

| Phase | Observed range |
| --- | ---: |
| Safetensors shard reading | 4-60 s |
| CUDA graph capture | 50-65 s |
| Post-load KV/profile/warmup | 93-105 s |
| Some model-runner initialization spans | 1,280-1,370 s |

The long model-runner spans require phase-level evidence before attributing
them to disk I/O, compilation, distributed initialization, memory profiling,
or graph capture.

## Observability

Startup methods now have opt-in structured markers. Enable them for a
diagnostic run with:

```bash
docker exec -w /workspace/re-SuperInfer/vllm-superinfer-v4 \
  -e VLLM_STARTUP_PHASE_LOGGING=1 \
  richard-base-dev-sysnice bash -lc \
  './scripts/launch_superinfer.sh --env-file .env.superinfer'
```

The markers use the form `STARTUP_PHASE phase=... event=start|end` and include
the process ID, process-relative elapsed time, and phase duration on end
events. The default is disabled, so normal startup log volume and behavior are
unchanged.

Instrumented boundaries include:

- Engine KV-cache initialization.
- Worker process initialization, device setup, and model loading.
- Worker memory profiling, KV-cache initialization, and warmup/graph capture.
- Model-runner loading, CUDA graph memory profiling, and CUDA graph capture.

Use the read-only cache report before and after container recreation tests:

```bash
docker exec -w /workspace/re-SuperInfer/vllm-superinfer-v4 \
  richard-base-dev-sysnice bash -lc \
  './.venv/bin/python scripts/report_startup_diagnostics.py --json'
```

The report includes the resolved cache root, ownership, permissions,
filesystem type, aggregate sizes, derived-cache presence, and configured-path
consistency. It does not create, delete, or modify files.

## Deferred Experiments

These experiments require an available GPU allocation in
`richard-base-dev-sysnice` and a controlled, runtime-equivalent benchmark:

1. From the host, invoke the container-local launch and benchmark scripts with
   `docker exec -w /workspace/re-SuperInfer/vllm-superinfer-v4
   richard-base-dev-sysnice bash -lc ...`; collect cold-ish, warm, and
   second-warm startup baselines with phase logs.
2. Verify whether the compile/AOT artifacts are reused by the active model and
   configuration, rather than assuming the cache directory is sufficient.
3. Test a persistent `VLLM_CACHE_ROOT` mount and compare startup timing.
4. Test `VLLM_ENABLE_STARTUP_PLAN=1` only after a baseline, verifying KV bytes,
   block count, graph coverage, memory use, and outputs.
5. Compare loader/prefetch alternatives only if they preserve model loading
   correctness and steady-state performance.

Any candidate that changes throughput, TTFT, TPOT, concurrency, KV capacity,
graph coverage, memory safety, or outputs is rejected.

## Rollback

The observability change can be disabled without code changes by omitting
`VLLM_STARTUP_PHASE_LOGGING` or setting it to `0`. Do not enable startup-plan
reuse or alter cache mounts as part of rollback. For a code rollback, revert
only the files introduced for startup diagnostics and this audit; preserve
unrelated SuperInfer work in the working tree.

## Current Blockers

- Both GH200 GPUs are occupied by another VLLM service, so no valid startup or
  steady-state baseline has been collected for this task.
- A controlled cold/warm/second-warm timing table is still missing.
- AOT artifact reuse for DeepSeek/DSpark and persistence of
  `/home/richard/.cache/vllm` across recreation of
  `richard-base-dev-sysnice` are unverified.
