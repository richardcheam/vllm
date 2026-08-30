# Implementation Context

## Scope

The target is this checkout only:

```text
/data1/home/az04297/re-SuperInfer/vllm-superinfer-v4
```

The production checkout is:

```text
/data1/home/az04297/re-SuperInfer/vllm-modern
```

It is a read-only behavioral reference and must not be changed.

## Source priority

1. `2601.20309v2.pdf`
2. `/data1/home/az04297/re-SuperInfer/SuperInfer`
3. `/data1/home/az04297/re-SuperInfer/vllm-modern`
4. vLLM `v0.26.0` APIs and invariants

The paper defines the intended RotaSched/VLT and DuplexKV mechanisms. The
official checkout is the native C++/CUDA implementation reference. The
production checkout is the strongest evidence for behavior that has already
been exercised on this machine, but it is not exact paper parity.

Upstream vLLM releases are a fourth implementation source, not a parity
authority. A v0.27.x feature may be backported when it improves this fixed
deployment, but it must be labeled as an accelerator unless it reproduces a
specific SuperInfer mechanism and passes that mechanism's correctness gate.

## Hardware evidence

The host was inspected on 2026-08-03:

- Two `NVIDIA GH200 144G HBM3e` GPUs.
- `nvidia-smi topo -m` reports `NV18` between the GPUs.
- GPU 0 is associated with CPU NUMA node 0 and CPUs 0-71.
- GPU 1 is associated with CPU NUMA node 1 and CPUs 72-143.
- The host exposes about 1.2 TiB of memory.
- CUDA driver reports version 580.95.05 and CUDA 13.0.

The topology-aware implementation should use local NUMA memory first and treat
remote memory as a measured fallback, not as an equivalent pool.

`SWAP_CPU_MEMORY_GB` is the SuperInfer CPU KV capacity. The generic
`KV_OFFLOADING_SIZE` setting is reserved for the separate `native-offload`
comparison profile; the launcher does not pass both options for SuperInfer.
The default profiles remain at their existing CPU capacities and use
`CPU_KV_ALLOCATION_MODE=zero`. The opt-in `.env.superinfer-cpu128` profile uses
128 GiB total, or 64 GiB per TP rank, with `empty` allocation to avoid
zero-filling CPU KV pages before they are populated by a completed store.
The maximum-throughput `.env.superinfer-serving` profile uses the same 128 GiB
capacity and enables high-risk proactive DSpark rotation while retaining
GPU-derived DeepSeek layout fallback.

## Model and DSpark evidence

The sysnice container contains:

```text
/workspace/models/DeepSeek-V4-Flash-0731
/workspace/models/DeepSeek-V4-Flash-DSpark
```

The model configuration reports:

- `model_type=deepseek_v4`
- `num_hidden_layers=43`
- `dspark_block_size=5`
- `dspark_target_layer_ids=[40, 41, 42]`
- `dspark_markov_rank=256`

vLLM 0.26.0 requires DSpark's speculative length to be at least its block
size. The initial baseline therefore uses five speculative tokens. DSpark also
rejects pipeline parallelism, so the initial deployment is TP=2, PP=1.

## Architectural direction

The port is performance-first within this fixed deployment envelope. It should
preserve v0.26.0's newer contracts rather than copy old vLLM internals:

- Use `KVCacheConfig`, `KVCacheTensor.block_stride`, and the v0.26 connector
  lifecycle for cache movement.
- Keep CPU and GPU block identifiers in separate namespaces.
- Use connector-owned transfer state and worker event ordering.
- Integrate rotation without breaking `SchedulerOutput`, speculative lookahead,
  block zeroing, or copy-on-write semantics.
- Enable block-first transfers only after DeepSeek-V4 layout validation.
- Keep explicit fallbacks for unsupported cache groups or execution modes.
- Treat `PROACTIVE_SWAP_BUDGET` as a per-step rotation/transfer cap, not as a
  permanent GPU free-block watermark. The free-block target remains driven by
  actual waiting pressure and vLLM admission headroom.

This is intentionally not a full SuperInfer clone. In particular, native
C++/CUDA/ZMQ DuplexKV, complete RotaSched/LVF queue semantics, and full
paper-level residency/retry behavior remain separate parity objectives. The
working strategy is to combine the highest-value mechanisms that are safe in
v0.26 with selected upstream accelerators, then measure actual saturated CPU
KV movement before claiming a gain.

## Performance definition

The primary result is the best stable saturation point, not the best result at
low load. A run is saturated only when GPU KV pressure causes actual CPU KV
residency and sustained transfers while throughput remains stable and transfer
backlog does not grow without bound.

GPU-fitting tests remain regression guards. They are not the primary optimization
target for SuperInfer.

## Environment

Runtime and benchmark commands must execute in:

```text
richard-base-dev-sysnice
```

The target has a dedicated `.venv` at `vllm-superinfer-v4/.venv`. It is created
with `uv` and uses system site packages so the container's compatible Torch build
is not replaced. Proxy credentials are not stored in this repository; if
dependency installation needs the supplied proxy, export it only in the shell
running the install command.
