# Forward-Port Plan

## Baseline

The target remains a v0.26-compatible SuperInfer adaptation because the runtime
is tied to the current container's Torch/CUDA stack and DeepSeek-V4/DSpark
integration. v0.27.1 is a patch release on v0.27.0; the major release includes
a broad core and dependency change, including a PyTorch 2.13 upgrade.

Do not replace this checkout wholesale while production behavior is still being
validated. Keep the existing `vllm-modern` tree untouched.

## Already Present

- v0.27-style NUMA utilities and subprocess binding.
- NUMA CLI/configuration: `--numa-bind`, `--numa-bind-nodes`, and
  `--numa-bind-cpus`.
- Custom batched CPU KV DMA with separate source/destination strides.
- Pre-forward H2D submission and independent load/store queues.
- Explicit residency telemetry and eager re-store/final-block handling.

## Relationship To SuperInfer

The v0.27.x items below are not SuperInfer parity features. They are upstream
performance and correctness accelerators that can strengthen the v0.26 port.
Each item maps to one or more checklist areas, but it does not close the
corresponding paper/native parity item by itself.

| Upstream feature | SuperInfer checklist relationship | Parity status |
|---|---|---|
| NUMA binding | GH200 topology and transfer locality | Accelerator only |
| Partial-tail offload | DuplexKV dirty-tail and final-block correctness | Accelerator; native semantics still partial |
| Single-copy MLA layout | DeepSeek transfer layout and CPU capacity | Accelerator; not DuplexKV parity |
| Canonical KV page mappings | Heterogeneous DeepSeek residency/DMA | Accelerator; not full layout parity |
| Batch CPU KV loads | DuplexKV transfer overlap | Accelerator; not native C++ swap |
| DeepSeek kernel improvements | Runtime model throughput | Independent model optimization |
| Quantized DSpark Markov heads | DSpark efficiency | Checkpoint-dependent accelerator |

The following remain separate long-term parity work and are not supplied by
these upstream backports:

- Native C++/CUDA/ZMQ DuplexKV swap path.
- Full RotaSched/LVF waiting/running/rotary queue semantics.
- Paper-level VLT/LVF state and admission behavior.
- Complete dirty/clean residency state machine and transfer retry protocol.
- Full cross-iteration executor overlap and native failure/recovery semantics.

## Backport Candidates

1. Reliable partial-tail KV offload, aligned with our dirty-tail handling.
2. Single-copy MLA CPU layout and deduplicated replicated MLA KV.
3. Canonical per-layer KV page mappings for heterogeneous layouts.
4. v0.27 CPU KV batch-load implementation, compared against our stride-aware DMA.
5. DeepSeek-V4 kernel improvements: workspace reuse, redundant-kernel removal,
   router/top-k skips, and compact indexer KV cache.
6. Quantized DSpark Markov heads from v0.27.1, only if the checkpoint format is
   compatible.

## Upgrade Gate

Consider moving the baseline to v0.27.x only after current SuperInfer has a
matched valid baseline, pressure tests show nonzero CPU KV movement, NUMA and
transfer overlap are measured, output/KV equivalence passes, and the container
supports the required PyTorch 2.13/Triton stack.

## Rule

Prefer self-contained backports over broad release merges. Every backport needs
a source compatibility test, a vanilla control run, and a SuperInfer pressure
run before entering the production profile.

Do not mark a parity checklist item complete merely because an upstream
accelerator has been backported. Record implementation coverage and parity
coverage separately.
