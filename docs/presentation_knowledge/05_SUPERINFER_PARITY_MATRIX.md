# SuperInfer Parity Matrix

Status labels are capability-level, not a percentage. A capability can be
implemented while its performance effect or physical behavior remains
unmeasured:

- `VERIFIED`: accepted evidence exists for the documented profile.
- `FUNCTIONAL`: implementation and focused tests exist; broader validation remains.
- `RUNTIME-REACHED; OVERLAP UNMEASURED`: the production path executes the
  mechanism, but copy/compute exposure or performance effect is not measured.
- `LOGICAL ONLY; PHYSICAL PLACEMENT UNVERIFIED`: metadata/planning exists, but
  physical memory placement has not been measured.
- `SAFETY-GATED SUBSET`: conservative behavior is implemented, but the full
  original policy or aggressive modes are not validated.
- `DIRECTIONAL EVIDENCE ONLY`: a comparison exists, but workload/transfer
  differences prevent a causal performance claim.
- `NOT VERIFIED`: code exists or is intended, but decisive measurement is absent.
- `NOT PORTED`: original behavior is not implemented.
- `SUPERSEDED BY MODERN VLLM`: modern vLLM already provides the relevant base mechanism.

| Capability | Original concept | Validated reasoning tree | Newer v0.26 integration | Status/evidence |
|---|---|---|---|---|
| GPU/CPU KV tier | Warm CPU capacity tier | Real 128 GiB CPU-KV path | 128 GiB profiles, including empty allocation | `VERIFIED` for documented profiles |
| D2H store | Move cold KV GPU -> CPU | Positive pressure/reload stores | Positive stores in v0.26 pressure/reload | `VERIFIED` |
| H2D reload | Restore CPU KV CPU -> GPU | Positive loads and exact outputs | Exact reload gate and loads | `VERIFIED` |
| Async queues | Background movement | Bounded load/store queues | Bounded queues plus quiescence | `VERIFIED` for accepted profiles |
| CUDA streams/events | Overlap and ordering | Separate directional streams/events | Compute-gated stores and pre-forward loads | `RUNTIME-REACHED; OVERLAP UNMEASURED` |
| Pinned CPU memory | Efficient DMA source/destination | Host registration path | Empty allocation and pinning variants | `FUNCTIONAL; PINNING PERFORMANCE UNMEASURED` |
| GPU topology discovery | Local/remote awareness | GPU/NUMA discovery and logical pools | Same plus v0.26 profiles | `VERIFIED DISCOVERY` |
| Logical local/remote planning | Locality-aware pool decisions | Logical labels and cost hints | Same plus v0.26 profiles | `FUNCTIONAL; PHYSICAL PLACEMENT UNVERIFIED` |
| Physical NUMA placement | Actual local pages | Not measured with `numa_maps`/`move_pages` | Not measured | `LOGICAL ONLY; PHYSICAL PLACEMENT UNVERIFIED` |
| Proactive swapping | Move before hard exhaustion | Guarded conservative movement | High-risk DSpark rotation candidate | `SAFETY-GATED SUBSET` |
| RotaSched/LVF | Paper scheduling semantics | Local VLT-like scoring and safe movement | More aggressive integration | `SAFETY-GATED SUBSET; FULL PAPER SEMANTICS NOT PORTED` |
| Block-first layout | Native block-first transfer | Disabled/fallback for DeepSeek | Disabled for DeepSeek/TP=2 | `NOT PORTED/NOT VERIFIED` |
| Prefix caching | Reuse shared prompt blocks | Enabled and integrated | Enabled and tested | `FUNCTIONAL` |
| Transfer quiescence | Drain asynchronous work | Post-fix pending-work gate | Explicit v0.26 gate | `VERIFIED` for accepted runs |
| Native DuplexKV | C++/CUDA/ZMQ transfer system | Not implemented | Native DMA bridge only | `NOT PORTED; BRIDGE IS NOT EQUIVALENT` |
| DSpark | Speculative execution integration | Disabled | Enabled, rollback still incomplete | `NOT VERIFIED` for high-risk stress |
| Full output correctness | Request result preservation | Zero mismatch reload and pressure | Exact reload zero mismatch | `VERIFIED` for profiles |
| Throughput gain | Performance improvement | Not claimed | Directional matrix only | `DIRECTIONAL EVIDENCE ONLY` |
| Lifecycle cleanup | Safe teardown | Distributed/CUDA/IPC/offload cleanup | Quiescence and process-group tools | `VERIFIED` operational path |

## Strongest Completed Pieces

- Real CPU-KV stores and loads.
- Exact-token pressure and reload harnesses.
- Transactional allocator/free-list safety.
- Quiescent shutdown and pending-transfer accounting.
- Conservative DeepSeek GPU-derived layout.

## Remaining Parity Gaps

- Physical page placement proof.
- Full block-first DeepSeek semantics.
- Native DuplexKV.
- Full RotaSched/LVF semantics.
- DSpark rejection/rollback/restore stress.
- Transfer overlap and scheduler-visible stall measurement.

## Sources

- `docs/100_reasoning_tree_offload_results.md:90-120`.
- `SUPERINFER_KNOWLEDGE_BASE.md:473-499`.
- `/home/az04297/re-SuperInfer/vllm-superinfer-v4-integration/docs/superinfer/feature-checklist.md`.
- `/home/az04297/re-SuperInfer/vllm-superinfer-v4-integration/docs/superinfer/validation-status.md`.
