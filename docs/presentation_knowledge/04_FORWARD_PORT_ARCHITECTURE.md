# Forward-Port Architecture

## Primary Validated Architecture

```text
Client request
     ↓
OpenAI-compatible API
     ↓
vLLM V1 EngineCore
     ↓
Scheduler and admission
     ↓
GPU BlockPool / KV manager
     ├───────────────┐
     │               │
 GPU-resident     CPU-KV connector
 blocks            / manager
                     ├───────────────┐
                     │               │
                D2H store        H2D load
                     │               │
              bounded queue  bounded queue
                     │               │
              D2H CUDA stream H2D CUDA stream
                     │               │
              CPU KV blocks / events
```

## Request And Movement Flow

### D2H Store

```text
forward computation
        ↓
compute-done CUDA event
        ↓
store metadata prepared
        ↓
D2H queue
        ↓
low-priority D2H stream
        ↓
CPU-KV block becomes discoverable
        ↓
GPU block ownership released when safe
```

### H2D Reload

```text
request references CPU-resident prefix
        ↓
CPU lookup/admission
        ↓
load metadata created
        ↓
H2D queue
        ↓
low-priority H2D stream
        ↓
event completion
        ↓
request resumes with restored GPU KV
```

## Core Components

| Mechanism | Purpose | Evidence |
|---|---|---|
| `SimpleCPUOffloadConnector` | Modern V1 connector boundary | `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py` |
| CPU-KV manager | CPU block allocation, lookup, store/load state | `vllm/v1/simple_kv_offload/manager.py` |
| Worker | CUDA streams, events, copy backend, cleanup | `vllm/v1/simple_kv_offload/worker.py` |
| Copy backend | Bounded async transfer submission | `vllm/v1/simple_kv_offload/copy_backend.py` |
| Scheduler | Admission, proactive movement, capacity waits | `vllm/v1/core/sched/scheduler.py` |
| Topology planner | Logical island/local/remote placement metadata | `vllm/v1/simple_kv_offload/topology.py` |
| Metrics | Transfer counts, bytes, age, capacity, queue/deferred waits | `vllm/v1/metrics/loggers.py` |

## Allocation And Ownership

The validated profile requests 128 GiB total CPU-KV capacity. With TP=2, each
rank receives approximately 64 GiB. The scheduler and worker track separate
GPU/CPU block identifiers and transfer state.

The production DeepSeek path uses the GPU-derived layout. Block-first is
disabled because DeepSeek-V4/TP=2 block-first addressing is not validated.

## Proactive Movement

`proactive_swap_budget=2400` is a scheduler movement cap/target input, not a
permanent GPU free-block reservation. The scheduler considers waiting pressure,
CPU capacity, safe candidates, pending transfers, and VLT-style timing/cost
terms before selecting movement.

The non-high-risk profile retains conservative candidate gates. The newer v0.26
integration has a separate high-risk DSpark rotation profile; it must not be
described as equivalent to the validated conservative profile.

## Quiescence

Asynchronous D2H work may outlive a client response. The engine therefore needs
to remain live until pending transfer events and completion metadata drain.
The post-fix implementation exposes pending push work to EngineCore and the
harness requires transfer queues and pending ages to reach zero.

## Difference From Original SuperInfer

The modern implementation preserves the high-value ideas while using modern
V1 interfaces. It does not reproduce native C++/CUDA/ZMQ DuplexKV or all
paper-level RotaSched/LVF semantics.

## Newer v0.26 Integration Track

The newer integration adds:

- vLLM v0.26 connector contracts;
- DSpark;
- `CPU_KV_ALLOCATION_MODE=empty`;
- native DMA bridge experiments;
- stronger quiescence gates;
- high-risk proactive rotation;
- additional exact-token benchmark profiles.

Its high-risk DSpark rejection/rollback behavior remains incomplete.

## Sources

- `docs/100_reasoning_tree_offload_results.md:176-200`.
- `vllm/v1/simple_kv_offload/manager.py`.
- `vllm/v1/simple_kv_offload/worker.py`.
- `vllm/v1/core/sched/scheduler.py`.
- `SUPERINFER_KNOWLEDGE_BASE.md:255-385`.
- `/home/az04297/re-SuperInfer/vllm-superinfer-v4-integration/docs/superinfer/developer-notes.md`.
