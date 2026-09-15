# Original SuperInfer Idea

## Conceptual Memory Hierarchy

```text
                 GPU HBM
       ┌─────────────────────┐
       │ active weights       │
       │ hot KV blocks        │
       │ CUDA workspace       │
       └──────────┬──────────┘
                  │
          GPU ↔ CPU movement
                  │
       ┌──────────▼──────────┐
       │ Grace / CPU memory  │
       │ warm KV blocks      │
       │ inactive capacity   │
       └─────────────────────┘
```

The idea is to make CPU memory a warm capacity tier rather than an emergency
overflow path. The scheduler and transfer system should know which blocks are
hot, which are cold, which are in flight, and which must be restored.

## Mechanisms To Explain

### CPU-KV Tier

KV blocks displaced from GPU HBM are stored in CPU memory. A later request can
load those blocks back before decode or prefill needs them.

### Topology Awareness

On dual GH200, GPU and CPU memory access is not necessarily uniform. A locality
aware design tries to associate a TP rank with its GPU-local NUMA region and
treat remote memory as a costlier fallback.

### Proactive Movement

Rather than wait until the GPU free-block pool is empty, the scheduler can move
eligible blocks before a waiting request becomes impossible to admit.

### Residency And Hot/Cold State

The system needs metadata for:

- GPU-resident blocks;
- CPU-resident blocks;
- store-in-flight blocks;
- load-in-flight blocks;
- dirty or invalid blocks;
- prefix-cache visibility.

### Scheduler Interaction

Movement is not independent of scheduling. A candidate request may be cheap or
expensive to evict depending on its KV size, predicted delay, transfer
bandwidth, and decode latency sensitivity.

## Why The Original Code Could Not Be Copied Directly

The original SuperInfer code targeted an older vLLM architecture. Modern vLLM
changed:

- scheduler ownership and output structures;
- V1 engine and EngineCore lifecycle;
- KV cache configuration and block interfaces;
- prefix-cache behavior;
- connector/worker abstractions;
- speculative decoding paths;
- model-specific hybrid KV groups;
- process and shutdown behavior.

Therefore the engineering task is capability parity:

```text
understand mechanism intent
        ↓
map intent to modern interfaces
        ↓
implement conservative behavior
        ↓
validate correctness and movement
        ↓
measure pressure and latency
```

It is not a line-by-line source port.

## What This Project Adds Beyond Generic vLLM

The validated reasoning-tree track adds or hardens:

- explicit CPU-KV connector activation;
- CPU-KV transfer telemetry;
- pending transfer age;
- guarded proactive movement;
- full-sequence capacity waiting;
- allocator/free-list rollback safety;
- topology metadata and logical local/remote labels;
- exact-token pressure/reload validation;
- lifecycle cleanup and process ownership.

## What It Does Not Automatically Mean

The following are not equivalent:

```text
topology discovered
≠ physical page placement proven

bytes transferred
≠ transfer hidden behind compute

request succeeded
≠ H2D reload occurred

high GPU memory usage
≠ active KV usage

high-risk option exists
≠ high-risk behavior is production-safe
```

## Sources

- `SUPERINFER_KNOWLEDGE_BASE.md:148-251`.
- `SUPERINFER_KNOWLEDGE_BASE.md:291-343`.
- `docs/010_superinfer_delta_map.md`.
- `/home/az04297/re-SuperInfer/vllm-superinfer-v4-integration/docs/superinfer/implementation-context.md:19-104`.
