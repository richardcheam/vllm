# NUMA And Memory Behaviour

## Hardware Mapping

```text
GPU 0 / TP rank 0
        |
        +-- NUMA node 0
        +-- CPUs 0-71

GPU 1 / TP rank 1
        |
        +-- NUMA node 1
        +-- CPUs 72-143
```

The newer topology audit reports NVLink `NV18` between the GPUs and associates
each GPU with a local CPU NUMA node.

## Process Placement

The validated launcher passes:

```text
--numa-bind --numa-bind-nodes 0 1
```

This binds GPU worker subprocesses to the NUMA node associated with each visible
GPU. It establishes process CPU/memory policy, not proof that every allocated
page is physically local.

## Logical Local/Remote Planning

With `--gh200-topology-tuned`, the connector discovers GPU PCI/NUMA and peer
topology, assigns logical islands, and plans CPU-KV blocks local-first. The
validated configuration uses:

```text
local CPU pool fraction: 0.75
local bandwidth hint:    966367641600 bytes/s
remote bandwidth hint:   300647710720 bytes/s
```

The planner records local/remote labels, remote fallbacks, and an estimated
remote penalty for proactive scoring.

## What Is Verified

- GPU-to-NUMA discovery works.
- Worker rank-to-node binding is logged.
- Logical local/remote pool planning exists.
- Accepted pressure runs reported zero remote fallbacks.

## What Is Not Verified

The project has not proven physical CPU page placement with a complete
`numa_maps`, `numastat`, `smaps`, `mbind`, or `move_pages` study. Therefore:

```text
logical local label ≠ physical page locality
```

The safe wording is “topology-aware logical locality planning and NUMA process
binding,” not “all CPU KV pages are physically local.”

## Memory Budget

Validated DS4 profile:

| Tier | Role |
|---|---|
| GPU HBM | model weights, CUDA workspace, hot KV, active requests |
| GPU KV | FP8, 256-token blocks, approximately 2.74M-token capacity in one observed startup |
| CPU KV | 128 GiB total, approximately 64 GiB per TP rank |
| Host RAM | approximately 1.2 TiB observed |

Model load used approximately 74.85 GiB per worker in one startup log. CUDA
graphs and lazy allocator/workspace growth explain additional `nvidia-smi`
resident memory after a meaningful request. Logical active KV usage can return
to zero while PyTorch retains allocator/workspace reservations.

## Newer v0.26 Track

The newer integration adds an opt-in `CPU_KV_ALLOCATION_MODE=empty` profile to
avoid startup zero-fill for large CPU-KV regions. The startup audit says this
option remains an experiment until cold/warm timings and steady-state invariants
are measured.

## Sources

- `docs/090_launch_recipe.md:221-241`.
- `docs/100_reasoning_tree_offload_results.md:145-199`.
- `SUPERINFER_KNOWLEDGE_BASE.md:291-343,804-825`.
- `/home/az04297/re-SuperInfer/vllm-superinfer-v4-integration/docs/superinfer/implementation-context.md:36-59`.
- `/home/az04297/re-SuperInfer/vllm-superinfer-v4-integration/docs/startup_optimization_audit.md:48-85`.
