# Server Hang And Lifecycle Postmortem

## Incident

The project experienced pressure and lifecycle failures during the forward-port.
The incidents were not one single defect; they fell into three classes:

1. allocator/free-list corruption under full GPU-KV pressure;
2. asynchronous transfer quiescence around final requests;
3. parent/process shutdown and IPC bookkeeping residue.

## Symptoms

Observed symptoms included:

- EngineCore death after block exhaustion;
- stores/loads followed by EngineCore death;
- linked-list/free-block counter inconsistency;
- server shutdown after parent process disappearance;
- `resource_tracker` semaphore/shared-memory warnings;
- one separate September 6 startup failure with NCCL CUDA OOM during communicator initialization.

## Root Causes And Fixes

### Allocator/Free-List

Late allocation failures left partial state. The fix made allocation and request
block changes transactional and hardened list/counter invariants.

### Transfer Quiescence

The engine could see no active requests while D2H stores still had pending CUDA
events. The fix connected pending push work to EngineCore liveness and added
quiescence gates to the harness.

### Parent Shutdown

In some historical logs, workers reported `Parent process exited` without an
API-parent shutdown line. The current evidence supports parent disappearance or
external termination, not a proven EngineCore fault. Server-only instrumentation
now records API PID, received signal names, and final API exit status when the
process has time to log them.

### IPC Warnings

`resource_tracker` warnings concern Python multiprocessing registrations. They
are not direct proof of GPU or host-RAM leakage. Repeated `/dev/shm` snapshots
and owner/process state are the safe investigation path.

## Reliability Evidence

The accepted post-fix runs include:

- 120/120 requests over five exact-token pressure rounds;
- 48/48 requests at 524K prompt / 10K output over two rounds;
- zero stale-capacity interval;
- zero final pending transfer age;
- zero EngineDead/CUDA OOM/Xid events in the cited post-fix soak;
- deterministic reload with zero output mismatches.

## Separate NCCL OOM Incident

The newer log `direct_reasoning_superinfer_8202_20260906_101659.log` records a
real startup failure:

```text
NCCL WARN Cuda failure 2 'out of memory'
RuntimeError: NCCL error: unhandled cuda error
```

That occurred during worker communicator initialization and is separate from
the September 7 parent-exit shutdown pattern.

## Regression Prevention

- allocator/free-list unit tests;
- scheduler capacity tests;
- exact-token pressure harness;
- exact reload harness;
- transfer-quiescence gate;
- server health/process diagnostics;
- scoped graceful stop;
- optional user-level systemd recovery with burst limiting.

## Remaining Risk

The evidence does not prove:

- every possible external signal source;
- no possible host/kernel/NVIDIA fault;
- full DSpark rollback correctness;
- physical NUMA page locality;
- universal workload stability.

## Sources

- `docs/100_reasoning_tree_offload_results.md:358-395,503-513`.
- `docs/reasoning_tree_8202_canary.md:73-121`.
- `docs/110_systemd_service_guide.md:372-437`.
- `vllm/v1/executor/multiproc_executor.py:276-307,827-845`.
- `vllm/v1/engine/utils.py:194-226`.
