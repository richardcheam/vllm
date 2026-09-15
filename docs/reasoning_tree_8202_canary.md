# Reasoning-Tree 8202 Canary

This is a reversible canary procedure. It does not change the
`vllm-modern` checkout. Do not run it while the current server owns port 8202
or the GPUs.

## Before Switching

Record the current service state without stopping it:

```bash
curl -fsS http://127.0.0.1:8202/health
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
```

When the service is free, use the normal operator-approved stop procedure and
confirm that port 8202 is unused and no `APIServer`, `EngineCore`, or
`VLLM::Worker` processes remain.

## Launch

Run from this checkout:

```bash
cd /home/az04297/re-SuperInfer/vllm-modern-reasoning
CONTAINER_NAME=richard-base-dev-sysnice PORT=8202 \
  bash start.sh
```

The launcher explicitly uses:

- `/workspace/re-SuperInfer/vllm-modern-reasoning`;
- `max_model_len=1048576`;
- tensor parallel size 2 and expert parallelism;
- FP8 KV cache and block size 256;
- CPU KV offload and proactive swap;
- `capacity_wait_timeout=30` seconds;
- normal SuperInfer mode with high-risk and block-first disabled;
- three short sequential API warmup requests after health readiness.

For the meaning of every serving argument, including GH200 topology tuning,
NUMA binding, CPU-KV offload, CUDA graphs, and scheduler weights, see the
[production launch recipe](090_launch_recipe.md).

`start.sh` reports `READY` only after all enabled warmup requests succeed. The
warmup is intentionally small so it primes the tokenizer, model request path,
CUDA execution, and short decode path without consuming meaningful context
capacity or exercising full CPU-KV pressure. If launch, readiness, or warmup
validation fails, `start.sh` invokes the scoped `stop.sh` cleanup path so a
failed startup does not leave a detached server occupying the port.

Startup request warmup can be disabled or tuned with environment variables:

```bash
WARMUP_ENABLED=0 bash start.sh
WARMUP_REQUESTS=1 WARMUP_MAX_TOKENS=4 bash start.sh
```

`WARMUP_TIMEOUT_S` limits each warmup request and defaults to `120` seconds.
Keep warmup requests short; use the pressure and reload harnesses for large
contexts and CPU-KV validation rather than increasing startup warmup size.

Startup logs are written under this checkout's `logs/` directory. The command
and profile sidecars record the effective launch settings. The in-container
launcher also records `[server-supervisor]` API PID and final exit status lines
in the server log. This is server-local lifecycle evidence only; it does not
restart the process or modify host watchdog/cron configuration.

## Verify

```bash
curl -fsS http://127.0.0.1:8202/health
curl -fsS http://127.0.0.1:8202/v1/models
```

During the canary, watch both the server log and GPU ownership:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
curl -fsS http://127.0.0.1:8202/metrics > /tmp/reasoning-tree-8202.prom
```

If the engine fails, preserve the server log, process list, and `nvidia-smi`
output before starting another server. The read-only diagnostics include:

- container state, image ID, restart count, exit code, and OOM flag;
- API/EngineCore/worker process ancestry and status;
- host and container GPU/CUDA/NVML visibility;
- host NVIDIA/OOM messages and container kernel messages;
- `/dev/shm` size, inode usage, object inventory, and open vLLM shared-memory FDs;
- endpoint health/metrics and bounded server/Docker log tails.

The collector does not stop processes, restart the container, delete `/dev/shm`
objects, or record the container environment. It is safe to run before rollback.
The `resource_tracker` semaphore/shared-memory warnings should be interpreted
as shutdown bookkeeping evidence first, not as proof of a GPU or host-RAM leak.
Compare repeated snapshots and `/dev/shm` usage before taking cleanup action.

If a later incident shows only `Parent process exited` without an API signal or
`[server-supervisor] api_exit=...` line, the API parent likely disappeared before
it could log the cause. Server-only instrumentation cannot attribute that event
to a particular external user or host process.

For a read-only snapshot before rollback:

```bash
cd /home/az04297/re-SuperInfer/vllm-modern-reasoning
CONTAINER_NAME=richard-base-dev-sysnice PORT=8202 \
  bash scripts/collect_vllm_failure_diagnostics.sh
```

This records host/container GPU ownership, process trees, `/dev/shm`, container
state, health, and metrics without killing or restarting anything. Live
`VLLM::Worker` processes in the snapshot mean VRAM is still actively owned by a
process; no-process VRAM residue should be investigated at the driver level.

Startup failure collection is disabled by default so healthy startup has no
additional Docker/GPU inspection overhead. Enable it for a diagnostic launch:

```bash
COLLECT_START_FAILURE_DIAGNOSTICS=1 \
  bash start.sh --config config/reasoning-tree-superinfer.env
```

When enabled, the collector runs only after launch/readiness/warmup failure and
before the scoped shutdown cleanup. It can also be tuned with
`DIAGNOSTICS_LOG_TAIL_LINES` and `DIAGNOSTICS_DOCKER_LOG_SINCE`.

## Roll Back

Stop the reasoning-tree canary through the approved operator procedure. Confirm
that its child workers have exited and GPU memory has been released. Then start
the existing `vllm-modern` service using its normal production start hook.

Do not use `git reset`, `git checkout`, or broad process-kill commands as part of
the rollback.
## Stop The Server

Use the same container that was used to launch the server. The stop script
matches the module-based command by port and first sends `SIGTERM` to the API
parent. This gives vLLM's own shutdown path an opportunity to close the engine
and perform process-local distributed, CUDA, shared-memory, and CPU-offload
cleanup. It then signals the captured descendants, including orphaned
`VLLM::EngineCore` and `VLLM::Worker_*` processes in the reasoning tree, before
using a last-resort `SIGKILL`:

```bash
CONTAINER_NAME=richard-base-dev-sysnice PORT=8202 \
  bash /home/az04297/re-SuperInfer/vllm-modern-reasoning/stop.sh
```

The API-parent grace period defaults to `30` seconds and the subsequent
process-tree shutdown window defaults to `120` seconds. Override them only when
needed, for example:

```bash
GRACEFUL_API_TIMEOUT_S=60 SHUTDOWN_TIMEOUT_S=300 \
  CONTAINER_NAME=richard-base-dev-sysnice PORT=8202 \
  bash /home/az04297/re-SuperInfer/vllm-modern-reasoning/stop.sh
```

The script does not perform a separate shell-level CPU-KV transfer drain; the
worker shutdown path synchronizes in-flight copy events. It also does not use
PID/PGID state files because this detached-container launch path does not write
matching state files. Process cleanup is therefore scoped to the configured
container, port, and reasoning-tree working directory.
