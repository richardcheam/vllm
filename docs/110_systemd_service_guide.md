# DeepSeek vLLM Service Guide

This guide explains how to use the optional user-level `systemd` service for
the DeepSeek vLLM deployment. The service is designed to keep the server
available when you are away, recover from an unexpected server failure, and
remain easy to pause during development.

The service is named:

```text
deepseek-vllm.service
```

The control command is:

```text
deepseek-vllmctl
```

The native vLLM CLI remains separate:

```text
/home/az04297/re-SuperInfer/vllm-modern-reasoning/.venv/bin/vllm
```

## What Is systemd?

`systemd` is Linux's service and process manager. It can start programs, keep
track of their state, restart them after failures, collect service logs, and
start them automatically at boot or user-session startup.

It is not the vLLM engine and it does not monitor model quality. In this setup,
systemd is responsible for service lifecycle only:

```text
start the supervisor
wait for the supervisor to exit
restart it after an unexpected failure
apply retry limits and backoff
provide status and journal logs
```

The service definition is the unit file:

```text
systemd/deepseek-vllm.service
```

After installation, the user-level copy is linked at:

```text
~/.config/systemd/user/deepseek-vllm.service
```

The repository remains the source of truth. The home-directory path is only
the user systemd discovery link created by `deepseek-vllmctl install`.

## System Service Versus User Service

Linux has two common systemd scopes.

### System service

Managed with commands such as:

```bash
sudo systemctl start some-service.service
sudo systemctl status some-service.service
```

System services normally use unit files under:

```text
/etc/systemd/system/
/usr/lib/systemd/system/
```

They usually run as `root` or as a dedicated system account and can affect the
whole machine. Installing one generally requires administrator privileges.

### User service

Managed with:

```bash
systemctl --user start deepseek-vllm.service
systemctl --user status deepseek-vllm.service
```

User services use unit files under:

```text
~/.config/systemd/user/
```

They run with the permissions and identity of the user who owns them. In this
deployment that user is `az04297`.

The DeepSeek service is deliberately a **user service** because:

- it runs the existing Docker commands as `az04297`;
- it does not need a root-owned unit;
- it does not claim ownership of unrelated containers or host services;
- `deepseek-vllmctl pause` can provide a user-controlled development hold;
- its unit, state, and logs are easy to inspect without changing global host
  service configuration.

The command difference matters:

```text
systemctl ...              -> system-level service manager, usually root scope
systemctl --user ...       -> az04297's user-level service manager
```

Do not replace `systemctl --user` with `sudo systemctl` for this service. That
would target a different systemd scope and would not manage the installed user
unit.

## What Linger Means

Normally, a user systemd manager is tied to the user's login session. When the
last session ends, user services may stop.

`Linger=yes` tells the host to keep the user systemd manager available after
logout and start it at boot. It is required if DeepSeek vLLM must continue
running while `az04297` is not logged in.

Check it with:

```bash
loginctl show-user az04297 -p Linger -p State -p RuntimePath
```

Expected:

```text
Linger=yes
```

Enabling linger is a small host-level systemd setting for this user:

```bash
sudo loginctl enable-linger az04297
```

It does not:

- install or start the DeepSeek service;
- change the Docker image or container;
- change GPU settings;
- change `/dev/shm`;
- install a kernel watchdog;
- create a cron job.

Disable it later with:

```bash
sudo loginctl disable-linger az04297
```

Do this only after disabling the DeepSeek service if automatic operation is no
longer wanted.

## Command Scope

| Command | Scope and effect |
|---|---|
| `systemctl --user daemon-reload` | Rereads `az04297`'s user unit files; does not start or stop services. |
| `systemctl --user start deepseek-vllm.service` | Starts only this user service. |
| `systemctl --user stop deepseek-vllm.service` | Stops only this user service; the control wrapper also performs scoped vLLM cleanup. |
| `systemctl --user enable deepseek-vllm.service` | Enables this user unit at user-manager startup; it does not mean “start now” in this wrapper's workflow. |
| `systemctl --user disable deepseek-vllm.service` | Removes automatic user-manager startup for this unit. |
| `systemctl --user status deepseek-vllm.service` | Reads this user service's state. |
| `journalctl --user -u deepseek-vllm.service` | Reads this user service's journal. |
| `sudo systemctl ...` | Targets the system/root service scope; do not use it for this unit. |
| `docker restart richard-base-dev-sysnice` | Restarts the entire existing container; only the supervisor's dedicated GPU/NVML recovery path should do this automatically. |

The normal operator interface is `deepseek-vllmctl`, which wraps the relevant
user-systemd and scoped-server operations.

## Scope And Safety

This setup uses a **user service**, not a system-wide root service.

The service runs as user `az04297` and is scoped to:

```text
Repository: /home/az04297/re-SuperInfer/vllm-modern-reasoning
Container:  richard-base-dev-sysnice
Port:       8202
```

It calls the repository's existing `start.sh`, `stop.sh`, and diagnostic
scripts. It does not modify:

- the Docker image;
- Docker daemon configuration;
- the container's mounts or `/dev/shm` settings;
- host cron jobs;
- the host kernel watchdog;
- unrelated containers or vLLM services;
- the native `.venv/bin/vllm` executable.

The service can start the **existing** Docker container if it is stopped. It
does not recreate or rebuild the container.

This deployment currently treats `richard-base-dev-sysnice` as dedicated to
this DeepSeek vLLM service. If that changes, set
`SUPERVISOR_CONTAINER_RECOVERY=disabled` before enabling supervision; otherwise
an NVML recovery can restart the shared container.

## Service Architecture

```text
deepseek-vllm.service
        |
        +-- scripts/run_supervised.sh
                |
                +-- start.sh
                        |
                        +-- docker exec -d
                                |
                                +-- vLLM API server
                                        |
                                        +-- EngineCore/workers
```

## Exact Start And Recovery Flow

When you run:

```bash
deepseek-vllmctl start
```

the sequence is:

```text
1. deepseek-vllmctl
   -> systemctl --user start deepseek-vllm.service

2. systemd
   -> starts scripts/run_supervised.sh

3. run_supervised.sh
   -> checks the maintenance hold
   -> takes the supervisor lock
   -> checks that richard-base-dev-sysnice exists and is running
   -> calls start.sh

4. start.sh
   -> checks whether port 8202 is already occupied
   -> calls scripts/launch_direct_deepseek_superinfer.sh
   -> waits for /health
   -> sends the configured short warmup requests
   -> reports READY

5. launch_direct_deepseek_superinfer.sh
   -> runs docker exec -d in richard-base-dev-sysnice

6. run_direct_deepseek_server_inside_container.sh
   -> builds the final vLLM command
   -> starts .venv/bin/python -m vllm.entrypoints.cli.main serve

7. run_supervised.sh
   -> checks the vLLM process tree every 10 seconds
   -> checks /health every 10 seconds
   -> remains alive while the service is healthy
```

After the one-time production setup, colleagues can use the familiar front
doors:

```bash
bash start.sh --config config/reasoning-tree-superinfer.env
bash stop.sh
```

When `deepseek-vllm.service` is enabled and no maintenance hold exists,
`start.sh` delegates to `systemctl --user start` and waits for the supervisor's
ready marker plus `/health`. It does not launch a second direct Docker process.
When the service is active, `stop.sh` delegates to `systemctl --user stop`; the
supervisor then performs the scoped vLLM cleanup. The supervisor sets an
internal child-mode variable so its own calls to `start.sh` and `stop.sh` do the
low-level work instead of delegating back to systemd.

When the unit is disabled or a maintenance hold exists, `start.sh` and
`stop.sh` retain their direct development behavior.

If the server process disappears or health fails three consecutive times:

```text
run_supervised.sh
  -> collects read-only diagnostics
  -> stops the scoped vLLM process tree
  -> checks nvidia-smi -L inside the container
  -> restarts the dedicated container only if GPU/NVML is unavailable
  -> exits nonzero

systemd
  -> waits RestartSec=60s
  -> starts run_supervised.sh again
```

If startup fails three times within the configured rate-limit window,
`systemd` stops retrying and marks the service rate-limited. It does not retry
forever.

`systemd` owns restart policy and backoff. `run_supervised.sh` does not restart
itself. It starts the server, checks process presence and `/health`, collects
failure evidence, stops the scoped process tree after a failure, and exits
nonzero. `systemd` then performs the controlled restart.

The service does not act as an inference-level watchdog. It does not inspect
request quality, GPU utilization, KV pressure, or throughput to decide whether
to restart a healthy service.

## State Files And Locks

The supervisor uses this user-owned directory:

```text
~/.local/state/deepseek-vllm/
```

### `maintenance`

Path:

```text
~/.local/state/deepseek-vllm/maintenance
```

This is a **presence file**, not a file containing `true` or `false`.

```text
file exists     -> maintenance mode active
file absent     -> maintenance mode inactive
```

`deepseek-vllmctl pause` and `deepseek-vllmctl stop` create it. The supervisor
checks only whether it exists. If it exists, the supervisor exits without
starting vLLM.

`deepseek-vllmctl resume` removes it and starts supervision again.

You can inspect it safely:

```bash
test -e "${XDG_STATE_HOME:-${HOME}/.local/state}/deepseek-vllm/maintenance" \
  && printf 'maintenance=active\n' \
  || printf 'maintenance=inactive\n'
```

Do not manually delete this file while a development session is running unless
you intentionally want automatic recovery to become active again.

### `supervisor.lock`

Path:

```text
~/.local/state/deepseek-vllm/supervisor.lock
```

This is an advisory `flock` lock used to prevent two copies of
`run_supervised.sh` from running at the same time. It is not a boolean file.
The file can remain on disk when the supervisor exits; the important state is
the kernel-held lock, which is released automatically when the owning process
closes its file descriptor or exits.

Consequences:

- one active supervisor can hold the lock;
- a second supervisor exits instead of starting a duplicate service;
- deleting the file is normally unnecessary and does not terminate a running
  supervisor;
- the lock does not indicate whether vLLM itself is healthy.

The supervisor log is separate:

```text
~/.local/state/deepseek-vllm/supervisor.log
```

It records startup, health failures, GPU recovery attempts, maintenance holds,
and supervisor termination events.

### `ready`

Path:

```text
~/.local/state/deepseek-vllm/ready
```

This is a presence marker written by `run_supervised.sh` only after `start.sh`
completes readiness and warmup. When `start.sh` delegates to systemd, it waits
for both this marker and a successful `/health` response. The marker is removed
when supervision stops or a failure is detected.

## First-Time Setup

Run these commands from the repository:

```bash
cd /home/az04297/re-SuperInfer/vllm-modern-reasoning
```

Confirm the launch recipe exists:

```bash
test -f config/reasoning-tree-superinfer.env
test -x scripts/run_supervised.sh
test -x scripts/deepseek-vllmctl
test -f systemd/deepseek-vllm.service
```

Install the user command and unit:

```bash
scripts/deepseek-vllmctl install
```

Installation performs these user-scoped actions:

- creates `~/.local/bin/deepseek-vllmctl` as a symlink to the repository command;
- creates `~/.config/systemd/user/deepseek-vllm.service` as a symlink to the
  repository unit;
- creates the user state directory;
- runs `systemctl --user daemon-reload`.

Installation does **not** enable or start the service.

## Linger

User services normally run while your user systemd manager is active. To keep
the user manager available after logout and at boot, user lingering must be
enabled.

Check the current setting:

```bash
loginctl show-user az04297 -p Linger -p State -p RuntimePath
```

Required result:

```text
Linger=yes
```

If it is not enabled, an administrator-approved command is:

```bash
sudo loginctl enable-linger az04297
```

Verify it afterward:

```bash
loginctl show-user az04297 -p Linger
```

`enable-linger` creates a host-level systemd user-linger marker for this user.
It does not start vLLM by itself, change Docker, or alter the container.

## `daemon-reload`

```bash
systemctl --user daemon-reload
```

This tells the user systemd manager to reread unit files. It is required after
installing or editing the unit so systemd sees the current definition.

It does not:

- start the service;
- stop or restart the service;
- start or stop Docker;
- enable automatic startup;
- change `/dev/shm`;
- modify the host watchdog or cron.

The `deepseek-vllmctl install` command already runs this operation.

## Inspect Before Enabling

Check the installed unit:

```bash
systemctl --user cat deepseek-vllm.service
systemctl --user show deepseek-vllm.service \
  -p FragmentPath -p ExecStart -p Restart -p RestartUSec \
  -p StartLimitIntervalUSec -p StartLimitBurst -p KillMode
```

Validate the repository unit without starting it:

```bash
systemd-analyze --user verify \
  /home/az04297/re-SuperInfer/vllm-modern-reasoning/systemd/deepseek-vllm.service
```

Check the control command:

```bash
PATH="${HOME}/.local/bin:${PATH}" deepseek-vllmctl --help
```

## Enable Versus Start

### Enable

```bash
deepseek-vllmctl enable
```

`enable` tells user systemd to start `deepseek-vllm.service` when the user
systemd manager starts. With `Linger=yes`, that can include boot and post-logout
operation.

`enable` does not necessarily start the service immediately in this wrapper.

Inspect the result:

```bash
systemctl --user is-enabled deepseek-vllm.service
```

Expected result:

```text
enabled
```

### Start

```bash
deepseek-vllmctl start
```

`start` starts supervision immediately. The supervisor then:

1. checks the maintenance hold;
2. starts the existing container if needed;
3. calls `start.sh`;
4. waits for `/health`;
5. waits for startup warmup to complete;
6. enters a process/health monitoring loop.

Check whether it is active:

```bash
systemctl --user is-active deepseek-vllm.service
```

Expected result:

```text
active
```

For the first production activation, verify port ownership before starting:

```bash
docker exec richard-base-dev-sysnice pgrep -af \
  'vllm.*serve.*--port 8202' || true
```

If another manually managed server is already using port `8202`, do not start
the service until that server is stopped or the ownership decision is clear.

## Recommended Production Activation

If the server is currently running manually, do not start the service on top of
it. First choose a controlled handoff window.

Stop the manually managed server:

```bash
cd /home/az04297/re-SuperInfer/vllm-modern-reasoning
CONTAINER_NAME=richard-base-dev-sysnice PORT=8202 bash stop.sh
```

Confirm no server remains:

```bash
docker exec richard-base-dev-sysnice pgrep -af \
  'vllm.*serve.*--port 8202' || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits
```

Then enable and start supervision:

```bash
deepseek-vllmctl enable
deepseek-vllmctl start
```

Confirm service and API state:

```bash
deepseek-vllmctl status
curl -fsS http://127.0.0.1:8202/health
```

Do not run this handoff while another operator is independently managing the
same port.

## Development Mode

Before editing code, running a benchmark, or manually stopping the server:

```bash
deepseek-vllmctl pause
```

`pause`:

- creates `~/.local/state/deepseek-vllm/maintenance`;
- stops the user service;
- calls the scoped `stop.sh` cleanup;
- leaves the Docker container itself running;
- prevents the service from starting automatically.

Now manual development operations are safe:

```bash
cd /home/az04297/re-SuperInfer/vllm-modern-reasoning
bash start.sh --config config/reasoning-tree-superinfer.env
# run benchmarks or tests
bash stop.sh
```

When development is complete, remove the hold and resume supervision:

```bash
deepseek-vllmctl resume
```

Do not use plain `systemctl --user start` while the maintenance hold exists;
the supervisor will exit without starting the server.

## Status And Logs

Show hold state and systemd status:

```bash
deepseek-vllmctl status
```

Equivalent systemd status command:

```bash
systemctl --user status deepseek-vllm.service --no-pager
```

Follow service-manager logs:

```bash
deepseek-vllmctl logs
```

Equivalent journal command:

```bash
journalctl --user -u deepseek-vllm.service -f
```

The detailed vLLM server log remains under:

```text
/home/az04297/re-SuperInfer/vllm-modern-reasoning/logs/
```

Find the current server log:

```bash
ls -lt /home/az04297/re-SuperInfer/vllm-modern-reasoning/logs/*.log
```

The server-local wrapper records lines such as:

```text
[server-supervisor] started api_pid=...
[server-supervisor] received signal=TERM api_pid=...; forwarding
[server-supervisor] api_exit=code status=0
[server-supervisor] api_exit=signal signal=TERM status=143
```

These lines can distinguish a clean exit, a logged signal, and a nonzero API
exit. An unlogged `SIGKILL` cannot be attributed to an external sender by
server-only code.

## Automatic Recovery

The unit uses:

```ini
Restart=on-failure
RestartSec=60s
TimeoutStartSec=10min
StartLimitIntervalSec=15min
StartLimitBurst=3
```

Meaning:

- `Restart=on-failure`: restart after a nonzero exit or unexpected failure;
- `RestartSec=60s`: wait 60 seconds before a restart, allowing CUDA/NCCL and
  process resources to settle;
- `TimeoutStartSec=10min`: allow up to 10 minutes for model loading, CUDA graph
  setup, and startup warmup;
- `StartLimitIntervalSec=15min`: count starts within a rolling 15-minute window;
- `StartLimitBurst=3`: allow at most three starts in that window before systemd
  rate-limits the service.

This prevents a broken configuration, unavailable GPU, repeated NCCL failure,
or model startup error from creating an endless restart loop.

Repeated server-log termination at the same wall-clock time without an
EngineCore, CUDA, NCCL, or OOM error indicates an external stop path is likely,
but server-only instrumentation cannot identify the sender. Once this service
is enabled, systemd will recover from that class of unexpected stop.

If rate-limited, inspect the reason:

```bash
systemctl --user status deepseek-vllm.service --no-pager
journalctl --user -u deepseek-vllm.service --since '30 minutes ago' --no-pager
```

After correcting the cause, reset the failed activation state and start again:

```bash
systemctl --user reset-failed deepseek-vllm.service
deepseek-vllmctl start
```

Do not increase the burst limit to hide repeated failures. Investigate the
failure artifacts first.

## GPU/NVML Recovery

The supervisor also handles a running-but-broken container GPU state. This is
different from a stopped container:

```text
container is running
  but nvidia-smi inside the container fails
        -> collect failure evidence
        -> scoped vLLM cleanup
        -> docker restart of the existing dedicated container
        -> wait for GPU visibility
        -> systemd retries after its 60-second delay
```

The defaults are:

```ini
SUPERVISOR_CONTAINER_RECOVERY=restart
SUPERVISOR_CONTAINER_RESTART_TIMEOUT_S=30
SUPERVISOR_GPU_RECOVERY_WAIT_S=15
```

The recovery path only runs after startup, process-presence, or health failure.
It first runs `nvidia-smi -L` inside the container. If GPU visibility is still
healthy, it does not restart the container. If GPU visibility is broken, it
restarts the existing container and checks GPU visibility again.

Disable container restart recovery for a shared container:

```bash
sed -i 's/^SUPERVISOR_CONTAINER_RECOVERY=.*/SUPERVISOR_CONTAINER_RECOVERY="disabled"/' \
  config/reasoning-tree-superinfer.env
systemctl --user daemon-reload
```

Then restart the service when it is safe to apply the change:

```bash
deepseek-vllmctl pause
deepseek-vllmctl resume
```

For the current dedicated-container deployment, leave it enabled. During
development, `deepseek-vllmctl pause` creates the maintenance hold before
stopping the service, so this recovery path cannot race with manual work.

## Failure Diagnostics

The supervisor enables failure-only diagnostics before scoped cleanup. Healthy
service operation does not run this collector.

Collected evidence can include:

- container state and OOM/restart status;
- process ancestry and status;
- GPU and CUDA/NVML visibility;
- `/dev/shm` usage and shared-memory descriptors;
- kernel NVIDIA/OOM messages;
- health and metrics responses;
- bounded server and Docker log tails.

To collect a snapshot manually without stopping anything:

```bash
cd /home/az04297/re-SuperInfer/vllm-modern-reasoning
CONTAINER_NAME=richard-base-dev-sysnice PORT=8202 \
  OUTPUT_DIR="/tmp/deepseek-vllm-incident-$(date -u +%Y%m%dT%H%M%SZ)" \
  bash scripts/collect_vllm_failure_diagnostics.sh
```

The collector is read-only with respect to the service, container, host
settings, and `/dev/shm`. It writes report files only to the selected output
directory.

## Safe Stop And Disable

For a temporary development stop:

```bash
deepseek-vllmctl pause
```

For a longer disablement:

```bash
deepseek-vllmctl disable
```

`disable` creates the maintenance hold, disables automatic user-manager
startup, stops the service, and runs scoped server cleanup.

Verify:

```bash
systemctl --user is-enabled deepseek-vllm.service 2>&1 || true
systemctl --user is-active deepseek-vllm.service 2>&1 || true
deepseek-vllmctl status
```

The Docker container remains available unless separately stopped by an
operator. No container restart is performed.

## Uninstall And Rollback

To remove the user service after it is disabled:

```bash
deepseek-vllmctl disable
deepseek-vllmctl uninstall
```

This removes only:

```text
~/.config/systemd/user/deepseek-vllm.service
~/.local/bin/deepseek-vllmctl
```

It does not remove the repository, virtual environment, Docker image,
container, model cache, or logs.

To disable lingering later, use the administrator-approved command:

```bash
sudo loginctl disable-linger az04297
```

That affects whether the user systemd manager persists after logout. It does
not uninstall vLLM or change the Docker container.

## Troubleshooting

### Service Is Already Running Manually

Check:

```bash
docker exec richard-base-dev-sysnice pgrep -af \
  'vllm.*serve.*--port 8202' || true
```

Do not start the systemd service on the same port. Either leave manual mode in
place, or stop it and perform the controlled production handoff.

### Service Exits Immediately

Check for the maintenance hold:

```bash
test -e "${XDG_STATE_HOME:-${HOME}/.local/state}/deepseek-vllm/maintenance" \
  && printf 'maintenance hold active\n' \
  || printf 'maintenance hold inactive\n'
```

Check service logs:

```bash
journalctl --user -u deepseek-vllm.service -n 200 --no-pager
```

Check the repository supervisor directly without starting it:

```bash
scripts/run_supervised.sh --help
```

### Service Cannot Start The Container

Inspect the existing container without modifying it:

```bash
docker inspect --format \
  'status={{.State.Status}} running={{.State.Running}} image={{.Config.Image}} restart={{.RestartCount}} oom={{.State.OOMKilled}}' \
  richard-base-dev-sysnice
```

The supervisor only starts the existing container. It does not recreate it.

### GPU Or NCCL Initialization Failure

Inspect the latest incident artifact and server log:

```bash
ls -lt failure_artifacts/
ls -lt logs/*.log
```

Look for:

```text
NCCL WARN Cuda failure 2 'out of memory'
NCCL error
Can't initialize NVML
EngineCore encountered a fatal error
WorkerProc initialization failed
```

Do not increase restart limits before determining whether the GPU is occupied,
the container has a driver compatibility issue, or a prior process still owns
memory.

### `resource_tracker` Warnings

Warnings such as:

```text
resource_tracker: There appear to be leaked semaphore objects
resource_tracker: There appear to be leaked shared_memory objects
```

are Python multiprocessing shutdown bookkeeping. They are not automatically
proof of GPU or host-RAM leakage. Compare `/dev/shm` usage and object inventory
across snapshots, and use the normal graceful stop path instead of deleting
shared-memory objects manually.

## Command Summary

| Command | Effect |
|---|---|
| `deepseek-vllmctl install` | Install user command and unit; reload systemd; do not start. |
| `deepseek-vllmctl enable` | Enable automatic start at user-manager startup; remove hold. |
| `deepseek-vllmctl start` | Start supervised service now. |
| `deepseek-vllmctl status` | Show maintenance hold and systemd status. |
| `deepseek-vllmctl logs` | Follow user-service journal logs. |
| `deepseek-vllmctl pause` | Create development hold and stop scoped server. |
| `deepseek-vllmctl resume` | Remove development hold and start supervision. |
| `deepseek-vllmctl disable` | Disable automatic startup, create hold, and stop server. |
| `deepseek-vllmctl uninstall` | Remove installed user unit and command link after stopping. |
| `.venv/bin/vllm` | Native vLLM CLI; not the service controller. |

## Current Installation Check

The following commands inspect the current setup without starting or stopping
the service:

```bash
systemctl --user is-enabled deepseek-vllm.service 2>&1 || true
systemctl --user is-active deepseek-vllm.service 2>&1 || true
loginctl show-user az04297 -p Linger -p State
readlink -f "${HOME}/.config/systemd/user/deepseek-vllm.service" 2>/dev/null || true
readlink -f "${HOME}/.local/bin/deepseek-vllmctl" 2>/dev/null || true
```

The intended initial state after installation is:

```text
unit: linked
enabled: no
active: no
maintenance hold: inactive
```

Enable and start only after the manual server on port `8202` has been handed
off deliberately.
