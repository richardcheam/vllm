# Deployment And Operations

## Deployment Architecture

```text
user systemd manager
        ↓
deepseek-vllm.service
        ↓
scripts/run_supervised.sh
        ↓
start.sh
        ↓
launch_direct_deepseek_superinfer.sh
        ↓
docker exec -d richard-base-dev-sysnice
        ↓
run_direct_deepseek_server_inside_container.sh
        ↓
.venv/bin/python -m vllm.entrypoints.cli.main serve
        ↓
/health + warmup + API clients
```

## Runtime Ownership

- `start.sh`: operator lifecycle, config loading, readiness, warmup, startup cleanup.
- Outer launcher: host-to-container handoff and effective command sidecars.
- Inner launcher: container-local environment and final vLLM arguments.
- `stop.sh`: scoped API-parent-first graceful shutdown.
- `run_supervised.sh`: user-service observer; systemd owns restart/backoff.
- `deepseek-vllmctl`: install, enable, start, status, pause, resume, disable, uninstall.

## User Systemd

The unit is:

```text
systemd/deepseek-vllm.service
```

It is user-scoped and runs as `az04297`. It uses:

```ini
Restart=on-failure
RestartSec=30s
StartLimitIntervalSec=15min
StartLimitBurst=3
```

The burst limit prevents a broken model/config/GPU from creating an infinite
restart loop.

## Production Handoff

Do not start systemd on top of a manually running server. The safe sequence is:

```bash
deepseek-vllmctl pause
docker exec richard-base-dev-sysnice pgrep -af 'vllm.*serve.*--port 8202' || true
deepseek-vllmctl resume
```

In a controlled handoff, enable the unit first if required:

```bash
```

## Development

```bash
deepseek-vllmctl pause
bash start.sh --config config/reasoning-tree-superinfer.env
# benchmark/edit/test
bash stop.sh
```

The maintenance hold prevents automatic recovery from racing with manual work.

## Health Checks

- `/health`: engine/API health.
- `/v1/models`: API/model routing.
- `/metrics`: KV, transfer, queue, capacity, and topology telemetry.
- Process tree: API, EngineCore, workers, resource tracker.
- GPU state: `nvidia-smi` and container CUDA check.

## Logging

Server logs:

```text
logs/direct_reasoning_superinfer_8202_*.log
```

Sidecars:

```text
*.log.cmd
*.log.profile
```

User-service logs:

```bash
journalctl --user -u deepseek-vllm.service -f
```

Failure artifacts:

```text
failure_artifacts/incident_*
```

## Recovery

The supervisor detects missing process trees and repeated `/health` failures,
collects read-only evidence, calls scoped `stop.sh`, and exits nonzero. systemd
then restarts it subject to backoff and burst limits.

The container is treated as an existing dependency. The supervisor may start it
if stopped, but does not recreate or rebuild it.

## Source Boundary

This deployment changes user-level systemd state and repository-local state
only. It does not change the host kernel watchdog or cron configuration.

## Sources

- `docs/110_systemd_service_guide.md`.
- `systemd/deepseek-vllm.service`.
- `scripts/run_supervised.sh`.
- `scripts/deepseek-vllmctl`.
- `docs/090_launch_recipe.md:21-105`.
