# Local Observability for Daily vLLM Operations

This setup runs locally only (loopback-bound ports) and provides:

- Prometheus: `127.0.0.1:19190` (uncommon local port)
- Grafana: `127.0.0.1:13300` (uncommon local port)
- Node Exporter (CPU/memory/pressure)
- Local file-based GPU collector (nvidia-smi snapshots)
- NUMA textfile metrics (`observability/node_exporter_textfile/numa.prom`)

## Start observability stack

```bash
docker compose -f observability/docker-compose.local.yml up -d
```

## Stop observability stack

```bash
docker compose -f observability/docker-compose.local.yml down
```

## Daily service scripts

- Start + warmup + collectors:
- `scripts/start_with_warmup_and_observability.sh`
- Queued start wrapper (waits for GPU free inside window):
  - `scripts/start_when_gpu_free.sh`
- Stop + summary report:
- `scripts/stop_with_daily_report.sh`

## Warm-up stages (A/B/C)

`scripts/start_with_warmup_and_observability.sh` runs three warm-up stages by default:

- Stage A (prefill-heavy): `users=24`, `requests=120`, `max_tokens=256`, `prompt_len=16384`
- Stage B (decode-heavy): `users=32`, `requests=200`, `max_tokens=1024`, `prompt_len=2048`
- Stage C (heavier load): `users=50`, `requests=300`, `max_tokens=1024`, `prompt_len=4096`

You can override with environment variables, for example:

```bash
STAGE_C_USERS=50 STAGE_C_REQUESTS=400 bash scripts/start_with_warmup_and_observability.sh
```

vLLM metrics scrape is pinned to `host.docker.internal:8202`.
Keep your vLLM service on port `8202` for this setup.

Artifacts are written under:

- `logs/daily_ops/<run_id>/`

Key files per run:

- `warmup_stdout.log`
- `vllm_metrics_snapshots.prom`
- `gpu_metrics.csv`
- `docker_stats.csv`
- `numa_snapshots.prom`
- `daily_summary.json`
- `daily_summary.txt`

## Cron example

```cron
# Queue start at 07:30; wait for free GPU inside 07:30-12:00 window
30 7 * * * /usr/bin/flock -n /tmp/vllm_start_queue.lock /usr/bin/env START_WINDOW=07:30 END_WINDOW=12:00 PORT=8202 /data1/home/az04297/re-SuperInfer/vllm-modern/scripts/start_when_gpu_free.sh >> /data1/home/az04297/re-SuperInfer/vllm-modern/logs/cron_start_queue.log 2>&1

# Queue start at 13:00; wait for free GPU inside 13:00-21:00 window
0 13 * * * /usr/bin/flock -n /tmp/vllm_start_queue.lock /usr/bin/env START_WINDOW=13:00 END_WINDOW=21:00 PORT=8202 /data1/home/az04297/re-SuperInfer/vllm-modern/scripts/start_when_gpu_free.sh >> /data1/home/az04297/re-SuperInfer/vllm-modern/logs/cron_start_queue.log 2>&1

# Stop + report at 12:00 and 21:00
0 12,21 * * * /usr/bin/flock -n /tmp/vllm_stop_report.lock /data1/home/az04297/re-SuperInfer/vllm-modern/scripts/stop_with_daily_report.sh >> /data1/home/az04297/re-SuperInfer/vllm-modern/logs/cron_stop_report.log 2>&1
```

## Validation checklist

1. Open Grafana at `http://127.0.0.1:13300`.
2. Confirm dashboard `vLLM Daily Ops (Local)` is present.
3. Confirm Prometheus target health at `http://127.0.0.1:19190/targets`.
4. Confirm NUMA metrics exist:
   - `cat observability/node_exporter_textfile/numa.prom`
5. Confirm daily summary generated after stop:
   - `logs/daily_ops/<run_id>/daily_summary.txt`
