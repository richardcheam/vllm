# Benchmark Plan

## Configurations

1. Vanilla vLLM `v0.20.1`.
2. Patched vLLM with SuperInfer flags parsed but no behavior.
3. Patched vLLM with telemetry/VLT but no active swapping.
4. Patched vLLM with debug/manual swapping.
5. Patched vLLM with active GH200 KV swapping.

## Workload Ramp

- 1 user
- 10 users
- 25 users
- 50 users
- 100 users only after smaller cases are stable

## Prompt Sizes

- 1K
- 4K
- 40K
- Larger contexts only after stable smaller runs

## Metrics

- Success rate
- Total requests
- Throughput
- TTFT mean/P50/P90/P99
- TBT mean/P50/P90/P99
- GPU memory
- CPU swap memory
- Number of swapped blocks
- H2D/D2H bytes, time, bandwidth
- Prefill/decode queue length
- Crashes/errors

## Failure Reduction Order

1. Disable MTP.
2. Disable prefix caching.
3. Reduce concurrency.
4. Reduce prompt length.
5. Disable CUDA graphs if necessary.
6. Test one request with manual swap.
7. Compare with vanilla.

## Reporting Rule

Do not claim performance improvements without benchmark data. If client streaming chunks are counted instead of tokenizer tokens, report them as `streaming_chunks_per_second`.

## 2026-06-08 Quick Pressure Sanity Run (OPT-125M)

Purpose: first pressure sanity check after transfer-path updates, using
throughput-only offline benchmarking before larger serving benchmarks.

- Runner: `vllm bench throughput`.
- Model: `facebook/opt-125m`.
- Shared settings:
  - `num_prompts=24`, `random_input_len=64`, `random_output_len=16`
  - `dtype=float16`, `enforce_eager=True`
  - `gpu_memory_utilization=0.2`, `max_model_len=128`
  - `max_num_seqs=4`, `num_gpu_blocks_override=16`
  - `swap_cpu_memory_gb=1.0`, `pin_memory_fix=True`, `swapper_block_first=True`
- Results:
  - Swap baseline (no proactive/VLT):
    - elapsed `2.566s`, `9.352 req/s`, `748.168 tok/s`
    - raw: `docs/benchmarking/superinfer_pressure_swap_baseline_opt125m_quick.json`
  - Swap + proactive/VLT (`proactive_swap_budget=4`, `vlt_beta_bandwidth=1.0`):
    - elapsed `2.947s`, `8.145 req/s`, `651.575 tok/s`
    - raw: `docs/benchmarking/superinfer_pressure_swap_proactive_vlt_opt125m_quick.json`

Notes:

- This is a quick single-run sanity pass, not a statistically stable benchmark.
- Earlier attempted settings (`max_model_len=512` with `num_gpu_blocks_override=8`)
  failed engine startup due to insufficient KV capacity; fixed by using
  `max_model_len=128`.
- Throughput here is expected to vary with pressure profile and run-to-run noise;
  use multi-run serving benchmarks before making performance claims.

## Lightweight Observability Runner (2026-06-09)

Use this for Docker serving runs when you want mechanism and hardware visibility
without standing up Prometheus/Grafana yet.

- Script:
  - `vllm-modern/scripts/run_lightweight_observability_benchmark.sh`
- What it captures during one benchmark run:
  - benchmark stdout + JSON summary (`test_concurrent.py`)
  - GPU metrics (`nvidia-smi` sampled every 1s)
  - GPU per-process memory snapshots (every 1s)
  - container stats (`docker stats --no-stream`, every 1s)
  - host VM stats (`vmstat 1`)
  - vLLM `/metrics` snapshots (every 5s)
- Output location:
  - `/data1/home/az04297/re-SuperInfer/benchmark_artifacts/<timestamp>_u<users>_r<requests>/`
- Default command:
  - `bash /data1/home/az04297/re-SuperInfer/vllm-modern/scripts/run_lightweight_observability_benchmark.sh`
- Example with explicit server log snapshotting:
  - `bash /data1/home/az04297/re-SuperInfer/vllm-modern/scripts/run_lightweight_observability_benchmark.sh --server-log /data1/home/az04297/re-SuperInfer/vllm-modern/superinfer_point3.log`

## Docker Compose Profiles (Local model path)

Use these compose/env pairs so vanilla and SuperInfer run from the same local
model copy under `/data1/home/az04297/re-SuperInfer`:

- Vanilla:
  - compose: `ds4/docker-compose.vanilla.local.yml`
  - env: `ds4/.env_vanilla_local`
- SuperInfer:
  - compose: `ds4/docker-compose.superinfer.yml`
  - env: `ds4/.env_superinfer`
  - image: `richard-vllm-modern-superinfer:v1`

Start commands:

- Vanilla:
  - `docker compose -f /data1/home/az04297/re-SuperInfer/ds4/docker-compose.vanilla.local.yml --env-file /data1/home/az04297/re-SuperInfer/ds4/.env_vanilla_local up -d`
- SuperInfer:
  - `docker compose -f /data1/home/az04297/re-SuperInfer/ds4/docker-compose.superinfer.yml --env-file /data1/home/az04297/re-SuperInfer/ds4/.env_superinfer up -d`

Stop commands:

- Vanilla:
  - `docker compose -f /data1/home/az04297/re-SuperInfer/ds4/docker-compose.vanilla.local.yml --env-file /data1/home/az04297/re-SuperInfer/ds4/.env_vanilla_local down`
- SuperInfer:
  - `docker compose -f /data1/home/az04297/re-SuperInfer/ds4/docker-compose.superinfer.yml --env-file /data1/home/az04297/re-SuperInfer/ds4/.env_superinfer down`
