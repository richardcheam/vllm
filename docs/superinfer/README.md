# SuperInfer Topology-Aware Port

This directory records the forward-port of the SuperInfer topology-aware
runtime into vLLM `v0.26.0`.

## Target profile

- Hardware: two NVIDIA GH200 144 GB HBM3e GPUs
- Interconnect: NVLink `NV18`
- GPU 0 local CPU NUMA node: 0
- GPU 1 local CPU NUMA node: 1
- Model: `DeepSeek-V4-Flash-0731`
- Runtime: tensor parallel size 2, pipeline parallel size 1
- Speculative decoding: DSpark, initially five speculative tokens
- Primary objective: maximum stable throughput under sustained KV-cache pressure

Pipeline parallelism is intentionally excluded from the initial profile because
the vLLM 0.26.0 DSpark loader rejects it. The production `vllm-modern` tree is
not modified by this project.

## Documents

- `implementation-context.md`: source-of-truth hierarchy, scope, and decisions.
- `saturation-benchmark-plan.md`: pressure ladder, metrics, and acceptance gates.
- `feature-checklist.md`: implementation and validation checklist for each major
  SuperInfer mechanism.
- `baseline-launch.md`: environment and baseline launch contract.
- `validation-status.md`: completed checks and current blockers.
- `baseline-results.md`: successful baseline report-generation instructions.
- `performance-priority-roadmap.md`: ordered next work by expected impact and risk.
- `forward-port-plan.md`: staged v0.27.x feature adoption and upgrade gates.
- `developer-notes.md`: code map, runtime flow, performance reasoning, and
  debugging guide.

## Recommended Serving Profile

The current maximum-throughput serving profile is `.env.superinfer-serving`.
It intentionally enables `PROFILE=superinfer-high-risk` and
`SUPERINFER_HIGH_RISK_MODE=1`, because the deployment goal is maximum
performance rather than the conservative correctness boundary. It keeps
`SWAPPER_BLOCK_FIRST=0`: DeepSeek-V4/TP=2 block-first addressing is still not
validated, so high-risk refers to proactive DSpark rotation, not an unvalidated
cache layout.

The profile uses:

- 128 GiB total CPU KV capacity, approximately 64 GiB per TP rank.
- `CPU_KV_ALLOCATION_MODE=empty` to avoid startup zero-fill.
- Explicit NUMA binding to nodes `0,1` and GH200 topology tuning.
- `MAX_NUM_SEQS=24` and `MAX_NUM_BATCHED_TOKENS=32768`, the stable long-context
  shape used by the latest GPU gates.
- Default Python DMA, because the latest clean pressure result did not show a
  native end-to-end advantage. Native DMA remains available as an experiment.
- A short best-effort post-readiness warmup, not the exhaustive benchmark ladder.

Start the service inside `richard-base-dev-sysnice`:

```bash
cd /workspace/re-SuperInfer/vllm-superinfer-v4
scripts/launch_superinfer.sh .env.superinfer-serving
```

Inspect the resolved command without starting a server:

```bash
scripts/launch_superinfer.sh .env.superinfer-serving --dry-run
```

The launcher records the selected environment in `.run/server.profile` and
refuses to silently reuse a running server started with another or unknown
profile. Stop only the process group owned by this checkout with:

```bash
scripts/stop_superinfer.sh .env.superinfer-serving
```

Rollback to the validated normal-mode serving configuration by stopping the
high-risk service and launching `.env.superinfer-service128` instead. The
normal profile keeps the same 128 GiB CPU tier, NUMA binding, and topology
tuning but leaves high-risk rotation disabled.

High-risk serving remains an operational risk: the corrected exact-token reload
gate and basic serving smoke passed, but DSpark rejection/rollback under
sustained proactive rotation has not completed. Monitor output correctness,
`offload_residency_dirty_blocks`, queue depth, pending transfers, and server
errors; roll back if any diverge.

The current live serving smoke artifact is
`benchmark_artifacts/serving_smoke/20260825_000506_high_risk`. It completed
`8/8` exact-4096-token requests at `219.8` completion tokens/s. This is a
startup/health smoke result, not a long-context saturation claim.

For an end-to-end serving-style profile search, use the staged one-lifetime
soak runner:

```bash
cd /workspace/re-SuperInfer/vllm-superinfer-v4
bash scripts/run_serving_soak.sh \
  --service-env .env.superinfer-serving
```

It launches each candidate once, moves through short to long exact-token
traffic, performs repeated-prefix 131K reload, tests 16K recovery traffic,
and writes a ranked recommendation. Stop any existing service first; scheduler
settings are fixed at server startup.

## Runtime scripts

Pass one committed profile directly to each script inside
`richard-base-dev-sysnice`:

```bash
scripts/launch_superinfer.sh .env.vanilla
scripts/launch_superinfer.sh .env.native-offload
scripts/launch_superinfer.sh .env.superinfer
scripts/launch_superinfer.sh .env.superinfer-serving
```

The benchmark and stop scripts accept the same positional environment file:

```bash
scripts/run_superinfer_bench.sh .env.superinfer
scripts/stop_superinfer.sh .env.superinfer
scripts/run_profile_matrix.sh
```

`--env-file PATH` remains supported. A plain `.env` is still used when no file
argument is provided. The profile files are safe to commit because they contain
no credentials or tokens.

```bash
scripts/launch_superinfer.sh
scripts/run_superinfer_bench.sh
scripts/stop_superinfer.sh
```

Validate the resolved command without launching a server:

```bash
scripts/launch_superinfer.sh --dry-run
```

When `LAUNCH_WARMUP_ENABLED=1`, launch calls
`scripts/warmup_superinfer.sh` after `/health` becomes ready. The production
warmup runs a small robust latency canary followed by a short basic throughput
check, saving JSON, GPU, and Prometheus snapshots under
`benchmark_artifacts/launch_warmup/`. Warmup is best-effort: a failed or timed
out stage is recorded and launch still leaves the healthy server running. The
exhaustive `WARMUP_*` ladder remains benchmark-only.

The scripts use `.run/server.pid` and `.run/server.pgid` to stop only the
process group they started. They send `SIGTERM`, wait for cleanup, and use
`SIGKILL` only after the configured timeout.

`SWAP_CPU_MEMORY_GB` controls SuperInfer capacity. `KV_OFFLOADING_SIZE` is used
only for the `native-offload` comparison profile.
`CPU_KV_ALLOCATION_MODE` defaults to `zero` and is unchanged in the normal,
reload, and pressure profiles. The opt-in `.env.superinfer-cpu128` profile uses
`SWAP_CPU_MEMORY_GB=128` and `CPU_KV_ALLOCATION_MODE=empty` for a larger CPU KV
working set; it is an experiment, not a production default.

The production SuperInfer profile enables vLLM NUMA binding with
`NUMA_BIND=1` and `NUMA_BIND_NODES=0,1`. The launcher passes these through as
`--numa-bind --numa-bind-nodes 0 1`; vLLM binds TP workers to GPU-local NUMA
nodes and EngineCore to the shard node union. This requires `numactl` and a
container permission set that permits NUMA policy (typically `SYS_NICE`). If
binding is unavailable, vLLM's NUMA helper fails closed or falls back according
to its subprocess resolver.

`run_superinfer_bench.sh` runs an exhaustive unique-prefix warmup ladder by
default, including a final saturation soak. Set `WARMUP_ENABLED=0` to disable
it, or tune the comma-separated `WARMUP_*` arrays in `.env`.

`run_profile_matrix.sh` sequentially runs `.env.vanilla`,
`.env.native-offload`, `.env.superinfer`, and `.env.superinfer-high-risk`. It
launches one profile, waits for readiness, runs warmup and measurements,
generates a report, stops that profile, and only then starts the next one.
Custom profiles can be passed as positional arguments.

The committed profiles use the robust unique-prefix client for both warmup and
measurement. The matrix also saves `gpu_before_bench.csv` and
`gpu_after_bench.csv` for each profile.

`.env.superinfer-high-risk` remains the older exhaustive benchmark profile and
is reported separately. `.env.superinfer-serving` is the deliberate
performance-first high-risk service candidate; it enables aggressive DSpark
rotation while retaining the validated GPU-derived DeepSeek layout and the
normal-mode rollback profile.

Each new robust benchmark JSON now persists TTFT/TBT mean and percentile
summaries. The runner also saves per-run `.gpu.csv`, `.metrics.txt`, and
`.log-diagnostics.txt` snapshots. Older artifacts may not contain these fields;
their reports correctly mark unavailable metrics instead of reconstructing them.

The documents describe intended work and must be updated when behavior or
supported combinations change.

For repeated-prefix CPU reload validation, run inside the runtime container:

```bash
docker exec -w /workspace/re-SuperInfer/vllm-superinfer-v4 \
  richard-base-dev-sysnice bash -lc \
  './scripts/run_reload_validation.sh --restart .env.superinfer-reload'
```

The reload harness disables launch warmup, bypasses the container HTTP proxy for
readiness and metrics, and writes `validation.json`. Exact output equality alone
is insufficient because a request can recompute on GPU; the validation file
must show positive store/load event and byte deltas.

To automate the expensive reload/debug loop without repeating an unchanged
source revision, use the optimizer controller:

```bash
scripts/run_superinfer_optimizer.sh \
  --env-file .env.superinfer-reload \
  --max-attempts 0
```

It waits for GPU idle samples, executes one reload attempt, preserves
artifacts, stops on a valid H2D reload, and otherwise waits for a source
fingerprint change. See `docs/superinfer/optimizer.md`.

This project is performance-first, not a full SuperInfer parity port. The
forward-port plan distinguishes upstream accelerators from native parity work
such as C++/CUDA/ZMQ DuplexKV and complete RotaSched/LVF semantics.
