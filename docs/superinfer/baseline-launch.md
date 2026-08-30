# Baseline Launch Contract

The baseline must run inside `richard-base-dev-sysnice`.

For maximum-throughput service rather than a conservative baseline, launch the
high-risk profile:

```bash
cd /workspace/re-SuperInfer/vllm-superinfer-v4
scripts/launch_superinfer.sh .env.superinfer-serving
```

This is intentionally separate from `.env.superinfer-high-risk`, which retains
the older exhaustive benchmark settings. The serving profile uses the validated
128 GiB CPU tier and 24-sequence/32768-batched-token shape, enables proactive
high-risk DSpark rotation, and leaves DeepSeek block-first disabled. Use
`.env.superinfer-service128` as the normal-mode rollback profile.

Use the profile directly; no copying to `.env` is required:

```bash
scripts/launch_superinfer.sh .env.vanilla
```

## Container checks

```bash
docker exec richard-base-dev-sysnice nvidia-smi -L
docker exec richard-base-dev-sysnice ls -ld \
  /workspace/models/DeepSeek-V4-Flash-0731 \
  /workspace/models/DeepSeek-V4-Flash-DSpark
```

## Target environment checks

```bash
docker exec richard-base-dev-sysnice bash -lc '
  cd /workspace/re-SuperInfer/vllm-superinfer-v4
  .venv/bin/python -c "import vllm; print(vllm.__file__)"
  .venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
'
```

The target `.venv` is created with `uv --system-site-packages` so the container's
compatible Torch installation is reused. Do not install a replacement Torch
build without recording the reason and benchmark impact.

## DSpark launch shape

The exact speculative-model wiring must be verified against the checkpoint before
the first server run. The non-negotiable constraints are:

```text
model=/workspace/models/DeepSeek-V4-Flash-0731

No model is downloaded by this project. If package installation needs network
access, proxy variables may be exported in the shell only; credentials must not
be written to scripts, environment files, logs, or documentation.

## Reproducible baseline

From the host:

```bash
docker exec richard-base-dev-sysnice bash \
  /workspace/re-SuperInfer/vllm-superinfer-v4/docs/superinfer/dspark-baseline.sh
```

The baseline script intentionally omits all SuperInfer flags. It is the control
profile for saturation comparisons. The served model name is
`deepseek-ai/DeepSeek-V4-Flash-0731`; clients should use that exact name.
