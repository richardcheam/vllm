# Successful SuperInfer Baseline

The generated baseline report is stored with the benchmark artifact directory:

```text
/workspace/re-SuperInfer/benchmark_artifacts/vllm-superinfer-v4/20260803_182055_superinfer/
```

Generate or refresh the report with:

```bash
python scripts/report_superinfer_bench.py \
  /workspace/re-SuperInfer/benchmark_artifacts/vllm-superinfer-v4/20260803_182055_superinfer \
  --output-dir /workspace/re-SuperInfer/benchmark_artifacts/vllm-superinfer-v4/20260803_182055_superinfer/report
```

The result is a successful SuperInfer baseline, not yet a matched vanilla
comparison or final saturation claim.
