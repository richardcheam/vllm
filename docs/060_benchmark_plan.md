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
