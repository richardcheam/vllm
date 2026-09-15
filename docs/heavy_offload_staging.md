# DeepSeek Offload Validation

This guide covers validation of the production-shaped reasoning-tree service.
The clients do not start, stop, or reconfigure vLLM. Run them only during an
approved maintenance window because the long-context workload is intentionally
heavy.

## Start The Service

Use the NUMA-capable sysnice container:

```bash
CONTAINER_NAME=richard-base-dev-sysnice PORT=8202 \
  bash /home/az04297/re-SuperInfer/vllm-modern-reasoning/start.sh
```

The effective service profile is documented in
`docs/100_reasoning_tree_offload_results.md`. In summary, it uses TP=2, FP8
GPU KV, 128 GiB total CPU KV, 24 server sequences, 32768 batched tokens, a
2400-block proactive budget, NUMA nodes 0 and 1, and DeepSeek-safe GPU-derived
CPU KV layout.

Check readiness before validation:

```bash
docker exec richard-base-dev-sysnice bash -lc \
  'curl --noproxy "*" -fsS http://127.0.0.1:8202/health'
```

The startup script sends three short sequential warmup requests after health
readiness and reports `READY` only when they succeed. These operational
warmups prime the tokenizer, model request path, CUDA execution, and short
decode path for the first users without creating large-context KV pressure;
they are not part of the pressure request counts. If readiness or warmup fails,
`start.sh` invokes the scoped stop path to clean up the detached server.

## Pressure Validation

The pressure client is `scripts/run_heavy_offload_soak.py`. It loads the local
tokenizer, builds unique prompts with exact token lengths, samples `/metrics`
every two seconds, and requires successful requests, observed transfers, no
stale-capacity interval, drained transfers, and final server health.

Validated pressure command:

```bash
docker exec -i richard-base-dev-sysnice bash -lc '
  cd /workspace/re-SuperInfer/vllm-modern-reasoning
  .venv/bin/python scripts/run_heavy_offload_soak.py \
    --url http://127.0.0.1:8202 \
    --model deepseek-ai/DeepSeek-V4-Flash-0731 \
    --tokenizer-path /workspace/re-SuperInfer/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062 \
    --users 24 \
    --pressure-requests 24 \
    --pressure-rounds 5 \
    --pressure-prompt-tokens 327680 \
    --expected-pressure-prompt-tokens 327680 \
    --max-prompt-tokens 327680 \
    --max-tokens 256 \
    --request-timeout-seconds 2400 \
    --stage-timeout-seconds 3600 \
    --output-dir heavy_offload_artifacts/pressure_320k_soak5_timeout2400
'
```

The accepted five-round result was:

```text
120/120 successful
90 store events
10 load events
545088170880 store bytes
5053783680 load bytes
100% maximum GPU KV usage
0 stale-capacity seconds
0 pending transfers
server healthy after completion
```

The heaviest completed test used `524288` prompt tokens (`2^19`) and a
`10000`-token maximum output across two rounds. It completed `48/48` requests,
with 26 store events, 2 load events, `71981445888` store bytes,
`3014390016` load bytes, 100% maximum GPU KV usage, zero stale-capacity
seconds, zero pending transfers, and healthy final server state. Treat this as
an extreme maintenance-window test.

The pressure prompt is the actual tokenized prompt, not a numeric placeholder.
Each request contains a unique request identifier in its text prefix, then
deterministic filler is repeated until the exact requested token length is
reached. This minimizes prefix-cache reuse and creates new KV data for pressure.

Do not infer logical DeepSeek token capacity as
`kv_total_blocks * block_size`; the hybrid/grouped layout does not map raw block
counts one-to-one to logical tokens.

## Reload Validation

The reload client is `scripts/validate_offload_reload.py`. It is separate from
the pressure client so deterministic reload correctness is not confused with
direct transfer-counter evidence.

```bash
docker exec -i richard-base-dev-sysnice bash -lc '
  cd /workspace/re-SuperInfer/vllm-modern-reasoning
  .venv/bin/python scripts/validate_offload_reload.py \
    --url http://127.0.0.1:8202 \
    --model deepseek-ai/DeepSeek-V4-Flash-0731 \
    --tokenizer-path /workspace/re-SuperInfer/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062 \
    --prompt-tokens 131072 \
    --store-requests 4 \
    --evict-requests 8 \
    --repeats 3 \
    --store-concurrency 1 \
    --evict-concurrency 24 \
    --reload-concurrency 4 \
    --max-tokens 64 \
    --output-json heavy_offload_artifacts/reload_validation.json
'
```

The accepted reload result was:

```text
exact prompt length: 131072 tokens
output mismatches: 0
prompt-length mismatches: 0
request failures: 0
server healthy after completion: true
```

The reload client does not collect transfer counters. Use the pressure client's
`metrics.jsonl` and `summary.json` for store/load bytes, event counts, pending
transfer counts, and transfer age.

## Interpret The Metrics

| Signal | Meaning |
|---|---|
| `offload_store_events > 0` | GPU-to-CPU KV movement occurred |
| `offload_load_events > 0` | CPU-to-GPU KV movement occurred |
| `kv_cache_free_blocks_min = 0` | The logical GPU KV pool reached full pressure |
| `num_requests_waiting_by_reason{reason="queue"} > 0` | Ordinary scheduler queueing |
| `num_requests_waiting_by_reason{reason="capacity"} > 0` | Full-sequence KV admission wait |
| `capacity > 0` and `running = 0` for more than 30 seconds | Stale-capacity candidate |
| `offload_pending_transfer_age_ms` keeps increasing | Transfer backlog or starvation |
| `/health` fails or workers exit | Engine/process failure |

An ordinary queue is expected when client concurrency reaches or exceeds the
server's `max_num_seqs=24`. It is not itself evidence of stale capacity.

## Stop And Cleanup

Use the matching container and port:

```bash
CONTAINER_NAME=richard-base-dev-sysnice PORT=8202 \
  bash /home/az04297/re-SuperInfer/vllm-modern-reasoning/stop.sh
```

The stop script uses a two-phase graceful shutdown:

1. It sends `SIGTERM` to the API parent first, allowing vLLM's own shutdown
   path to close the engine and synchronize process-local distributed, CUDA,
   shared-memory, and CPU-KV resources. This API-parent grace period defaults to
   `GRACEFUL_API_TIMEOUT_S=30` seconds.
2. It then sends `SIGTERM` to every captured descendant, including orphaned
   `VLLM::Worker_*` and `VLLM::EngineCore` processes that have the reasoning-tree
   working directory. The remaining process tree gets up to
   `SHUTDOWN_TIMEOUT_S=120` seconds before `SIGKILL` is used as a last resort.

The script intentionally does not wait for transfer counters from the shell.
The worker-side CPU-KV shutdown path synchronizes in-flight CUDA copy events;
an external transfer wait would duplicate that lifecycle and could hang during
an already-failed engine shutdown. The current detached Docker launcher also
does not create PID/PGID state files, so cleanup is scoped using the container,
serving port, process command, and reasoning-tree working directory rather than
blindly reusing state files from another launcher.

## Evidence And Limitations

See `docs/100_reasoning_tree_offload_results.md` for the complete configuration,
metrics, failure chronology, and claim boundary. The accepted results validate
CPU-KV offload stability for this profile. They do not prove native DSpark,
reference empty CPU allocation, native block-first layout, or performance gains
against vanilla vLLM.
