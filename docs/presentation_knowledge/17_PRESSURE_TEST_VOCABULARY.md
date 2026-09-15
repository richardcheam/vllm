# Pressure Test Vocabulary

This file defines exactly what the accepted pressure names mean. A “round” is
not a time duration; it is one sequential request wave.

## Harness Lifecycle

```text
wait for /health
      ↓
tiny bootstrap request
      ↓
wait for usable KV-block metric
      ↓
optional workload warmup stage
      ↓
pressure round 1
      ↓
pressure round 2
      ↓
...
      ↓
stop metrics sampler
      ↓
validate requests, transfers, capacity waits, quiescence, final health
```

The harness does not start, stop, or reconfigure the server. It sends requests
to the configured URL and samples `/metrics` every two seconds.

## Bootstrap Request

Before capacity discovery, the harness sends one tiny request:

- prompt: one block (`256` tokens by default);
- output: 1 token;
- request ID: `-1`;
- purpose: populate scheduler statistics so `vllm:kv_cache_total_blocks` is
  available.

The bootstrap request is not counted as a pressure request and is not part of
the reported pressure-round request count.

## Workload Warmup Stage

The harness has an optional `--warmup-requests` stage. The accepted production
pressure runs used:

```text
warmup_requests = 0
```

Therefore their request counts contain the bootstrap request plus pressure
rounds, not an additional workload-warmup wave. This is separate from the
three short API warmups performed by `start.sh` during server startup.

## One Pressure Round

The accepted 327K profile uses:

```text
users:              24
pressure requests:  24
prompt:             exactly 327,680 tokenizer tokens
maximum output:     256 tokens
```

The harness creates 24 unique prompts and sends them through a semaphore with a
maximum of 24 concurrent requests. A round completes when all 24 request tasks
return or the stage deadline is reached.

One accepted round:

```text
24/24 successful
18 store events
2 load events
100% GPU KV usage
0 stale-capacity seconds
0 pending transfers
healthy server
```

This is a full-pressure wave, but not yet a repeated soak.

## Two-Round Pressure Soak

```text
round 1: 24 requests
round 2: 24 requests
total:   48 requests
```

Rounds execute sequentially. Round 2 starts after round 1 returns; there is no
special sleep or reset between rounds. This tests whether the same server can
repeatedly reach pressure, move KV state, drain transfers, and continue serving.

Accepted result:

```text
48/48 successful
36 store events
4 load events
100% GPU KV usage
0 stale-capacity seconds
0 pending transfers
healthy server
```

This is the primary acceptance shape in the reasoning-tree results.

## Three-Round Run

```text
3 × 24 requests = 72 total requests
```

Accepted result:

```text
72/72 successful
53 store events
6 load events
100% GPU KV usage
0 stale-capacity seconds
0 pending transfers
healthy server
```

It provides more repeated-pressure evidence but does not introduce a new
workload shape.

## Five-Round Extended Soak

```text
5 × 24 requests = 120 total requests
```

Accepted configuration:

```text
prompt:          exactly 327,680 tokens
maximum output: 256 tokens
request timeout: 2,400 seconds
stage timeout:   3,600 seconds
```

Accepted result:

```text
120/120 successful
90 store events
10 load events
545,088,170,880 store bytes
5,053,783,680 load bytes
100% GPU KV usage
0 stale-capacity seconds
0 pending transfers
healthy server
```

Runtime was approximately 3,409 seconds, or 56.8 minutes. This is an extended
maintenance-window soak, not a smoke test or startup warmup.

An earlier five-round attempt used a 900-second request timeout and was manually
interrupted after client-side timeouts. It is not EngineCore-crash evidence and
is not acceptance evidence.

## Aggressive 524K/10K Test

This is a different workload shape, not merely a larger round count:

```text
prompt:             exactly 524,288 tokens
maximum output:     10,000 tokens
users:              24
requests per round: 24
rounds:             2
total requests:     48
```

Accepted result:

```text
48/48 successful
26 store events
2 load events
71,981,445,888 store bytes
3,014,390,016 load bytes
100% GPU KV usage
0 stale-capacity seconds
0 pending transfers
healthy server
```

Runtime was approximately 4,906 seconds, or 81.8 minutes. This is an extreme
long-context/large-output capacity test and should not be presented as a short
readiness or ordinary soak test.

## Why Unique Prompts Matter

Pressure prompts include request-specific text identifiers and deterministic
filler. This minimizes prefix-cache reuse and forces new KV data. Reload uses
repeated original token sequences after distinct eviction traffic so CPU-resident
prefixes can be found and loaded.

## Acceptance Vocabulary

| Term | Exact meaning |
|---|---|
| Bootstrap | One tiny request used to populate scheduler/KV metrics |
| Workload warmup | Optional harness stage before pressure; zero in accepted pressure runs |
| Pressure round | One wave of 24 pressure requests in the accepted profile |
| Two-round soak | Two sequential 24-request waves, 48 total |
| Three-round run | Three sequential waves, 72 total |
| Extended soak | Five sequential waves, 120 total, long timeout, approximately 56.8 minutes |
| Aggressive test | Two waves at 524K prompt / 10K output, approximately 81.8 minutes |
| Quiescence | Pending stores, loads, queue state, and transfer age return to the acceptance bound |

## Sources

- `scripts/run_heavy_offload_soak.py:332-572`.
- `scripts/run_heavy_offload_soak.py:575-668`.
- `scripts/validate_offload_reload.py:127-246,249-292`.
- `docs/heavy_offload_staging.md:37-90`.
- `docs/100_reasoning_tree_offload_results.md:13-66,221-250,415-461`.
