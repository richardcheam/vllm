# SuperInfer Connector (v0.30.0)

## Scope

This is an opt-in integration layer on top of the official vLLM `v0.30.0`
checkout. The official DeepSeek-V4.1-Flash container recipe remains the
untouched baseline. SuperInfer must not change vanilla behavior when no
connector is configured.

The validated v0.26 implementation and evidence remain in:

```text
/data1/home/az04297/re-SuperInfer/vllm-superinfer-v4
```

The official V4.1 baseline recipe is maintained separately in:

```text
/home/az04297/dsv4.1/deepseek-v41-gh200-harness
```

## Activation

The connector is selected through the existing V1 KV-transfer configuration:

```json
{
  "kv_connector": "SuperInferConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "schema_version": 1,
    "capacity_gb": 128,
    "copy_backend": "dma",
    "allocation": {"mode": "auto"},
    "topology": {"mode": "auto"},
    "policy": {"mode": "conservative", "proactive_budget": 0}
  }
}
```

An external package can use `kv_connector_module_path` without modifying the
vLLM factory. This checkout also registers `SuperInferConnector` lazily as a
thin wrapper over the v0.30 CPU offload connector while the implementation is
being ported.

The current v0.30 slice makes the following options effective:

- `capacity_gb` maps to the connector CPU capacity.
- `allocation.mode` maps to zero-filled or empty CPU allocation.
- `allocation.mode: auto` currently resolves to zero-filled allocation.
- `copy_backend` is currently restricted to `dma`; unsupported backends fail
  closed instead of being silently ignored.
- Only `policy.mode: conservative` and `proactive_budget: 0` are accepted;
  scheduler policy actions are not implemented in this v0.30 slice.
- `transfer_queue_depth` and `metrics` are rejected because the current v0.30
  implementation does not consume those options.
- `SuperInferConnectorStats` and connector Prometheus registration provide the
  typed telemetry surface. The v0.30 manager now populates interval store/load,
  lookup, pending-work, capacity, and boundary-store observations. Boundary
  counters are emitted as deltas from their cumulative internal values.
- Pure VLT score/ranking helpers are present for policy tests, but scheduler
  victim selection remains disabled.
- Generic GPU/CPU locality discovery is available through the connector
  capability snapshot. Complete discovery requires GPU inventory, NUMA nodes,
  and a parsed peer matrix. Auto mode can report partial data; required mode
  fails if locality data is incomplete. Island IDs are omitted for partial
  discovery, and discovery does not yet change CPU block allocation.

## Ownership

- `KVTransferConfig`: connector selection, role, module path, and extras.
- `CacheConfig`: generic GPU KV/offload capacity only.
- `SchedulerConfig`: generic scheduling limits and future movement budgets.
- Connector extras: transfer backend, allocation, topology, and policy.
- Environment variables: container, device visibility, NUMA permissions, and
  deployment-only settings.

High-risk rotation, block-first layout, and GH200-specific bandwidth hints are
not enabled by this wrapper. They remain experimental follow-up work.

## Baseline Contract

Compare SuperInfer only against the official V4.1 recipe with the same:

- image tag and digest;
- model revision;
- TP/PP/EP layout;
- tokenizer and request trace;
- warmup and page-cache handling;
- KV dtype and block size;
- output correctness checks;
- performance measurement window.

The baseline harness is outside this checkout at:

```text
/home/az04297/dsv4.1/deepseek-v41-gh200-harness
```

Do not compare its eager single-sequence UVA expert-offload result directly to
a multi-user SuperInfer KV-pressure benchmark without a matched workload.

## Current Status

The v0.30 connector wrapper currently validates and applies configuration while
reusing the existing CPU offload implementation. The next porting stages are:

1. Feed locality mapping into CPU block allocation after GPU validation.
2. Validate V4.1 model/cache layout and DeepSeek mHC warmup.
3. Add conservative SuperInfer policy only after telemetry/correctness passes.

No performance improvement claim is made by the wrapper alone.
