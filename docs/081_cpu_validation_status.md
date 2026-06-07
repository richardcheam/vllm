# SuperInfer CPU Validation Status

Date: 2026-06-06

This document summarizes the closed CPU-testable forward-port checkpoint. It is
not a production-readiness claim; GPU runtime validation remains blocked until
GPU resources are available.

## Current Readiness

- CPU-testable guarded SuperInfer scheduler/offload slice: complete for the
  current pre-GPU scope.
- Full SuperInfer modern parity: not complete.
- GPU/model-output validation: blocked by GPU availability.
- Next required phase: run the GPU validation checklist before any further
  scheduler/offload expansion.

## Implemented And CPU-Tested

- SuperInfer/GH200 CLI and config knobs parse and default to disabled.
- `swap_cpu_memory_gb` maps to `SimpleCPUOffloadConnector` with lazy offload.
- Debug single-request swap throttling is wired and tested.
- Lazy offload scan depth is controlled by `proactive_swap_budget`.
- Guarded scheduler-side proactive preemption is active only under strict gates.
- Proactive candidates include decode-tail and small decode-backlog requests.
- Unsafe proactive candidates are excluded: prefill chunks, speculative work,
  output placeholders, missing connector, eager offload mode, paused scheduler,
  no waiting work, sufficient free blocks, pending transfers, and shared/refcounted
  KV blocks (`ref_cnt > 1`).
- VLT helper is scheduler-active in guarded proactive victim selection when
  VLT/SLO knobs are enabled.
- VLT scheduler inputs include request timing hints and connector-estimated swap
  bandwidth.
- Request rotary metadata/state transitions are implemented and tested.
- Simple CPU offload scheduler store/load, duplicate handling, cleanup, lazy
  scan, telemetry, and partial GPU-prefix + CPU-load paths are tested.

## CPU Test Suites Recently Passing

- `tests/v1/core/test_vlt.py tests/v1/test_request.py tests/v1/kv_connector/unit/test_config.py -vv` -> 22 passed.
- `tests/v1/core/test_scheduler.py -k "test_proactive_swap or test_generation_timing or test_priority_scheduling_preemption or test_preempt_during_execution" -vv` -> 31 passed.
- `tests/v1/simple_kv_offload/test_scheduler.py -vv` -> 16 passed.
- `tests/engine/test_arg_utils.py -k "test_superinfer_cli_defaults_are_disabled or test_superinfer_noop_flags_from_cli" -vv` -> 2 passed.

## Still Blocked Or Pending

- Small-model vanilla-vs-modern generated-output comparison.
- Conservative swap-on GPU smoke test.
- VLT/SLO GPU smoke test.
- DeepSeek-V4-Flash readiness validation.
- Full LVF/RotaSched policy integration beyond the guarded proactive path.
- DuplexKV/native block-first/full-duplex transfer optimizations.
- Refcount-aware proactive offload of shared prefix-cache blocks.
- Performance benchmarking.

## Recommended Next Action When GPU Is Available

Run `docs/080_gpu_validation_checklist.md` before further broadening scheduler
behavior. Any GPU-only failure should be treated as a targeted runtime fix, not
as a reason to expand policy scope further.

## CPU Phase Closure

The CPU phase is intentionally stopped here. Additional CPU-only policy
expansion would increase runtime risk without proving CUDA KV movement,
generated-output correctness, or DeepSeek-V4 MLA cache safety. Resume directly
with GPU validation when hardware is available.
