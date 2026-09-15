# Presentation Data Gaps

## Gap: Controlled Baseline Comparison

**Why needed:** A performance slide needs a matched vanilla/current-vLLM
baseline.

**Can repository answer it?** Not conclusively for the accepted DS4 pressure
profile.

**Required experiment:** Same hardware, prompt/output lengths, concurrency,
request count, KV dtype, prefix-cache setting, success criteria, and transfer
volume where relevant.

**Priority:** P0 before any speedup claim.

## Gap: Physical NUMA Page Placement

**Why needed:** Distinguish logical topology planning from actual CPU-KV page
locality.

**Required evidence:** rank PID mapping, `/proc/<pid>/numa_maps`, `smaps_rollup`,
`numastat -p`, and measured local/remote allocation behavior.

**Priority:** P0.

## Gap: Copy/Compute Overlap

**Why needed:** Bandwidth alone does not show whether movement is visible to users.

**Required evidence:** CUDA event timelines, D2H/H2D service time, compute
intervals, scheduler-visible wait, and derived overlap ratio.

**Priority:** P0.

## Gap: Scheduler-Visible Transfer Stall

**Why needed:** Explain TTFT/TPOT effects under restoration pressure.

**Required evidence:** enqueue wait, completion wait, request blocked time,
capacity wait, and per-phase transfer age.

**Priority:** P1.

## Gap: DSpark High-Risk Rollback

**Why needed:** The newer v0.26 integration advertises a performance-first
high-risk profile, but rejection/rollback under sustained proactive rotation is
not complete.

**Required evidence:** accepted/rejected draft counters, dirty-tail lifecycle,
rollback correctness, output equality, repeated pressure, and quiescence.

**Priority:** P1 for the newer-track narrative.

## Gap: Startup Timing

**Why needed:** The project is beginning startup optimization, but existing spans
are not a controlled cold/warm/second-warm table.

**Required evidence:** phase markers, model load, graph capture, KV profiling,
CPU-KV allocation, cache state, peak/final memory, and repeated timings.

**Priority:** P1.

## Gap: Persistent Compile Cache

**Why needed:** Determine whether the container cache survives recreation and
whether the active DeepSeek path reuses it.

**Required evidence:** cache root/filesystem, before/after recreation, compile
artifact fingerprints, startup phases, and unchanged steady-state results.

**Priority:** P1.

## Gap: External Shutdown Attribution

**Why needed:** Server-local evidence cannot identify an external `SIGKILL`
sender.

**Can repository answer it?** No; this requires host audit/process accounting.

**Presentation wording:** State that the server recorded parent disappearance,
while exact external attribution remains unavailable without host-level auditing.

**Priority:** P2; do not modify host settings without separate approval.

## Gap: Broader Model/Workload Generality

The accepted results are for DeepSeek-V4-Flash-0731 and documented profiles.
They do not prove multi-node behavior, arbitrary agentic traffic, native DSpark
parity, or all context/output combinations.

**Priority:** P2.

## Gap: Clean Provenance Separation

Both repositories contain dirty working trees and untracked artifacts. The final
presentation should cite the working-tree docs/logs and committed history
separately rather than implying every claim belongs to one clean commit.

**Priority:** P1 for final slide footnotes.
