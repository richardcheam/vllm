# Claims And Evidence Table

| Presentation claim | Status | Evidence | Safe wording |
|---|---|---|---|
| CPU-KV D2H stores work | VERIFIED | Accepted pressure store events/bytes; `docs/100_reasoning_tree_offload_results.md:37-48` | “The validated profile performs real GPU-to-CPU KV stores.” |
| CPU-KV H2D reloads work | VERIFIED | Reload/pressure load events and exact output gate | “The validated profile restores CPU-resident KV with deterministic output.” |
| 524K prompt/10K output survives pressure | VERIFIED | `docs/100_reasoning_tree_offload_results.md:13-29,63-66` | “This aggressive profile completed 48/48 requests.” |
| Full GPU KV pressure is tolerated | VERIFIED for profile | 100% GPU KV, zero pending transfers | “Observed for the documented profile.” |
| Allocator/free-list pressure fixes are effective | VERIFIED for profiles | Core tests plus accepted pressure | “Validated against the exercised failure paths.” |
| Transfer quiescence works | VERIFIED for profiles | Post-fix quiescence gate and soak | “The tested engine drains pending transfer work.” |
| NUMA topology is discovered | VERIFIED | Logs/metrics and topology code | “GPU-to-NUMA topology discovery and worker binding work.” |
| CPU KV pages are physically NUMA-local | NOT YET VERIFIED | No complete page-placement study | Do not claim physical locality. |
| GH200 topology tuning improves speed | NOT YET VERIFIED | No controlled local/remote causal comparison | “Logical locality cost modeling is implemented.” |
| Native DMA is faster | NOT YET VERIFIED | Newer track has non-matched directional probes | “Native DMA correctness is shown; performance is directional.” |
| Full SuperInfer parity | NOT YET VERIFIED | Native DuplexKV and full RotaSched/LVF absent | “Selected SuperInfer capabilities were forward-ported.” |
| DSpark high-risk rotation is production-safe | NOT YET VERIFIED | Rejection/rollback stress incomplete | “DSpark/high-risk is a newer research track.” |
| Startup optimization is solved | NOT YET VERIFIED | Cache persistence/AOT reuse and phase table incomplete | “Startup instrumentation and candidates exist; optimization remains open.” |
| The old server shutdown was an EngineCore crash | NOT PROVEN | September 7 log shows parent exit; no EngineCore error | “The API parent disappeared; external cause is unresolved.” |
| `resource_tracker` warnings prove a memory leak | NOT PROVEN | Warnings occur after shutdown; `/dev/shm` snapshots needed | “They are IPC bookkeeping warnings until growth is demonstrated.” |
| Production lifecycle is repeatable | FUNCTIONAL | start/stop/diagnostics/systemd design | “A scoped, inspectable lifecycle is implemented.” |
| Throughput improved by a fixed percentage | NOT CLAIMED | No matched causal baseline in primary tree | Do not present a speedup percentage. |
