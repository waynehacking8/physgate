# Phase 0 benchmark #7 — LLM planning latency

Model: `claude-opus-4-8` · N=8 candidates · 3 trials · 2026-06-03 14:16:54

| Stage | P50 | P95 |
|---|---|---|
| Planner (1 call, 8 plans) | 56.18 s | 56.21 s |
| Planner (8 parallel calls) | 15.92 s | 20.9 s |
| Safety critic | 14.64 s | 16.27 s |

**Gate criterion check:** the architecture estimates LLM planning at
10-50 s and 75-85% of pipeline wall-clock. Compare with the L2 physics
validation wall-clock from benchmark #4 to confirm the bottleneck.
