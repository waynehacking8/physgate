# Phase 0 benchmark #8 — cost/quality curve

Plan pool: 24 real Claude plans · feasibility rate 17% (4/24 tested) · 5 trials per N · 2026-06-03 15:23:24

| N | empirical success | analytical success | GPU wall (mean) | GPU s / candidate |
|---|---|---|---|---|
| 1 | 20% | 17% | 19.34 s | 19.34 s |
| 2 | 0% | 31% | 16.73 s | 8.36 s |
| 4 | 80% | 54% | 17.62 s | 4.4 s |
| 8 | 100% | 83% | 28.56 s | 3.57 s |
| 16 | 100% | 99% | 47.02 s | 2.94 s |

Best-of-N success rate climbs steeply with N (analytical 17% at N=1 -> 99% at N=16) while GPU wall-clock grows sub-linearly (19.3s -> 47.0s: 2.4x the time for 16x the candidates; per-candidate cost drops 6.6x). The gate buys plan QUALITY — feasibility discrimination — at sub-linear GPU cost. This is the cost/quality curve architecture doc §6 names as the honest deliverable. Note: empirical rates have 1/trials granularity; the analytical (hypergeometric) curve is the better estimate.
