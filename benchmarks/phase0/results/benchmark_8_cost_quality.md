# Phase 0 benchmark #8 — cost/quality curve

> [!CAUTION]
> **DEPRECATED — ARTIFACT (2026-06-03, [REBUILD.md](../../../docs/design/REBUILD.md)).**
> This result measures a **layering defect**, not LLM plan quality. Obstacle avoidance
> was missing from the low level (straight-line driver), so "feasible" meant *"the LLM
> happened to route via the hand-placed `waypoint_W`"* — the 4 feasible plans out of 24
> are exactly the 4 that named that waypoint. With deterministic A\* navigation in the
> right layer (Phase 1 of the rebuild), feasibility of well-formed plans is **~100%**
> ([feasibility_after_rebuild.json](../../rebuild/results/feasibility_after_rebuild.json):
> mock 6/6, real Claude 8/8) and this best-of-N rescue curve **vanishes**.
>
> Kept for the historical record only. The replacement evaluation is
> [benchmarks/orchestration/](../../orchestration/) — agent-orchestrator metrics
> (decomposition, preconditions, recovery, infeasibility recognition).

Plan pool: 24 real Claude plans · feasibility rate 17% (4/24 tested) · 5 trials per N · 2026-06-03 15:23:24

| N | empirical success | analytical success | GPU wall (mean) | GPU s / candidate |
|---|---|---|---|---|
| 1 | 20% | 17% | 19.34 s | 19.34 s |
| 2 | 0% | 31% | 16.73 s | 8.36 s |
| 4 | 80% | 54% | 17.62 s | 4.4 s |
| 8 | 100%* | 83% | 28.56 s | 3.57 s |
| 16 | 100%* | 99% | 47.02 s | 2.94 s |

_\* empirical rates have 1/trials granularity (5 trials per N) — sampling noise, not exact
100%; the analytical (hypergeometric) column is the better estimate._

Original (now superseded) interpretation: best-of-N success climbs steeply with N while GPU
wall-clock grows sub-linearly. **Why it is wrong:** the "infeasible" plans were not bad
plans — they were well-formed decompositions that the broken low level could not execute
because it drove straight lines through the obstacle. The curve measures how often the LLM
guessed the magic waypoint, which is not a property of anything the project should claim.
