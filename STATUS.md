# STATUS.md — physgate MVP build status

**Build date:** 2026-06-03 (autonomous build session)
**Machine:** RTX Pro 6000 Blackwell Max-Q (sm_120), driver 580.159.03, CUDA 12.8

## Component status

| # | Component | Status | Tests |
|---|---|---|---|
| A1 | `planner/schemas.py` — Plan/PlanStep contracts | ✅ complete | 8 |
| A2 | `gate/l1_kinematic.py` — Go2 joint-limit checks | ✅ complete | 12 |
| A3 | `gate/scoring.py` — physics scoring + best-of-N | ✅ complete | 11 |
| A4 | `world/scene_graph.py` — scene build/query/effects | ✅ complete | 12 |
| A5 | `orchestrator/graph.py` — LangGraph state machine | ✅ complete | 9 |
| B6 | PhysX libcuda workaround | ✅ pre-existing | — |
| B7 | PyTorch 2.7.0+cu128, sm_120 verified | ✅ complete | — |
| B8 | Isaac Sim 5.1.0 + Isaac Lab 2.3.2 (pip, py3.11/uv) | ✅ complete | — |
| B9 | Go2 headless smoke test (64 envs, rsl_rl) | ✅ passed | — |
| C10 | `gate/reset_workaround.py` — same-state parallel reset | ✅ complete | 2 (isaac) |
| C11 | `gate/l2_physics.py` — N-plan parallel physics rollout | ✅ complete | 4 (isaac) |
| C12 | `gate/parallel.py` — L1→L3→L2→scoring pipeline | ✅ complete | 11 |
| C13 | `world/usd_semantics.py` — USD stage → Scene | ✅ complete | 2 (isaac) |
| D14 | `planner/planner.py` — Claude planner + mock | ✅ complete | 13 |
| D15 | `planner/critic.py` — safety critic + mock | ✅ complete | 12 |
| D16 | `mcp_server/` — 3 MCP tools + FastMCP server | ✅ complete | 14 |
| E17 | `executor/` — plan executor + backends (mock & Isaac) | ✅ complete | 6 + 1 (isaac) |
| E18 | `examples/fetch_and_place.py` — end-to-end demo | ✅ complete | 3 |
| E19 | demo run proof | ✅ see below | — |

_(test counts approximate; run `pytest` for the authoritative numbers)_

## Test summary

- **Pure-logic suite** (any venv): `pytest` → **113 passed** (no GPU, no API key)
- **Isaac integration suite** (env_isaaclab venv): `pytest tests/test_isaac_sim_gate.py`
  → 9 tests covering C10/C11/C13/E17 against live Isaac Sim

## What is real vs. simplified (MVP scope)

| Aspect | MVP implementation | Production path |
|---|---|---|
| LLM planner/critic | Mock (deterministic) when no `ANTHROPIC_API_KEY`; real Claude client implemented and auto-selected when key present | set the key — no code change |
| Robot locomotion | Kinematic base driving (root pose writes); legs hold standing pose | learned locomotion policy (rsl_rl) + skill library |
| L2 physics validation | Box/object dynamics are fully physical (placement stability, contacts); robot path collisions are swept-geometry checks | contact-sensor based collision detection with policy-driven motion |
| Pick/place | Kinematic attach/detach + physical release dynamics | gripper articulation + grasp physics |
| Approval gate | Auto-approve in demo | LangGraph interrupt → human approval |
| Orchestrator state | MemorySaver (in-process) | PostgresSaver |
| Audit streams | Not implemented | Langfuse/OTEL + MCAP + Merkle |

## Demo runs

### Offline (mock planner, symbolic physics) — `python examples/fetch_and_place.py`

8 candidates → 6 survive critic → best-of-N selection → 4/4 steps executed →
**DONE**. (transcript: `benchmarks/demo_runs/demo_output_offline.txt`)

### Isaac Sim (real GPU physics) — `python examples/fetch_and_place.py --isaac`

**End-to-end PASS** (transcript: `benchmarks/demo_runs/demo_output_isaac.txt`):

- scene perceived from USD stage semantics (C13 in the loop),
- 8 candidates → 6 survivors → **Isaac Lab L2 parallel physics validation**
  in 8 identical envs (C10 + C11),
- the gate selected `mock_2_cautious` (detour route, **0 collisions**) over
  the faster direct routes (**1 collision** each, −1000 score penalty) —
  physics validation correctly traded speed for safety,
- the winning plan executed physically in Isaac Sim (SimBackend, E17):
  **5/5 steps, box ended resting on the shelf** → OUTCOME: DONE.

## Known issues

1. `isaaclab_mimic` is installed without its `robomimic` dependency (egl-probe
   does not build with CMake 4.x). Do not use mimic workflows in this env.
2. `pip check` reports version conflicts between isaacsim pins and
   wandb/wheel/fastapi — harmless for our usage (see DECISIONS.md D-007).
3. Isaac integration tests must run as a separate pytest invocation in the
   `env_isaaclab` venv; the Omniverse app launches at module import and lives
   for the whole process.
4. The kinematic-base approach means robot self-collision and gait stability
   are NOT validated — that requires the locomotion-policy upgrade.

## Next steps (recommended order)

1. Train a Go2 locomotion policy (rsl_rl, ~30 min on this GPU) and replace
   kinematic base driving with velocity-command tracking.
2. Wire the Claude API planner end to end (set `ANTHROPIC_API_KEY`) and measure
   plan quality vs the mock (Phase 0 benchmark #7: LLM latency P50/P95).
3. Phase 0 benchmarks #3-#5 (GPU utilization knee, L2 wall-clock, warm-start).
4. Approval gate UX (LangGraph interrupt instead of auto-approve).
5. Audit streams (Langfuse + MCAP + Merkle checkpoints).


---

# Milestone 2 — Policy locomotion, benchmarks, approval & audit (2026-06-03, second session)

## Goal completion

| # | Goal | Status |
|---|---|---|
| 1 | Train Go2 locomotion policy → replace kinematic base driving | ✅ **DONE** — rsl_rl, 300 iters, mean reward 35.0; integrated into L2 gate + executor; 11/11 Isaac tests pass |
| 2 | Real-LLM demo + planning latency (benchmark #7) | 🔄 **code complete, running autonomously** — OAuth wired in; blocked on shared subscription quota (see D-012); patient retry chain auto-commits results when quota recovers |
| 3 | Phase 0 benchmarks #3–#5 | ✅ **DONE** — results below |
| 4 | Approval UX (LangGraph interrupt) + audit streams | ✅ **DONE** — interrupt approval + three-stream Merkle audit, all tested |

## New components

| Component | What it does | Tests |
|---|---|---|
| `world/locomotion.py` | Go2PolicyController (exact 48-dim obs layout) + WaypointNavigator | 2 (isaac) |
| `gate/l2_physics.py::rollout_plans_with_policy` | N plans walk in N parallel envs; stuck/fall detection; placement physics | 2 (isaac) |
| `executor/sim_backend.py::PolicySimExecutor` | winning plan executed by the walking robot | — |
| `orchestrator/graph.py::HUMAN_APPROVAL` | LangGraph interrupt approval + resume_with_approval | 6 |
| `audit/` (merkle, records, trail) | three-stream audit + periodic Merkle checkpoints + tamper detection | 9 |
| `planner: ANTHROPIC_AUTH_TOKEN` | Claude subscription OAuth token support | 4 |
| benchmarks #3/#4/#5/#7 scripts | Phase 0 measurement suite | — |

**Test totals: 133 pure-logic + 11 Isaac integration = 144, all green.**

## Phase 0 benchmark results (RTX Pro 6000 Blackwell Max-Q, 300W)

### #3 — GPU saturation curve

| envs | env-steps/s | scaling efficiency |
|---|---|---|
| 1 | 281 | 1.00 |
| 16 | 4,360 | 0.97 |
| 64 | 16,237 | 0.90 |
| 256 | 64,249 | 0.89 |
| 1024 | 250,227 | **0.87** |

**The GPU scales near-linearly to ≥1024 envs.** N=8 best-of-N uses <1% of available
parallelism — there is enormous headroom for larger candidate sets.

### #4 — L2 validation wall-clock

| Mode | 1 plan | 8 plans (parallel) | marginal cost of +7 plans |
|---|---|---|---|
| Kinematic | 7.3 s | 20.2 s | 12.9 s |
| **Walking policy** | 17.1 s | 17.4 s | **0.32 s** |

**Headline result: with the walking policy, validating 8 candidates costs 1.9%
more than validating 1.** Best-of-N quality is effectively free in GPU time
(confirms architecture doc §6).

### #5 — Warm-start latency

App launch 3.8 s + 8-env scene 1.9 s + identical reset 0.1 s = **6.6 s total**
(the design assumed 10–30 s — better than expected).

### #7 — LLM planning latency (real Claude Opus 4.8, subscription OAuth via `claude -p`)

| Stage | P50 | P95 | Notes |
|---|---|---|---|
| Planner — 1 call returning 8 plans | 56.2 s | 56.2 s | 8/8 valid plans every trial |
| Planner — **8 parallel calls**, 1 plan each | **15.9 s** | 20.9 s | **3.5× faster** — validates the original design sketch |
| Safety critic | 14.6 s | 16.3 s | consistently prunes 2/8 reckless candidates |

3 trials per strategy; auth path = `ClaudeCodeHeadlessClient` (`claude -p`
subprocess, ~1-3 s CLI startup included in each call).

**Gate criterion check:** with parallel planning (15.9 s) + critic (14.6 s) +
L2 policy validation (17.4 s, benchmark #4), the LLM accounts for ~64% of
pipeline wall-clock — the architecture's 75-85% estimate was slightly
pessimistic, and parallel candidate generation is the clear win.

## What physics validation now proves (policy mode)

With the trained policy, L2 results are fully physical:
- **Cautious (detour) plans**: robot walks around the pillar, places the box on
  the shelf → SUCCESS (2/8 candidates feasible)
- **Direct (straight-line) plans**: robot physically blocked by the pillar →
  stuck detection → BLOCKED failure
- **Reckless plans**: pruned by the critic before reaching physics

The gate selects a cautious plan every time — physics-validated best-of-N
working end to end with real locomotion.
