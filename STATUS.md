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

> Historical (milestone 1) numbers. Current totals after milestone 3 (the
> architecture-correction rebuild) are in the milestone 3 section below.

- **Pure-logic suite** (any venv): `pytest` → 113 passed at milestone 1 (no GPU, no API key)
- **Isaac integration suite** (env_isaaclab venv): `pytest tests/test_isaac_sim_gate.py`
  → 9 tests at milestone 1, covering C10/C11/C13/E17 against live Isaac Sim

## What is real vs. simplified (MVP scope)

| Aspect | MVP implementation | Production path |
|---|---|---|
| LLM planner/critic | Mock (deterministic) when no `ANTHROPIC_API_KEY`; real Claude client implemented and auto-selected when key present | set the key — no code change |
| Robot locomotion | Kinematic base driving (root pose writes); legs hold standing pose | learned locomotion policy (rsl_rl) + skill library |
| L2 physics validation | Box/object dynamics are fully physical (placement stability, contacts); robot path collisions are swept-geometry checks | contact-sensor based collision detection with policy-driven motion |
| Pick/place | Kinematic attach/detach + physical release dynamics | gripper articulation + grasp physics |
| Approval gate | Auto-approve in demo _(milestone 2 added the LangGraph interrupt)_ | LangGraph interrupt → human approval |
| Orchestrator state | MemorySaver (in-process) | PostgresSaver |
| Audit streams | Not implemented at milestone 1 _(milestone 2 added the in-memory three-stream Merkle audit)_ | Langfuse/OTEL + MCAP back-ends |

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
| 1 | Train Go2 locomotion policy → replace kinematic base driving | ✅ **DONE** — rsl_rl, 300 iters, mean reward 35.0; integrated into L2 gate + executor; 12/12 Isaac tests pass |
| 2 | Real-LLM demo + planning latency (benchmark #7) | ✅ **DONE** — see "Real-LLM end-to-end demo" + benchmark #7 below |
| 3 | Phase 0 benchmarks #3–#5 | ✅ **DONE** — results below |
| 4 | Approval UX (LangGraph interrupt) + audit streams | ✅ **DONE** — interrupt approval + three-stream Merkle audit, all tested |

## New components

| Component | What it does | Tests |
|---|---|---|
| `world/locomotion.py` | Go2PolicyController (exact 48-dim obs layout) + WaypointNavigator | 2 (isaac) |
| `gate/l2_physics.py::rollout_plans_with_policy` | N plans walk in N parallel envs; stuck/fall detection; placement physics | 3 (isaac) |
| `executor/sim_backend.py::PolicySimExecutor` | winning plan executed by the walking robot | — |
| `orchestrator/graph.py::HUMAN_APPROVAL` | LangGraph interrupt approval + resume_with_approval | 6 |
| `audit/` (merkle, records, trail) | three-stream audit + periodic Merkle checkpoints + tamper detection | 9 |
| `planner/headless_client.py` | Claude subscription OAuth via `claude -p` headless mode (D-015) | 15 |
| `gate/l3_scene.py::check_step_targets` | anti-hallucination: step targets must exist in scene (D-016) | 5 |
| benchmarks #3/#4/#5/#7 scripts | Phase 0 measurement suite | — |

**Test totals: 153 pure-logic + 12 Isaac integration = 165, all green.**

## Real-LLM end-to-end demo (Claude Opus 4.8 + walking policy) ✅

Transcript: `benchmarks/demo_runs/demo_output_real_llm_isaac.txt`

- **ClaudePlanner** (via `claude -p`, subscription OAuth): 8 well-formed diverse
  plans, all referencing only real scene objects (prompt HARD CONSTRAINTS, D-016)
- **ClaudeCritic**: 7/8 survived
- **L2 physics (walking policy, 8 parallel envs)**: only **1/7 physically
  feasible** — `plan_waypoint_return_06` (pick box → detour via waypoint →
  shelf). _Milestone 3 note: this "discrimination" was the routing artifact —
  the 6 "infeasible" plans were well-formed decompositions that the
  straight-line low level could not execute (REBUILD.md). After the rebuild,
  8/8 real Claude plans are feasible._
- **Execution**: walking robot ran the winning plan, **5/5 steps, box on shelf**
- **OUTCOME: DONE**; Merkle audit integrity OK

First attempt escalated (all plans rejected) — root cause was LLM-hallucinated
object ids passing through every validation layer; fixed with defense-in-depth
validation (D-016). The escalation itself was correct fail-safe behaviour.

## Phase 0 benchmark results (RTX Pro 6000 Blackwell Max-Q, 300W)

**All 8 Phase 0 benchmarks are now complete** (#1 hardware gate in milestone 1;
#2–#8 below and in `benchmarks/phase0/results/`). Every go/no-go criterion
passed — the design assumptions of the architecture doc hold on this hardware.

### #2 — Same-state parallel reset + rollout determinism (bug #2133)

| envs | reset identity dev | cross-cycle drift | rollout determinism |
|---|---|---|---|
| 8 | 0.388 mm | 0.000 mm | verdicts + collision counts identical across runs (max time diff 0.12 s) |
| 64 | 0.948 mm | 0.000 mm | — |

**Verdict: GO.** All envs reset to the same state within the 1 mm tolerance,
resets do not drift, and identical plan batches produce identical L2 verdicts —
best-of-N comparisons are valid and reproducible. Note: identity deviation
grows with env count (0.39 mm at 8 → 0.95 mm at 64); re-check if N is ever
pushed past 64.

### #6 — Warp (Newton foundation) JIT + kernel cache on sm_120

Newton itself is not in the current stack (Isaac Lab 2.3.2 = PhysX), so this
measures **Warp 1.14** — the kernel framework Newton is built on:

- Cold JIT of a representative sim-kernel set: **0.53 s** (go/no-go line: 1 h)
- Kernel cache **persists across processes** (cached load <0.01 s, below timing
  resolution)

**Verdict: GO.** Newton/Warp dev iteration on Blackwell is not blocked. Re-run
against Newton itself when it enters the stack.

### #3 — GPU saturation curve

| envs | env-steps/s | scaling efficiency |
|---|---|---|
| 1 | 281 | 1.00 |
| 16 | 4,360 | 0.97 |
| 64 | 16,237 | 0.90 |
| 256 | 64,249 | 0.89 |
| 1024 | 250,227 | **0.87** |

**Scaling efficiency declines monotonically (1.00 → 0.87) and the curve has NOT
reached a saturation knee within the measured range** — 1024 envs is the largest
measured point, not a saturation point. N=8 validation uses a small fraction of
the measured parallelism; how far the GPU scales beyond 1024 envs is unmeasured.

### #4 — L2 validation wall-clock

| Mode | 1 plan | 8 plans (parallel) | marginal cost of +7 plans |
|---|---|---|---|
| Kinematic | 7.3 s | 20.2 s | 12.9 s |
| **Walking policy** | 17.1 s | 17.4 s | **0.32 s** |

**Marginal cost of +7 candidates is small in policy mode (0.32 s), but L2
validation is not free end to end** — the ~17 s base rollout dominates the
wall-clock. _The "1/7 feasible / 2/8 feasible" discrimination this section
originally cited as the gate's value was the routing artifact (see milestone 3 /
REBUILD.md); the wall-clock measurements above remain valid._

### #5 — Warm-start latency

App launch 3.8 s + 8-env scene 1.9 s + identical reset 0.1 s (+ ~0.8 s untimed
inter-stage overhead) → **6.6 s total** (the design assumed 10–30 s — better
than expected).

### #8 — Cost/quality curve — **DEPRECATED (artifact, see REBUILD.md)**

> [!CAUTION]
> **This entire result is an artifact of a layering defect**, not a finding about
> LLM plan quality. Obstacle avoidance was missing from the low level
> (straight-line driver), so "feasible" meant "the LLM happened to route via the
> hand-placed waypoint_W". With deterministic navigation in the right layer
> (milestone 3), feasibility of well-formed plans is **~100%** and this curve
> vanishes. Kept for the historical record; details in
> `benchmarks/phase0/results/benchmark_8_cost_quality.md`. The replacement
> evaluation is the **orchestration suite** (`benchmarks/orchestration/`).

Pool of 24 real Claude plans; measured pool "feasibility" 17%; 5 samples per N:

| N | empirical success | analytical success | GPU wall (mean) | GPU s / candidate |
|---|---|---|---|---|
| 1 | 20% | 17% | 19.3 s | 19.3 s |
| 2 | 0%* | 31% | 16.7 s | 8.4 s |
| 4 | 80%* | 54% | 17.6 s | 4.4 s |
| 8 | 100%* | 83% | 28.6 s | 3.6 s |
| 16 | 100%* | 99% | 47.0 s | 2.9 s |

_\* all empirical rates have 1/trials granularity (5 trials per N) — sampling
noise; the analytical (hypergeometric) column is the better estimate._

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

## What physics validation proved at milestone 2 — and why it was wrong

> [!CAUTION]
> The milestone 2 framing below ("physics discriminates routes") was an
> **artifact**: route feasibility was a property of the broken low level
> (straight-line driver), not of the plans. Milestone 3 (REBUILD.md) corrected
> the layering — see the milestone 3 section for what the gate actually
> validates now (orchestration quality, not route geometry).

At milestone 2, L2 results discriminated:
- "Cautious (detour) plans" succeeded (2/8) — because they happened to route via
  the hand-placed waypoint_W,
- "Direct plans" got physically blocked — because the low level drove straight
  lines through the pillar instead of navigating around it,
- Reckless plans were pruned by the critic (this part still holds).


---

# Milestone 3 — Architecture-correction rebuild (2026-06-03, REBUILD.md)

Four adversarial code reviews concluded the milestone 2 headline ("17% of LLM
plans feasible, best-of-N rescues it to 99%") was an **artifact**: obstacle
avoidance lived in the wrong layer (the LLM had no coordinates; the low level
drove straight lines), so "feasibility" measured whether the LLM happened to
name the hand-placed rescue waypoint. This milestone rebuilt the layering per
[`docs/design/REBUILD.md`](docs/design/REBUILD.md).

## Goal completion

| Phase | Goal | Status |
|---|---|---|
| 1 | Remove waypoint_W + straight-line driving → deterministic A* navigation (Nav2-compatible boundary) | ✅ **DONE** — commit `50c05ce` |
| 2 | Fix the missed `release_boxes` (placement momentum) + reset identity + locomotion envelope | ✅ **DONE** — this commit's parent |
| 3 | Re-design evaluation around agent orchestration (not navigation geometry) | ✅ **DONE** — `physgate/eval/` + `benchmarks/orchestration/` |
| 4 | Fix reproducibility (`pip install -e ".[dev]"`), README/STATUS contradictions, benchmark #3/#8 annotations | ✅ **DONE** — this commit |

## Architecture after the rebuild (dual-system)

- **HIGH (LLM agent)**: semantic task decomposition only — pick/place ordering,
  preconditions, recovery. Never sees coordinates, never plans routes.
- **LOW (deterministic)**: A* occupancy-grid navigation
  (`physgate/nav/path_planner.py`, Nav2-swappable interface) + trained Go2
  walking policy. **Always routes around obstacles, for every plan.**
- **Sim-Gate**: validates orchestration quality (the thing that can actually be
  wrong), not route geometry (which the low level guarantees).

## Headline results

### Feasibility — the artifact is eliminated

| | pre-rebuild (milestone 2) | post-rebuild (milestone 3) |
|---|---|---|
| Mock planner survivors | 2/8 (25%) | **6/6 (100%)** |
| Real Claude survivors | 1/7 (14%) | **8/8 (100%)** |

Well-formed pick→carry→place decompositions are ~always physically feasible.
Feasibility no longer discriminates plans — and that is the *correct* outcome:
route feasibility was never a property of the plans.
(`benchmarks/rebuild/results/feasibility_after_rebuild.json`)

### Orchestration evaluation — what the gate now measures

7 scenarios (ordering / preconditions / recovery / multi-step / infeasible),
real LangGraph orchestrator, fault injection
(`benchmarks/orchestration/results/orchestration_eval.json`):

| Metric | Mock planner | Real Claude |
|---|---|---|
| End-to-end success (feasible tasks) | 0.60 | **1.00** |
| Infeasible recognition | 1.00 | 1.00 |
| Recovery from transient failures | 1.00 | 1.00 |
| Decomposition validity | 1.00 | 1.00 |
| Invalid-plan catch rate | 1.00 | 1.00 |
| **Orchestrator score** | **0.92** | **1.00** |

The eval differentiates orchestrators on the dimension that matters: the mock
fails exactly its known weaknesses (occupied-gripper precondition, multi-step
tasks); Claude handles both.

_Data-integrity note: an earlier run scored Claude 0.96 because the
decomposition checker assumed an empty initial gripper (D-021) — the data
plausibility review caught that Claude's "set the held box down first" plan
completed the task yet was scored invalid. With the checker fixed and the suite
re-run, the ONLY field that changed between runs was that one flag (verified by
per-scenario diff); every other outcome was identical._

## Key findings logged this milestone

- **D-018**: placement momentum transfer; carry is bookkeeping (+0.6 m
  overhead); reset identity requires actuator-target resets; locomotion
  envelope is [0.4, 0.6] m/s.
- **D-019**: relation vocabulary is part of the tool contract — when an eval
  says "the LLM is worse than a trivial baseline", first suspect the eval's
  interface.
- **D-020**: the trained policy cannot turn in place (a "keep standing" fixed
  point); the navigator never commands pure rotation (TURN_CREEP_SPEED).
- **D-021**: decomposition checking must start from the scenario's actual
  initial state — the empty-gripper assumption scored a correct plan as
  invalid (found by the data plausibility review).
- **D-022**: navigation infeasibility (unreachable goal) surfaces as a clean
  `infeasible_navigation` gate verdict and a failed executor action — never a
  crash. Completes the REBUILD Phase 3 failure-mode coverage.

## Test summary (current)

- **Pure-logic suite**: `pytest` → **216 passed** (orchestration eval, viz
  encoder/renderer, README generator)
- **Isaac integration suite**: `pytest tests/test_isaac_sim_gate.py` → **19 passed**
  (world-reuse determinism separately verified by ×2 consecutive full-suite runs)
- **Reproducibility**: fresh venv + `pip install -e ".[dev]"` + `pytest` → green
  (mcp + anthropic now in `[dev]`; anthropic imported lazily)
- **Dynamic recordings**: `benchmarks/rebuild/record_rollout.py` captures the
  Isaac camera + trajectory log from a real validation rollout, runs automated
  plausibility checks, and regenerates `docs/media/` (embedded in the README)

## What the project is now about

> Validating an **agent orchestrator's** decomposition / precondition /
> recovery / infeasibility-recognition ability against real physics — not
> rescuing a broken low level with best-of-N sampling. The pretty 99%
> best-of-N curve is gone because the problem it "solved" should never have
> existed.
