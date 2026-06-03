# physgate — Architecture Correction & Rebuild Plan

**Status: AUTHORITATIVE. Supersedes the best-of-N framing in `architecture.md` §6 and
the benchmark #8 conclusions.** Date: 2026-06-03. Source: four independent adversarial
code-review passes over this repo.

---

## 1. What went wrong (verified with file:line)

The headline result — "real-LLM plans are only ~17% physically feasible, and best-of-N
selection with a physics gate raises task success to 99%" — **is an artifact of a layering
defect, not a finding about LLM planning quality.** Obstacle avoidance was placed in the
wrong layer.

Evidence:
- `physgate/gate/schemas.py:63-69` — `SceneObject` carries **no coordinates**.
- `physgate/world/scene_graph.py:92-103` — `to_query_scene_payload` sends the LLM only
  ids / labels / affordances / relations. **No obstacle position reaches the planner.**
- `physgate/planner/planner.py:103-108` — the LLM may only emit `move_to_pose(target=<id>)`
  and is forbidden from inventing waypoints.
- `physgate/gate/l2_physics.py:421-431` + `physgate/world/locomotion.py:116-144` — the low
  level is a **straight-line point-to-point driver**. No path planner, no avoidance.
- `physgate/world/fetch_scene.py:34-43` — `obstacle_P` and the rescue marker `waypoint_W`
  are **hand-placed** so that only the `box → waypoint_W → shelf` ordering clears the pillar.

Consequence: of 24 pooled plans, the feasible ones are **exactly** the 4 that route through
`waypoint_W`. "17% feasible" measures *whether the LLM happened to name the hand-placed
waypoint*, not plan quality. **With a deterministic path planner, navigation would route
around the pillar regardless, the 17% would approach 100%, and the entire best-of-N rescue
curve would vanish.**

Also a real bug (HIGH): `physgate/gate/l2_physics.py:595-618` — `rollout_plans_with_policy`'s
`place` handler never calls `release_boxes`, so the box drops with zero momentum. The claim
that the gate penalises "fast, sloppy" placements does not hold in policy mode.

## 2. The correct architecture (dual-system; non-negotiable)

```
HIGH level — Agent (LLM):  task decomposition, skill ordering, precondition reasoning,
                           failure replanning. Emits SEMANTIC skills:
                           navigate_to(<object id>) / pick / place.
                           Does NOT do obstacle avoidance. Needs no coordinates.
                                   │  semantic skills
LOW level — DETERMINISTIC (Nav2):  navigate_to(X) → Nav2 global path planner over a
                                   costmap/occupancy grid → always routes around obstacles,
                                   reproducible. Avoidance is GUARANTEED here, not guessed.
                                   │
Sim-Gate — REPOSITIONED as agent-orchestrator EVALUATION (not a best-of-N physics filter):
   validates what the deterministic layer cannot guarantee AND the agent's own quality —
   task decomposition correctness, ordering, preconditions, recovery, recognising
   genuinely-infeasible requests. NOT "did the straight line miss the pillar."
```

This returns the project to its actual purpose: **validating whether the agent orchestrator
is good** (decomposition / tool use / recovery), not "physics filters a weak LLM."

## 3. Rebuild plan

### Phase 1 — Remove the artifact, put Nav2 (or deterministic A*) as the low level
- Remove the `waypoint_W` workaround everywhere (scene, prompt, layout). It is a physics
  hack masquerading as a semantic object.
- Replace straight-line driving with deterministic path planning. Integration points:
  `l2_physics.py` `synthesize_base_trajectory` (the interpolation), `_compile_mission`
  (expand one `goto` into a planned waypoint sequence), `executor/sim_backend.py`
  `move_to_pose`. Build a `plan_path(start_xy, goal_xy, obstacles) -> list[waypoint]`
  boundary and call it from both the gate and the executor.
- If full ROS 2 Nav2 integration is too heavy, ship an equivalent **deterministic A*
  occupancy-grid planner** that always routes around obstacles, behind the same interface
  so Nav2 can drop in later. Record the trade-off in DECISIONS.md. **The straight-line
  driver must be removed regardless.**
- Verify: any reasonable `navigate_to(shelf_A)` plan now succeeds (feasibility ≈ 100%,
  not 17%) — navigation is no longer an LLM lottery.

### Phase 2 — Fix the real bug
- `l2_physics.py:595-618`: add the `release_boxes` momentum transfer to the policy-rollout
  `place` handler, matching the kinematic rollout, so placement quality is actually tested.

### Phase 3 — Redesign evaluation around ORCHESTRATION failure modes
Design a task suite whose failure modes are about orchestration, not navigation geometry:
- wrong ordering (place before pick; pick without being near the object)
- unmet preconditions (place on an occupied shelf; pick with a full gripper)
- recovery (a skill fails → does the orchestrator replan rather than give up?)
- multi-object / multi-step tasks where decomposition quality matters
- genuinely infeasible requests (ungraspable object; goal Nav2 also cannot reach) → does
  the agent recognise and report instead of trying forever?

Metrics become agent-orchestrator metrics: end-to-end task success across the suite,
decomposition correctness, fraction of invalid plans caught, fraction of failures
successfully replanned. The Sim-Gate validates plan logic + physical outcome; Nav2 handles
"how to get there." best-of-N, if retained, applies only where decomposition is genuinely
uncertain — it is no longer the headline.

### Phase 4 — Reproducibility & doc honesty (code-review findings)
- `pyproject.toml`: move `mcp` into `[dev]` (or change the quick-start to `[dev,mcp,llm]`)
  so `pip install -e ".[dev]" && pytest` actually passes. Verify it.
- Fix README test counts (148/11 → real numbers). Fix STATUS.md contradictions (audit
  implemented-vs-not, stale 113 count, missing sampling-noise asterisk on N=8/16 100%,
  ROS 2 shown in the diagram but not implemented → mark not-implemented).
- Mark benchmark #8 **DEPRECATED / artifact** (keep it, explain why). Change benchmark #3's
  "1024-env saturation / huge headroom" to "efficiency still declining, knee not reached."

## 4. Boundaries
- The straight-line driver MUST be removed — this is the core of the rebuild.
- Do not produce any further "best-of-N rescues a weak LLM" artifact narrative.
- If full Nav2 stalls, ship deterministic A* with the same interface and record it; do not
  block the whole rebuild on Nav2 packaging.

## 5. Honest note
The code quality itself is sound (≈85% substantive tests, immutability, Pydantic contracts,
honest sim-vs-symbolic disclosure). The problem was a single architectural error — obstacle
avoidance in the wrong layer — that propagated into a misleading headline. Fixing the layer
fixes the thesis.
