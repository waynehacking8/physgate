# DECISIONS.md — autonomous build decisions log

Decisions made with "reasonable default + keep going" authority during the
autonomous MVP build (2026-06-03). Each entry: what was decided, why, and what
to revisit.

## D-001: PhysX libcuda.so workaround — already present, skipped

The symlink `/usr/lib/x86_64-linux-gnu/libcuda.so → libcuda.so.1` already exists
(created 2026-05-05, root-owned). No sudo needed. Step B6 skipped.

## D-002: ANTHROPIC_API_KEY not set → planner/critic run in mock mode

`ANTHROPIC_API_KEY` is not set in the environment. Per the build brief:

- `planner/planner.py` and `planner/critic.py` implement the **real Claude API
  client** (anthropic SDK, reads `ANTHROPIC_API_KEY`), but fall back to a
  deterministic **MockPlanner / MockCritic** when the key is absent.
- The end-to-end demo therefore proves the *pipeline* (plan → critique → gate →
  select → execute) with mock-generated candidate plans.
- **To run the demo with a real LLM: `export ANTHROPIC_API_KEY=sk-ant-...` and
  re-run `examples/fetch_and_place.py`.** No code change needed.

## D-003: Orchestrator checkpointing — MemorySaver instead of PostgresSaver

Architecture doc specifies LangGraph + PostgresSaver for the MVP. Standing up a
Postgres instance is orthogonal to proving the pipeline and adds an external
service dependency. Decision: use `langgraph.checkpoint.memory.MemorySaver` for
the MVP/demo; the graph code takes the checkpointer as a parameter so swapping
in PostgresSaver later is a one-line change at the call site.

## D-004: N = 8 candidate plans (fixed)

Per build brief — no saturation benchmarking to tune N. `DEFAULT_NUM_CANDIDATES = 8`.

## D-005: Python 3.11 via uv (no sudo available)

Isaac Sim 5.1.0 pip wheels are Python-3.11-only; this box has system Python
3.10 and no passwordless sudo (so no deadsnakes PPA). Used `uv python install 3.11`
+ `uv venv ~/env_isaaclab --python 3.11 --seed` instead. Two venvs now exist:

- `~/Desktop/physgate/.venv` (py3.10): pure-logic dev/tests
- `~/env_isaaclab` (py3.11): Isaac Sim 5.1.0 + Isaac Lab 2.3.2 + physgate;
  **this is the venv for anything touching the GPU/simulator**

## D-006: Isaac Lab install fixes (robomimic skipped)

`./isaaclab.sh --install` failed on two optional native builds:

1. `flatdict==4.0.1` (isaaclab core dep): its legacy setup.py needs
   `pkg_resources`, which **setuptools >= 81 removed**. Fix: `pip install
   "setuptools<81"` then `pip install flatdict==4.0.1 --no-build-isolation`.
2. `egl-probe` (robomimic dep, needed only by isaaclab_mimic imitation
   learning): CMakeLists incompatible with CMake 4.x. **Not fixed — robomimic
   is not needed for the Go2 velocity task.** isaaclab_mimic is installed but
   its robomimic extra is absent; do not use mimic workflows in this env.

Core isaaclab 0.54.2 then installed cleanly with `--no-build-isolation`.

## D-007: pip version-pin conflicts left as-is (isaacsim pins win)

isaacsim 5.1.0 pins `packaging==23.0`, `click==8.1.7`; other packages want
newer. These pins are mutually unsatisfiable; resolution: keep isaacsim's and
isaaclab's pins satisfied (packaging 23.0, click 8.1.7, starlette 0.49.1) and
accept `pip check` warnings from wandb/wheel/fastapi — those only matter for
features we don't use (wandb logging, wheel building, fastapi endpoints).

## D-008: B9 verification result

`Isaac-Velocity-Flat-Unitree-Go2-v0`, headless, 64 envs, 5 rsl_rl training
iterations: **PASSED** on first try. No PhysX CPU fallback (the libcuda.so
symlink workaround was already in place). Extension cache was pre-populated by
the `isaacsim[extscache]` pip extra, so first launch took seconds, not 10+ min.

## D-009: L2 physics = kinematic base + dynamic objects (no locomotion policy)

The MVP does not train a Go2 locomotion policy. Instead:

- the robot **base is kinematically driven** along the plan's path (root-pose
  writes each physics step); legs hold the standing configuration,
- **objects are fully dynamic**: the box is carried kinematically but released
  with the approach momentum at placement — GPU physics then decides whether it
  stays on the shelf (slow approach) or slides/bounces off (fast approach),
- robot-path collisions with static obstacles are **swept-geometry checks**
  (a kinematic body produces no contact response, so geometry is the honest check).

This gives real physics-based plan discrimination (placement stability, route
collisions) at zero training cost. The upgrade path is documented in STATUS.md:
train an rsl_rl locomotion policy and replace pose-writes with velocity commands.

## D-010: route diversity via waypoint detour

For physics validation to be able to *prefer* one plan over another, candidate
plans must differ physically. MockPlanner's "cautious" variant routes via the
`waypoint_W` marker (clearing the obstacle pillar); "direct"/"scan_first"
variants go straight (sweeping through the pillar's inflated footprint). The
obstacle pillar sits exactly on the straight line box → shelf. Manipulation
targets (the shelf) are NOT counted as path obstacles — approaching them is the
point; interaction quality is judged by placement dynamics instead.

## D-011: Isaac pytest pattern

Raw ``isaacsim.SimulationApp`` inside a pytest fixture silently kills the
process (kit parses pytest's argv). Isaac tests instead use the Isaac Lab
pattern: ``AppLauncher(headless=True).app`` at module import, no ``app.close()``.
Isaac tests live in ``tests/test_isaac_sim_gate.py`` (marked ``isaac``), are
skipped automatically outside the Isaac venv, and run as their own pytest
invocation inside it.

## D-012: LLM credentials — Claude subscription OAuth token supported

The operator provided a Claude subscription OAuth token (sk-ant-oat01-...)
instead of a standard API key. OAuth tokens authenticate with
`Authorization: Bearer` + the `anthropic-beta: oauth-2025-04-20` header.
`make_anthropic_client()` handles both credential types; `ANTHROPIC_AUTH_TOKEN`
is checked alongside `ANTHROPIC_API_KEY` everywhere. The token lives in
`~/.config/physgate/credentials.env` (mode 600, outside the repo). Note: the
subscription is shared with interactive Claude Code use, so 429 rate limits are
expected — all benchmark/demo code retries with exponential backoff.

## D-013: Policy locomotion parameters

The trained Go2 policy (rsl_rl, 300 iterations, mean reward 35.0) replaces
kinematic base driving. Integration choices:

- **Commanded speed floor 0.4 m/s** — the policy tracks sub-0.4 m/s commands
  poorly (creeps/stalls). Plan "caution" is expressed by the detour route, not
  by very low speeds.
- **Stuck detection** — an env that makes <0.15 m of progress in 10 s while a
  goto step is active is marked `blocked` and fails L2. This is how
  physically-blocked routes (straight line through the pillar) fail fast
  instead of consuming the full timeout.
- **Arrival tolerance 0.35 m**, heading deadband 0.6 rad, heading gain 1.5.
- Pick/place remain kinematic attach/release (no gripper articulation on Go2);
  placement is still fully physical (release + settle + read where it landed).

## D-014: Audit MVP scope

`physgate/audit/` implements the three-stream model + periodic Merkle
checkpoints from architecture doc §4 with an in-memory trail and JSONL export.
Langfuse/OTEL (decision stream) and MCAP (physical stream) back-ends are NOT
wired yet — they plug in behind `AuditTrail.record()` without changing callers.

## D-015: Subscription OAuth tokens must go through `claude -p` (corrects D-012)

D-012's diagnosis was wrong. The 429 `rate_limit_error` responses were NOT
shared-quota exhaustion — Anthropic **rejects subscription OAuth tokens
(sk-ant-oat01) on the raw API** (`api.anthropic.com/v1/messages`) regardless of
quota. The opaque `{"message": "Error"}` 429 is the rejection signature; no
amount of retrying can succeed.

The officially supported programmatic path for a Claude subscription is
**Claude Code headless mode** (`claude -p`), authenticated via the
`CLAUDE_CODE_OAUTH_TOKEN` env var (the token from `claude setup-token` is
exactly this kind of token).

Implementation (`physgate/planner/headless_client.py`):

- `ClaudeCodeHeadlessClient` wraps `claude -p --output-format json` behind the
  same `client.messages.create()` interface as the Anthropic SDK, so
  ClaudePlanner/ClaudeCritic work unchanged.
- Prompt goes via **stdin** (scene JSON exceeds argv limits); the planner/critic
  system prompt replaces Claude Code's via `--system-prompt`.
- `--bare` must NOT be used (bare mode ignores CLAUDE_CODE_OAUTH_TOKEN).
- Credential routing in `make_anthropic_client()`:
  1. `ANTHROPIC_API_KEY` → raw Anthropic SDK (pay-per-token, unrestricted)
  2. `CLAUDE_CODE_OAUTH_TOKEN` / `ANTHROPIC_AUTH_TOKEN` starting with
     `sk-ant-oat` → `claude -p` headless client
  3. other bearer tokens (gateway/proxy) → raw SDK with Bearer auth
- Trade-off: each call carries ~1-3 s of CLI startup overhead, and parallel
  candidate generation uses concurrent subprocesses instead of async HTTP.

Verified live 2026-06-03: ClaudePlanner over headless client generated valid
plans with claude-opus-4-8 in 18.5 s (2 candidates).

## D-016: Anti-hallucination validation (root cause of the first real-LLM demo escalation)

The first real-LLM Isaac demo (2026-06-03) escalated: all 7 critic-surviving
plans failed L2 at "0.0s, 0 collisions". Root cause chain:

1. Claude, asked for 8 *diverse* plans, can invent object ids that are not in
   the scene (extra waypoints, staging areas). LLM output is stochastic — some
   runs produce only valid ids, some do not.
2. ClaudeCritic (also an LLM) did not reliably reject those plans.
3. L3 only checked *declared preconditions* — a plan moving to a hallucinated
   waypoint without declaring "waypoint_X exists" passed L3.
4. `_compile_mission` silently skipped unknown move targets (`continue`),
   degrading plans to `[pick, place]`-only missions that "complete" in 0.02 s
   with the box never moved → misleading "box not on shelf" failures.

Fixes (defense in depth):

- **L3 `check_step_targets`**: every step's `target` arg must name an object in
  the scene; violations are explicit (`unknown_object`). Runs before
  preconditions in `_l3_check` — hallucinated plans never reach physics.
- **L2 `UnknownTargetError`**: `_compile_mission` now raises instead of
  silently skipping; `rollout_plans_with_policy` converts that into an explicit
  failed PhysicsResult. Backstop for direct L2 callers.
- **Planner prompt**: HARD CONSTRAINTS section forbids inventing ids and pins
  the exact args schema (standoff_m/speed ranges, skill arg format).
- **Demo report**: failed plans now print their violation details, and the full
  candidate step dump is included — no more undiagnosable failures.

Lesson recorded: validate at every trust boundary; never silently degrade
LLM-provided references. An LLM critic is not a substitute for deterministic
structural validation.

## D-017: Deterministic A* navigation replaces the straight-line driver (REBUILD.md Phase 1)

Four adversarial code reviews established that the "17% feasible / best-of-N
rescues it" headline was an **artifact of a layering defect**: obstacle
avoidance was implicitly pushed to the LLM (which had no coordinates), while
the low level drove straight lines through obstacles. The hand-placed
`waypoint_W` was a physics hack masquerading as a semantic object. See
`docs/design/REBUILD.md` §1 for the file:line evidence.

Decision: ship a **deterministic A\* occupancy-grid planner**
(`physgate/nav/path_planner.py`) instead of full ROS 2 Nav2 integration.

Trade-off (recorded per REBUILD.md §3):

- **Why not Nav2 now:** Nav2 requires the ROS 2 executor stack (not yet built),
  a costmap server, TF tree, and lifecycle nodes — heavy infrastructure that is
  orthogonal to fixing the layering defect. The A* planner provides the same
  guarantee (always routes around obstacles, deterministically) behind the same
  interface contract.
- **Nav2 drop-in boundary:** `plan_path(start_xy, goal_xy, obstacles) ->
  list[waypoint]` mirrors Nav2's `ComputePathToPose` over a costmap. When the
  ROS 2 executor lands, a Nav2-backed implementation replaces the module behind
  this signature; callers (gate, executor) do not change.
- **What was removed:** `waypoint_W` (scene, prompt, mock planner, semantics)
  and ALL straight-line interpolation toward targets
  (`synthesize_base_trajectory`, `_compile_mission`, `SimBackend.move_to_pose`).

Supporting refactor: scene layout constants moved to `physgate/world/layout.py`
(pure logic) so navigation and trajectory compilation are unit-testable without
Isaac; trajectory/mission compilation moved to `physgate/gate/trajectory.py`
(pure) and `l2_physics.py` consumes it.

Consequence for results: the pre-rebuild "17% feasible → best-of-N 99%" curve
(benchmark #8) is an artifact and is marked DEPRECATED. With navigation in the
right layer, feasibility of well-formed plans is ~100%, and the Sim-Gate's role
is **agent-orchestrator evaluation** (decomposition, ordering, preconditions,
recovery, infeasibility recognition), not route rescue.

## D-018: Placement momentum transfer + locomotion tracking envelope (REBUILD.md Phase 2)

**Bug fixed:** `rollout_plans_with_policy`'s `place` handler never called
`release_boxes` — `write_box_poses` zeroes velocity, so the box always dropped
dead onto the shelf and "fast, sloppy placements fail" never held in policy
mode (REBUILD.md §1, HIGH-severity finding).

Fix: at placement, the box is released with velocity = approach speed ×
RELEASE_VELOCITY_GAIN along the robot's heading (matching the kinematic
rollout's semantics). Measured effect (deterministic, Isaac):

| commanded approach speed | box slide after release | outcome |
|---|---|---|
| 0.4 m/s | 9 cm  | stays on shelf |
| 0.6 m/s | 17 cm | stays on shelf |
| 0.8 m/s | 29 cm | stays on shelf (0.8 × 0.4 m shelf) |

Placement quality is now physically tested: the slide scales with the
commanded speed, and faster approaches genuinely risk pushing the box off.

**Two locomotion robustness findings from the same verification:**

1. **Waypoint corner clusters stall the follower.** String-pulled A* paths can
   leave corners centimetres apart where the path hugs a clearance boundary;
   at ≥0.6 m/s the walking robot overshoots them and oscillates without net
   progress (stuck detection fires). Fix: `_merge_close_waypoints` — corners
   closer than the tracking tolerance are merged (the tracking-error clearance
   inflation already absorbs the ≤0.35 m corner cut).
2. **A kinematically-carried box must never be able to touch the robot.** The
   carry has no attachment joint (D-013's documented simplification): the box is
   pose-written every control step, which makes it an immovable obstacle for
   PhysX. Carried at +0.25 m overhead (the original choice), the pitching trunk
   /head contacts it during accelerations and turns, and the robot gets crushed
   into a stall — intermittently, depending on gait phase and PhysX state
   history. This is why pre-rebuild routes (no sharp turns while carrying)
   never hit it. Fix: the carry position is now **+0.6 m overhead — explicitly
   bookkeeping, not physics**. It clears every robot posture, the shelf, and
   the pillar. What remains physical about manipulation: the box leaves its
   original location at pick, and is released WITH momentum at placement (the
   physics that decides placement success). A proper gripper articulation /
   attachment joint replaces this when manipulation becomes a real subject of
   validation.

**Reset identity requires resetting actuator TARGETS, not just joint states.**
`reset_scene_to_identical_state` originally wrote root/joint states but left the
articulation's joint position targets from the previous rollout in place; a
rollout that ended with a crouched/stalled robot poisoned every subsequent
rollout in the same process (the settle steps drove the joints back toward the
stale crouch). Fixed: the reset now also writes default position/velocity/effort
targets. This was the source of the "works in a fresh world, fails in a reused
world" flakiness.

**The locomotion stack's reliable envelope is [0.4, 0.6] m/s commanded.** Above
~0.6 m/s, the policy + waypoint-follower combination intermittently collapses
the robot into a crouch-stall (not a fall — projected gravity stays "upright"),
sensitive to gait phase and PhysX state history. This is a low-level capability
limit of the *current* policy (300-iteration rsl_rl flat-terrain training) +
P-controller navigator, not an orchestration property. Mission speeds are
clamped to that envelope and the planner prompt advertises 0.4–0.6; plans
commanding out-of-range speeds run at the nearest reliable speed instead of
failing on a limitation the LLM cannot know about. Revisit after training a
more robust policy (more iterations, velocity-command curriculum) or replacing
the P-controller navigator with a proper local planner.

## D-019: Relation vocabulary is part of the tool contract (orchestration-eval finding)

The first real-LLM run of the orchestration evaluation (Phase 3) scored Claude
*below* the mock planner — because every Claude plan failed the symbolic
executor's precondition recheck. Root cause: a **vocabulary mismatch**, not an
orchestration deficiency. Claude grounded relations in scene object ids
("go2 near box_03", "go2 holding box_03" — entirely reasonable), while
MockWorldBackend's internal convention is "robot near <id>" / "gripper holding
<id>". The mock planner only "passed" because it was written against the same
private convention (and declares almost no relational preconditions at all).

Decision: the relation vocabulary is part of the **tool contract** and is now
documented in the planner system prompt (RELATION VOCABULARY section). An LLM
that still uses other subjects after being told the contract is exhibiting a
real instruction-following deficiency; before being told, it was not.

Evaluation-integrity lesson: when an eval shows "the LLM is worse than a
trivial baseline", first suspect the eval's interface, not the LLM. The mock
baseline had insider knowledge of the backend's private vocabulary.

## D-020: Never command pure rotation — the policy cannot turn in place (Phase 2 verification finding)

The definitive Phase 2 verification (full Isaac suite ×2 + feasibility) exposed a
deterministic stall: robots froze mid-mission at sharp path corners with
"no progress for 10 s while navigating", dropping mock feasibility to 67%.

Root cause (benchmarks/rebuild/diag_turn_in_place.py): the trained Go2 flat
policy has a **"keep standing" fixed point** — commanded pure rotation
(vx=0, wz=0.9) from a standstill it executes only ~10% of the turn, while the
same wz with vx=0.2 tracks ~95% and with vx=0.4 tracks ~99%. The waypoint
navigator's heading deadband commanded exactly that: heading error ≥ 0.6 rad →
vx=0 + wz → robot decelerates to standstill → never turns → deadlock. Whether a
given rollout deadlocked depended on the exact joint state on corner arrival,
which is why *identical* plans diverged (one passed, one stalled) and why the
failure looked plan/env-random.

Decision: `WaypointNavigator` never commands pure rotation. Outside the heading
deadband it commands `TURN_CREEP_SPEED = 0.2 m/s` forward + the turn rate: the
robot turns on a ~0.22 m arc, which stays inside the path planner's
WAYPOINT_TRACKING_TOLERANCE (0.35 m) clearance inflation, so turning arcs cannot
violate obstacle clearance.

Evaluation-integrity lesson (same family as D-019): a stall that *looks* like
"some plans are physically infeasible" can be a low-level controller fixed
point. Before attributing failures to the plan layer, check what command the
navigator was issuing when the robot stopped.

## D-021: Decomposition checks must start from the scenario's initial state (data-review finding)

The post-rebuild data plausibility review caught a contradiction in the
orchestration results: Claude's plan for the occupied-gripper scenario
**completed the task** (box on shelf, relations verified) yet was scored
`decomposition_valid=False`, dragging decomposition validity to 0.8 and the
orchestrator score to 0.96.

Root cause: `check_decomposition` assumed the gripper starts EMPTY. The
occupied-gripper scenario starts with the gripper holding another box, so
Claude's correct first step — set the held box down, then fetch the target —
tripped the checker's "place requires holding something" rule on a state the
checker did not model. Same family as D-019: an eval-interface bug masquerading
as an LLM deficiency.

Fix: `check_decomposition(plan, initially_held=...)`; the runner derives the
initial held object from the scenario scene's `("gripper", "holding", X)`
relation. After the fix and a full re-run, the per-scenario diff against the
previous run shows exactly ONE changed field (that flag); every other outcome
is identical across both runs — the score change (0.96 → 1.00) is fully
attributable to the checker fix, not to LLM nondeterminism.

Evaluation-integrity lesson: **a completed task with an "invalid" decomposition
is a contradiction that must be investigated, not reported.** Plausibility
review of every metric against its raw per-item data is now part of the
benchmark workflow (and the README results section is generated from the JSONs
by `benchmarks/render_readme.py`, so reported numbers cannot drift from data).

## D-022: Navigation infeasibility is a gate verdict, not a crash (REBUILD Phase 3 completion)

The audit against REBUILD.md Phase 3's failure-mode list found the "goal
navigation cannot reach" case unhandled: `compile_mission` correctly raised
`PathPlannerError` for unreachable goals (defense-in-depth backstop), but
neither Isaac rollout nor the executor caught it — an unreachable target
crashed the gate instead of producing an infeasible verdict.

Fix (TDD, 3 new Isaac tests): both rollouts return an
`infeasible_navigation` violation (retryable=False — replanning cannot fix
geometry) and `SimBackend.move_to_pose` returns a failed action, so the
orchestrator escalates cleanly.

Where each Phase 3 example is tested (and why):

| REBUILD.md example | Where it is tested |
|---|---|
| wrong ordering / inverted plans | orchestration suite (symbolic) |
| pick with a full gripper | orchestration suite (symbolic) |
| place on an occupied shelf | **not applicable** — the shelf is multi-capacity; single-occupancy would contradict multi_step_two_boxes |
| recovery (transient/persistent) | orchestration suite (symbolic, fault injection) |
| multi-step decomposition | orchestration suite (symbolic) |
| infeasible: ungraspable object | orchestration suite (symbolic) |
| infeasible: unreachable goal | **gate level** (Isaac tests) — the symbolic suite has no geometry, so reachability can only be tested where navigation actually runs |
