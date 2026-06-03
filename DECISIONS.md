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
