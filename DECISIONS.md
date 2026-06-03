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
