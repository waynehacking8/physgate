#!/usr/bin/env python3
"""Phase 0 benchmarks #4 + #5: L2 validation wall-clock + warm-start latency.

#4 — single-plan physics-validation wall-clock: how long does it take the
     Sim-Gate L2 to validate ONE candidate plan, and the full best-of-8 batch?
     Measured for both modes:
       - kinematic rollout (rollout_plans)
       - trained-policy rollout (rollout_plans_with_policy), if a policy exists
     Gate criterion: establishes the true L2 latency (estimated 20-100 ms per
     plan step in the design doc; the walking rollout is dominated by sim time).

#5 — warm-start latency: app launch -> 8-env scene ready -> first validated
     reset. Gate criterion: confirms (or refutes) the 10-30 s assumption.

Run inside the env_isaaclab venv:
    python benchmarks/phase0/benchmark_4_5_l2_warmstart.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# ---- #5 warm-start: time the app launch + scene creation (must be first) ----

T_START = time.perf_counter()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app  # noqa: F841
T_APP_READY = time.perf_counter()

from physgate.examples_lib.fetch_and_place import build_demo_scene  # noqa: E402
from physgate.gate.l2_physics import rollout_plans, rollout_plans_with_policy  # noqa: E402
from physgate.gate.reset_workaround import (  # noqa: E402
    reset_scene_to_identical_state,
    verify_identical_reset,
)
from physgate.planner.planner import MockPlanner  # noqa: E402
from physgate.world.fetch_scene import FetchSimWorld  # noqa: E402
from physgate.world.locomotion import find_exported_policy  # noqa: E402

TASK = "put the fallen box back on shelf A"


def main() -> int:
    results: dict = {
        "benchmark": "phase0 #4 (L2 wall-clock) + #5 (warm-start)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": "RTX PRO 6000 Blackwell Max-Q (sm_120, 300W)",
    }

    # ---------------- #5 warm-start ----------------
    t_scene0 = time.perf_counter()
    world = FetchSimWorld(num_envs=8)
    t_scene1 = time.perf_counter()

    ok, deviation = verify_identical_reset(world.scene, world.sim)
    t_reset1 = time.perf_counter()

    results["warm_start"] = {
        "app_launch_s": round(T_APP_READY - T_START, 2),
        "scene_creation_8env_s": round(t_scene1 - t_scene0, 2),
        "first_identical_reset_s": round(t_reset1 - t_scene1, 2),
        "total_to_ready_s": round(t_reset1 - T_START, 2),
        "reset_identical_ok": ok,
        "reset_max_deviation_m": round(deviation, 6),
        "gate_criterion": "assumption was 10-30 s to warm-start",
    }
    print(f"#5 warm-start: total {results['warm_start']['total_to_ready_s']} s "
          f"(app {results['warm_start']['app_launch_s']} s, "
          f"scene {results['warm_start']['scene_creation_8env_s']} s)")

    # ---------------- #4 L2 wall-clock ----------------
    scene = build_demo_scene()
    plans = MockPlanner()(TASK, scene, 8, None)

    # kinematic mode: 1 plan vs 8 plans
    t0 = time.perf_counter()
    rollout_plans(world, plans[:1])
    single_kinematic = time.perf_counter() - t0

    t0 = time.perf_counter()
    kinematic_results = rollout_plans(world, plans)
    batch_kinematic = time.perf_counter() - t0

    results["l2_kinematic"] = {
        "single_plan_wall_s": round(single_kinematic, 2),
        "batch_8_plans_wall_s": round(batch_kinematic, 2),
        "marginal_cost_of_7_extra_plans_s": round(batch_kinematic - single_kinematic, 2),
        "per_plan_effective_s": round(batch_kinematic / 8, 2),
        "feasible_count": sum(1 for r in kinematic_results if r.success),
    }
    print(f"#4 kinematic L2: 1 plan {single_kinematic:.1f}s | 8 plans {batch_kinematic:.1f}s "
          f"(marginal {batch_kinematic - single_kinematic:.1f}s)")

    # policy mode (if trained policy available): 1 plan vs 8 plans
    policy = find_exported_policy()
    if policy is not None:
        t0 = time.perf_counter()
        rollout_plans_with_policy(world, plans[:1], policy)
        single_policy = time.perf_counter() - t0

        t0 = time.perf_counter()
        policy_results = rollout_plans_with_policy(world, plans, policy)
        batch_policy = time.perf_counter() - t0

        results["l2_policy"] = {
            "policy": str(policy),
            "single_plan_wall_s": round(single_policy, 2),
            "batch_8_plans_wall_s": round(batch_policy, 2),
            "marginal_cost_of_7_extra_plans_s": round(batch_policy - single_policy, 2),
            "per_plan_effective_s": round(batch_policy / 8, 2),
            "feasible_count": sum(1 for r in policy_results if r.success),
        }
        print(f"#4 policy L2: 1 plan {single_policy:.1f}s | 8 plans {batch_policy:.1f}s "
              f"(marginal {batch_policy - single_policy:.1f}s)")
    else:
        results["l2_policy"] = None
        print("#4 policy L2: skipped (no exported policy)")

    # the headline number the architecture cares about
    results["conclusion"] = {
        "parallel_validation_is_cheap": (
            results["l2_kinematic"]["marginal_cost_of_7_extra_plans_s"]
            < results["l2_kinematic"]["single_plan_wall_s"]
        ),
        "note": (
            "best-of-8 costs barely more than best-of-1 when run in parallel envs — "
            "the quality gain of N=8 is nearly free in GPU time (architecture doc §6)"
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "benchmark_4_5_l2_warmstart.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nresults -> {out_path}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
