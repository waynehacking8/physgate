#!/usr/bin/env python3
"""REBUILD.md Phase 1 acceptance: plan feasibility after removing the routing artifact.

Before the rebuild, "17% feasible" measured whether the LLM happened to name the
hand-placed waypoint_W — an artifact of obstacle avoidance living in the wrong
layer. After the rebuild (deterministic A* navigation), any well-formed
pick -> carry -> place decomposition must be physically feasible.

This benchmark measures feasibility for:

  1. mock planner candidates (deterministic, critic-surviving),
  2. real Claude plans (when LLM credentials are configured),

by rolling every plan out with the trained walking policy in parallel Isaac envs.

Acceptance: feasibility ~= 100% (vs 17% before the rebuild).

Run inside the env_isaaclab venv:

    python benchmarks/rebuild/verify_feasibility.py            # mock only
    python benchmarks/rebuild/verify_feasibility.py --llm      # + real Claude plans
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
TASK = "put the fallen box back on shelf A"


def measure_feasibility(world, plans, survivors, policy, label: str) -> dict:
    """Roll out critic-surviving plans with the walking policy; report feasibility."""
    from physgate.gate.l2_physics import rollout_plans_with_policy

    t0 = time.perf_counter()
    results = rollout_plans_with_policy(world, survivors, policy)
    wall_s = time.perf_counter() - t0

    feasible = [r for r in results if r.success]
    return {
        "planner": label,
        "candidates_generated": len(plans),
        "critic_survivors": len(survivors),
        "physically_feasible": len(feasible),
        "feasibility_of_survivors": round(len(feasible) / len(survivors), 3) if survivors else 0.0,
        "gpu_wall_s": round(wall_s, 1),
        "per_plan": [
            {
                "plan_id": r.plan_id,
                "success": r.success,
                "collisions": r.collision_count,
                "time_s": r.completion_time_s,
                "failure": (
                    r.failure.violations[0].detail if (r.failure and r.failure.violations) else None
                ),
            }
            for r in results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", action="store_true", help="also measure real Claude plans")
    parser.add_argument("--num-plans", type=int, default=8)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app  # noqa: F841

    from physgate.examples_lib.fetch_and_place import build_demo_scene
    from physgate.planner.critic import MockCritic, make_critic
    from physgate.planner.planner import MockPlanner, llm_credentials_available, make_planner
    from physgate.world.fetch_scene import FetchSimWorld
    from physgate.world.locomotion import find_exported_policy

    policy = find_exported_policy()
    if policy is None:
        print("ERROR: no exported Go2 policy found — train it first (rsl_rl)")
        return 1

    scene = build_demo_scene()
    world = FetchSimWorld(num_envs=args.num_plans)
    measurements: list[dict] = []

    # ---- mock planner (deterministic baseline) ----
    print(f"=== mock planner: {args.num_plans} candidates ===")
    mock_plans = MockPlanner()(TASK, scene, args.num_plans, None)
    mock_survivors = MockCritic()(mock_plans, scene)
    entry = measure_feasibility(world, mock_plans, mock_survivors, policy, "mock")
    measurements.append(entry)
    print(
        f"    {entry['physically_feasible']}/{entry['critic_survivors']} survivors feasible "
        f"({entry['feasibility_of_survivors']:.0%}) in {entry['gpu_wall_s']}s"
    )

    # ---- real Claude plans ----
    if args.llm:
        if not llm_credentials_available():
            print("WARNING: --llm requested but no LLM credentials found; skipping")
        else:
            print(f"=== real Claude planner: {args.num_plans} candidates ===")
            planner = make_planner()
            critic = make_critic()
            llm_plans = planner(TASK, scene, args.num_plans, None)
            llm_survivors = critic(llm_plans, scene)
            entry = measure_feasibility(world, llm_plans, llm_survivors, policy, "claude")
            measurements.append(entry)
            print(
                f"    {entry['physically_feasible']}/{entry['critic_survivors']} survivors feasible "
                f"({entry['feasibility_of_survivors']:.0%}) in {entry['gpu_wall_s']}s"
            )

    # ---- verdict ----
    feasibilities = [m["feasibility_of_survivors"] for m in measurements]
    artifact_eliminated = all(f >= 0.8 for f in feasibilities)

    output = {
        "benchmark": "rebuild Phase 1 - plan feasibility after deterministic navigation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": "RTX PRO 6000 Blackwell Max-Q (sm_120, 300W)",
        "context": (
            "Pre-rebuild feasibility was 17% (24-plan pool) / 1-in-7 (real-LLM demo) — an "
            "ARTIFACT of obstacle avoidance living in the wrong layer (straight-line driver "
            "+ hand-placed waypoint_W). After the rebuild, navigation is deterministic A* "
            "and every well-formed decomposition should be feasible."
        ),
        "measurements": measurements,
        "acceptance": {
            "criterion": "feasibility of critic-surviving plans >= 80% (~100% expected)",
            "artifact_eliminated": artifact_eliminated,
        },
        "interpretation": (
            "Feasibility is no longer a route lottery: "
            + "; ".join(
                f"{m['planner']}: {m['physically_feasible']}/{m['critic_survivors']} "
                f"({m['feasibility_of_survivors']:.0%})"
                for m in measurements
            )
            + ". The 17%-feasible result was an artifact of the layering defect, "
            "as REBUILD.md §1 concluded."
            if artifact_eliminated
            else "FEASIBILITY STILL LOW — the artifact is NOT eliminated; investigate per_plan failures."
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "feasibility_after_rebuild.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nresults -> {out_path}")
    print(json.dumps(output["acceptance"], indent=2))
    print(output["interpretation"])
    return 0 if artifact_eliminated else 1


if __name__ == "__main__":
    sys.exit(main())
