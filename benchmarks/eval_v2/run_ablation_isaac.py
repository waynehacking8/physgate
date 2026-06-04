#!/usr/bin/env python3
"""E1 ablation with REAL Isaac physics L2 — requires env_isaaclab venv + GPU.

Runs the A3_physics condition on a subset of instances alongside the full
symbolic conditions. This proves the Isaac physics gate produces equivalent
results to the symbolic surrogate on the standard instance set.

Usage (from repo root, Isaac venv activated):
    source ~/env_isaaclab/bin/activate
    python benchmarks/eval_v2/run_ablation_isaac.py --physics-instances 5

The AppLauncher must initialize before any other Isaac import.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Parse args BEFORE AppLauncher (it consumes unknown args otherwise)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--physics-instances", type=int, default=5)
parser.add_argument("--seed", type=int, default=2026)
args, _remaining = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Launch the Omniverse app at module level (same pattern as test_isaac_sim_gate)
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: E402

simulation_app = AppLauncher(headless=True).app

# Now safe to import Isaac-dependent code
from physgate.eval_v2.ablation import (  # noqa: E402
    CONDITIONS_WITH_PHYSICS,
    build_plan_pool,
    run_ablation,
)
from physgate.eval_v2.scenario_gen import generate_instances  # noqa: E402
from physgate.executor.sim_backend import FetchSimWorld  # noqa: E402
from physgate.gate.l2_physics import IsaacL2Gate  # noqa: E402
from physgate.gate.parallel import run_gate  # noqa: E402


def main() -> int:
    n_feasible = args.physics_instances
    n_infeasible = max(2, n_feasible // 3)

    print(f"generating {n_feasible} feasible + {n_infeasible} infeasible instances (seed={args.seed})...")
    instances = generate_instances(n_feasible=n_feasible, n_infeasible=n_infeasible, seed=args.seed)
    pool = build_plan_pool(seed=args.seed)
    n_clean = sum(1 for p in pool if not p.ground_truth_invalid)
    print(f"plan pool: {len(pool)} plans ({n_clean} valid, {len(pool) - n_clean} defect-injected)")

    print("creating Isaac Sim world (kinematic L2)...")
    t0 = time.perf_counter()
    world = FetchSimWorld(num_envs=max(len(pool), 8))
    l2_fn = IsaacL2Gate(world)

    def isaac_gate_fn(plans, scene):
        return run_gate(plans, scene, l2_fn=l2_fn)

    print(f"Isaac world ready ({time.perf_counter() - t0:.1f}s)")

    print(f"running ablation ({len(CONDITIONS_WITH_PHYSICS)} conditions including A3_physics)...")
    report = run_ablation(
        instances, pool,
        conditions=CONDITIONS_WITH_PHYSICS,
        isaac_gate_fn=isaac_gate_fn,
    )
    wall_s = time.perf_counter() - t0

    for condition, outcome in report.items():
        ci = outcome.success_ci
        print(
            f"  {condition:20s} success={outcome.success_rate:.2f} "
            f"[{ci[0]:.2f}, {ci[1]:.2f}]  false_exec={outcome.false_execution_rate:.2f}  "
            f"rejection={outcome.rejection_rate:.2f}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "config": {
            "n_feasible_instances": n_feasible,
            "n_infeasible_instances": n_infeasible,
            "seed": args.seed,
            "plan_pool_size": len(pool),
            "includes_isaac_physics": True,
        },
        "conditions": {
            condition: {
                "success_rate": outcome.success_rate,
                "success_ci_95": list(outcome.success_ci),
                "n_plan_trials": outcome.n_plan_trials,
                "plan_level_success_rate": outcome.plan_level_success_rate,
                "false_execution_rate": outcome.false_execution_rate,
                "rejection_rate": outcome.rejection_rate,
                "n_instances": outcome.n_instances,
            }
            for condition, outcome in report.items()
        },
        "wall_s": round(wall_s, 1),
    }
    out = RESULTS_DIR / "ablation_isaac.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nresults -> {out}  ({wall_s:.0f}s)")

    if hasattr(world, "close"):
        world.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
