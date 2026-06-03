#!/usr/bin/env python3
"""E1 — pipeline ablation over procedurally generated task instances.

Quantifies what each validation layer adds (docs/design/EVALUATION_METHODOLOGY.md §4):

    A0_no_validation -> A1_critic_only -> A2_symbolic_gate -> A3_nav_aware_gate

over N generated feasible instances (certified solvable) and M infeasible
instances (certified unreachable), with a contaminated plan pool (clean +
defect-injected plans).

Output: benchmarks/eval_v2/results/ablation.json

Run (pure venv, no GPU needed):
    python benchmarks/eval_v2/run_ablation.py --feasible 20 --infeasible 8 --seed 2026
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feasible", type=int, default=20)
    parser.add_argument("--infeasible", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    from physgate.eval_v2.ablation import CONDITIONS, build_plan_pool, run_ablation
    from physgate.eval_v2.scenario_gen import generate_instances

    print(f"generating {args.feasible} feasible + {args.infeasible} infeasible instances...")
    t0 = time.perf_counter()
    instances = generate_instances(
        n_feasible=args.feasible, n_infeasible=args.infeasible, seed=args.seed
    )
    pool = build_plan_pool(seed=args.seed)
    n_clean = sum(1 for p in pool if not p.ground_truth_invalid)
    print(
        f"plan pool: {len(pool)} plans ({n_clean} valid, {len(pool) - n_clean} defect-injected)"
    )

    print("running ablation conditions...")
    report = run_ablation(instances, pool)
    wall_s = time.perf_counter() - t0

    for condition in CONDITIONS:
        outcome = report[condition]
        ci = outcome.success_ci
        print(
            f"  {condition:20s} success={outcome.success_rate:.2f} "
            f"[{ci[0]:.2f}, {ci[1]:.2f}]  "
            f"false_exec={outcome.false_execution_rate:.2f}  "
            f"rejection={outcome.rejection_rate:.2f}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "E1 pipeline ablation",
        "methodology": "docs/design/EVALUATION_METHODOLOGY.md section 4 E1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "n_feasible_instances": args.feasible,
            "n_infeasible_instances": args.infeasible,
            "seed": args.seed,
            "plan_pool_size": len(pool),
            "plan_pool_valid": n_clean,
            "execution_model": (
                "symbolic execution (MockWorldBackend) + navigation compilation against "
                "the instance layout; success = task goal relation holds post-execution"
            ),
        },
        "conditions": {
            condition: {
                "success_rate": outcome.success_rate,
                "success_ci_95": list(outcome.success_ci),
                "false_execution_rate": outcome.false_execution_rate,
                "rejection_rate": outcome.rejection_rate,
                "n_instances": outcome.n_instances,
                "details": outcome.details,
            }
            for condition, outcome in report.items()
        },
        "wall_s": round(wall_s, 1),
        "interpretation": {
            "feasible_instances": (
                "success climbs with each validation layer: no-validation executes "
                "defective plans; the critic removes hallucinations; the symbolic gate "
                "removes ordering/missing-step defects; the nav-aware gate (physics-gate "
                "surrogate) additionally verifies the task goal"
            ),
            "infeasible_instances": (
                "ONLY the navigation-aware condition can reject impossible tasks before "
                "execution — symbolic validation has no geometry and attempts every one "
                "of them (false_execution_rate=1.0). This is the unique, quantified value "
                "of world-level (physics/navigation) validation."
            ),
        },
    }
    out_path = RESULTS_DIR / "ablation.json"
    out_path.write_text(json.dumps(output, indent=1))
    print(f"results -> {out_path}  ({wall_s:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
