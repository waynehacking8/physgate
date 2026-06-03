#!/usr/bin/env python3
"""Phase 0 benchmark #2: bug #2133 same-state parallel reset + rollout determinism.

Go/No-Go criterion (benchmarks/phase0/README.md): if N envs cannot be reset to
an identical state, the parallel best-of-N validation method (C2) must be
redesigned. This benchmark quantifies three properties the L2 gate depends on:

  A. reset identity      after reset, all N envs are in the same state
                         (max cross-env deviation < 1 mm)
  B. reset repeatability state after reset cycle i == state after cycle j
                         (resets do not drift)
  C. rollout determinism the SAME plan batch rolled out twice from identical
                         resets produces the SAME physics verdicts — this is
                         what makes L2 verdicts reproducible/scientific.

Each env count runs in a subprocess (Isaac launches once per process). Run
inside the env_isaaclab venv:

    python benchmarks/phase0/benchmark_2_reset_determinism.py
    python benchmarks/phase0/benchmark_2_reset_determinism.py --worker 8   # (internal)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ENV_COUNTS = [8, 64]
RESET_CYCLES = 5
IDENTITY_TOLERANCE_M = 1e-3
RESULTS_DIR = Path(__file__).parent / "results"
TASK = "put the fallen box back on shelf A"


def worker(num_envs: int, reset_cycles: int) -> None:
    """Subprocess body: reset identity/repeatability (+ rollout determinism at 8 envs)."""
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app  # noqa: F841

    import torch

    from physgate.gate.reset_workaround import (
        max_state_deviation,
        reset_scene_to_identical_state,
    )
    from physgate.world.fetch_scene import FetchSimWorld

    world = FetchSimWorld(num_envs=num_envs)

    # ---- A + B: reset identity and repeatability across cycles ----
    identity_deviations: list[float] = []
    reference_state: torch.Tensor | None = None
    cross_cycle_deviations: list[float] = []
    for _cycle in range(reset_cycles):
        reset_scene_to_identical_state(world.scene, world.sim)
        identity_deviations.append(float(max_state_deviation(world.scene)))

        # snapshot env 0's robot root + box position as the cycle's state signature
        signature = torch.cat(
            [
                (world.robot.data.root_pos_w[0] - world.env_origins[0]),
                (world.box.data.root_pos_w[0] - world.env_origins[0]),
            ]
        ).clone()
        if reference_state is None:
            reference_state = signature
        else:
            cross_cycle_deviations.append(float((signature - reference_state).abs().max()))

    result: dict = {
        "num_envs": num_envs,
        "reset_cycles": reset_cycles,
        "identity_max_deviation_m": round(max(identity_deviations), 6),
        "identity_deviations_m": [round(d, 6) for d in identity_deviations],
        "repeatability_max_deviation_m": (
            round(max(cross_cycle_deviations), 6) if cross_cycle_deviations else 0.0
        ),
    }

    # ---- C: rollout determinism (operational 8-env config only) ----
    if num_envs == 8:
        from physgate.examples_lib.fetch_and_place import build_demo_scene
        from physgate.gate.l2_physics import rollout_plans_with_policy
        from physgate.planner.planner import MockPlanner
        from physgate.world.locomotion import find_exported_policy

        policy = find_exported_policy()
        if policy is not None:
            plans = MockPlanner()(TASK, build_demo_scene(), 8, None)
            runs = []
            for _ in range(2):
                t0 = time.perf_counter()
                results = rollout_plans_with_policy(world, plans, policy)
                runs.append(
                    {
                        "wall_s": round(time.perf_counter() - t0, 2),
                        "verdicts": {r.plan_id: r.success for r in results},
                        "times": {r.plan_id: r.completion_time_s for r in results},
                        "collisions": {r.plan_id: r.collision_count for r in results},
                    }
                )
            verdicts_match = runs[0]["verdicts"] == runs[1]["verdicts"]
            collisions_match = runs[0]["collisions"] == runs[1]["collisions"]
            time_diffs = [
                abs(runs[0]["times"][pid] - runs[1]["times"][pid]) for pid in runs[0]["times"]
            ]
            result["rollout_determinism"] = {
                "verdicts_identical": verdicts_match,
                "collision_counts_identical": collisions_match,
                "max_completion_time_diff_s": round(max(time_diffs), 2),
                "run_1": runs[0],
                "run_2": runs[1],
            }
        else:
            result["rollout_determinism"] = None

    print("RESULT_JSON: " + json.dumps(result))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker", type=int, default=None, help="(internal) run as worker for N envs"
    )
    parser.add_argument("--env-counts", type=int, nargs="+", default=ENV_COUNTS)
    parser.add_argument("--reset-cycles", type=int, default=RESET_CYCLES)
    args = parser.parse_args()

    if args.worker is not None:
        worker(args.worker, args.reset_cycles)
        return 0

    per_count: list[dict] = []
    for n in args.env_counts:
        print(f"=== benchmark #2: {n} envs (subprocess) ===")
        proc = subprocess.run(
            [
                sys.executable,
                __file__,
                "--worker",
                str(n),
                "--reset-cycles",
                str(args.reset_cycles),
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT_JSON: "):
                entry = json.loads(line[len("RESULT_JSON: ") :])
                per_count.append(entry)
                print(
                    f"    identity dev {entry['identity_max_deviation_m'] * 1000:.3f} mm | "
                    f"repeatability dev {entry['repeatability_max_deviation_m'] * 1000:.3f} mm"
                )
                break
        else:
            print(f"    WORKER FAILED for {n} envs:\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}")

    # ---- go/no-go evaluation ----
    identity_ok = all(e["identity_max_deviation_m"] <= IDENTITY_TOLERANCE_M for e in per_count)
    repeatability_ok = all(
        e["repeatability_max_deviation_m"] <= IDENTITY_TOLERANCE_M for e in per_count
    )
    det = next(
        (e.get("rollout_determinism") for e in per_count if e.get("rollout_determinism")), None
    )
    determinism_ok = bool(det and det["verdicts_identical"] and det["collision_counts_identical"])

    output = {
        "benchmark": "phase0 #2 - bug #2133 same-state parallel reset + rollout determinism",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": "RTX PRO 6000 Blackwell Max-Q (sm_120, 300W)",
        "tolerance_m": IDENTITY_TOLERANCE_M,
        "per_env_count": per_count,
        "go_no_go": {
            "reset_identity_ok": identity_ok,
            "reset_repeatability_ok": repeatability_ok,
            "rollout_verdicts_deterministic": determinism_ok,
            "verdict": "GO" if (identity_ok and repeatability_ok and determinism_ok) else "NO-GO",
        },
        "interpretation": (
            "All envs reset to the same state within tolerance, resets do not drift "
            "across cycles, and identical plan batches produce identical L2 verdicts — "
            "the parallel best-of-N method (C2) is valid and its verdicts are reproducible."
            if (identity_ok and repeatability_ok and determinism_ok)
            else "Reset or determinism failure — see per_env_count details; C2 needs redesign."
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "benchmark_2_reset_determinism.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nresults -> {out_path}")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
