#!/usr/bin/env python3
"""physgate MVP demo: fetch-and-place, end to end.

    natural language -> Claude plans (N=8) -> critic -> Sim-Gate best-of-N
    -> execute in simulation -> report

Usage:
    python examples/fetch_and_place.py                 # offline (mock planner + symbolic physics)
    ANTHROPIC_API_KEY=... python examples/fetch_and_place.py   # real Claude planner/critic
    python examples/fetch_and_place.py --isaac          # Isaac Sim L2 physics + execution
                                                         # (requires the env_isaaclab venv)
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="put the fallen box back on shelf A",
        help="natural-language task for the robot",
    )
    parser.add_argument(
        "--isaac",
        action="store_true",
        help="use Isaac Sim for L2 physics + execution (requires env_isaaclab venv)",
    )
    args = parser.parse_args()

    from physgate.examples_lib.fetch_and_place import run_fetch_and_place

    kwargs = {}
    if args.isaac:
        # Late import: only valid inside the Isaac Sim python environment.
        from physgate.executor.sim_backend import SimBackend
        from physgate.gate.l2_physics import isaac_l2

        kwargs["l2_fn"] = isaac_l2
        kwargs["backend_factory"] = SimBackend

    if os.environ.get("ANTHROPIC_API_KEY"):
        print("[planner] ANTHROPIC_API_KEY found -> using Claude API planner/critic")
    else:
        print("[planner] no ANTHROPIC_API_KEY -> using deterministic mock planner/critic")
        print("[planner] (export ANTHROPIC_API_KEY=... to plan with the real LLM)")

    outcome = run_fetch_and_place(task=args.task, **kwargs)
    print()
    print(outcome["report"])
    return 0 if outcome["outcome"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
