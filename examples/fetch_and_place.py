#!/usr/bin/env python3
"""physgate MVP demo: fetch-and-place, end to end.

    natural language -> Claude plans (N=8) -> critic -> Sim-Gate best-of-N
    -> execute in simulation -> report

Usage:
    python examples/fetch_and_place.py                 # offline (mock planner + symbolic physics)
    python examples/fetch_and_place.py --isaac          # Isaac Sim L2 physics + execution
                                                         # (requires the env_isaaclab venv)

Real Claude planner/critic activates automatically when LLM credentials are set:
ANTHROPIC_API_KEY (API key) or CLAUDE_CODE_OAUTH_TOKEN (subscription, via claude -p).
"""

from __future__ import annotations

import argparse
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
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="pause at the approval gate and ask for human confirmation (LangGraph interrupt)",
    )
    args = parser.parse_args()

    kwargs = {}
    if args.isaac:
        # Isaac Sim must be launched BEFORE any isaaclab/physgate-sim import.
        print("[isaac] launching headless Isaac Sim (this takes ~30-60 s)...")
        from isaaclab.app import AppLauncher

        _simulation_app = AppLauncher(headless=True).app  # noqa: F841 — keeps the app alive

        from physgate.executor.sim_backend import PolicySimExecutor, SimBackend
        from physgate.gate.l2_physics import IsaacL2Gate, PolicyL2Gate
        from physgate.gate.reset_workaround import reset_scene_to_identical_state
        from physgate.world.fetch_scene import get_shared_world
        from physgate.world.locomotion import find_exported_policy
        from physgate.world.usd_semantics import scene_from_stage

        print("[isaac] building 8-env fetch-and-place scene...")
        world = get_shared_world(num_envs=8)
        reset_scene_to_identical_state(world.scene, world.sim)

        # perceive the symbolic scene FROM the simulation (USD semantics, C13)
        print("[isaac] perceiving scene from USD stage semantics...")
        kwargs["scene"] = scene_from_stage(env_index=0)

        policy = find_exported_policy()
        if policy is not None:
            # trained Go2 locomotion policy: robots WALK (L2 gate + executor)
            print(f"[isaac] using trained locomotion policy: {policy}")
            kwargs["l2_fn"] = PolicyL2Gate(world, policy)
            kwargs["executor_fn_override"] = PolicySimExecutor(world, policy)
        else:
            # fall back to kinematic base driving
            print("[isaac] no trained policy found -> kinematic base driving")
            kwargs["l2_fn"] = IsaacL2Gate(world)
            kwargs["backend_factory"] = lambda scene: SimBackend(scene, world=world)

    from physgate.examples_lib.fetch_and_place import run_fetch_and_place

    from physgate.planner.planner import llm_credentials_available

    if llm_credentials_available():
        print("[planner] LLM credentials found -> using real Claude planner/critic")
    else:
        print("[planner] no LLM credentials -> using deterministic mock planner/critic")
        print("[planner] (set ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN for the real LLM)")

    if args.interactive:
        kwargs["auto_approve"] = False

    outcome = run_fetch_and_place(task=args.task, **kwargs)
    print()
    print(outcome["report"])
    print(f"audit: {len(outcome['audit_trail']._records)} records, "
          f"merkle root {outcome['audit_merkle_root'][:16]}..., "
          f"integrity={'OK' if outcome['audit_trail'].verify_integrity() else 'TAMPERED'}")
    return 0 if outcome["outcome"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
