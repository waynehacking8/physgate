"""Sim-Gate Layer 2: parallel physics validation in Isaac Lab (C11).

Given N candidate plans, runs each one in its own Isaac Lab env (all envs reset
to an IDENTICAL initial state via gate/reset_workaround.py), simultaneously:

1. each plan is compiled to a navigation-routed trajectory/mission
   (``gate/trajectory.py`` — every move_to_pose goes through the deterministic
   A* path planner, so routes ALWAYS avoid obstacles),
2. all N robots execute their plan while the scene objects (the box) simulate
   dynamically on the GPU,
3. physics outcomes are read back per env:
     - task success  = the box physically ended up resting on the shelf
       (after the placement release, dynamics decide whether it stays or
       slides/bounces off — fast, sloppy plans drop it),
     - collisions    = path segments sweeping through static obstacles
       (kinematic mode) / obstacle-proximity events on the walked path (policy mode),
     - time / energy = from the executed trajectory.

What L2 validates (REBUILD.md §2): the AGENT's plan quality — decomposition
correctness, step ordering, preconditions, and physical outcome (placement
stability). It does NOT validate route geometry; the navigation layer
guarantees that for every plan.

IMPORTANT: import only after SimulationApp launch.
"""

from __future__ import annotations

from physgate.gate.rollouts import (  # noqa: F401  (public API re-exports)
    PLACE_REACH_M,
    rollout_plans,
    rollout_plans_with_policy,
)
from physgate.gate.schemas import Scene
from physgate.gate.scoring import PhysicsResult
from physgate.gate.trajectory import (  # noqa: F401  (re-exports for existing callers)
    PLACE_DROP_HEIGHT,
    RELEASE_VELOCITY_GAIN,
    ROBOT_MASS_KG,
    SCAN_HOLD_STEPS,
    SETTLE_STEPS,
    SKILL_HOLD_STEPS,
    PlanTrajectory,
    TrajectoryEvent,
    UnknownTargetError,
    box_on_shelf,
    compile_mission,
    count_path_collisions,
    synthesize_base_trajectory,
    yaw_to_quat,
)
from physgate.planner.schemas import Plan
from physgate.world.fetch_scene import FetchSimWorld


# ------------------------------------------------------------------ L2Fn glue


class IsaacL2Gate:
    """L2Fn-compatible callable backed by Isaac Lab parallel physics (kinematic)."""

    def __init__(self, world: FetchSimWorld):
        """Initialize with a shared Isaac Lab world for kinematic rollouts."""
        self._world = world

    def __call__(self, plans: list[Plan], scene: Scene) -> list[PhysicsResult]:
        """Run kinematic parallel physics rollouts for the given plans."""
        return rollout_plans(self._world, plans)

    __name__ = "isaac_l2"


class PolicyL2Gate:
    """L2Fn-compatible callable: parallel physics with the trained walking policy."""

    def __init__(self, world: FetchSimWorld, policy_path):
        """Initialize with a shared world and trained locomotion policy path."""
        self._world = world
        self._policy_path = policy_path

    def __call__(self, plans: list[Plan], scene: Scene) -> list[PhysicsResult]:
        """Run policy-driven parallel physics rollouts for the given plans."""
        return rollout_plans_with_policy(self._world, plans, self._policy_path)

    __name__ = "isaac_l2_policy"
