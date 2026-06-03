"""Tests for plan -> trajectory/mission compilation with deterministic navigation.

REBUILD.md Phase 1: the straight-line driver is removed. Every move_to_pose step
is expanded through the A* path planner, so compiled trajectories and missions
ALWAYS route around obstacles — feasibility no longer depends on the LLM naming
a magic waypoint.

Pure logic — no GPU, no Isaac (gate/trajectory.py must not import Isaac).
"""

from __future__ import annotations

import numpy as np

from physgate.gate.trajectory import (
    UnknownTargetError,
    box_on_shelf,
    compile_mission,
    count_path_collisions,
    synthesize_base_trajectory,
)
from physgate.planner.planner import MockPlanner
from physgate.planner.schemas import Plan, PlanStep, ToolName
from physgate.world.layout import SCENE_LAYOUT

import pytest

TASK = "put the fallen box back on shelf A"
DT = 0.005


def _direct_plan(plan_id: str = "direct", speed: float = 0.5) -> Plan:
    """Move to box -> pick -> move to shelf -> place. The straight box->shelf
    line passes through the pillar; navigation must route around it."""
    return Plan(
        plan_id=plan_id,
        task=TASK,
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.MOVE_TO_POSE,
                args={"target": "box_03", "standoff_m": 0.3, "speed": speed},
                preconditions=["box_03 exists"],
            ),
            PlanStep(
                step_id=2,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "pick", "target": "box_03"},
                preconditions=["box_03 exists", "gripper_empty"],
            ),
            PlanStep(
                step_id=3,
                tool=ToolName.MOVE_TO_POSE,
                args={"target": "shelf_A", "standoff_m": 0.4, "speed": speed},
                preconditions=["shelf_A exists"],
            ),
            PlanStep(
                step_id=4,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "place", "target": "shelf_A"},
                preconditions=[],
            ),
        ],
    )


# -------------------------------------------------------- kinematic trajectory


def test_trajectory_routes_around_obstacle():
    """The compiled base trajectory must NEVER sweep through the pillar — this is
    the core artifact fix (the old straight-line driver swept right through it)."""
    traj = synthesize_base_trajectory(_direct_plan(), DT)
    assert count_path_collisions(traj.positions) == 0


def test_trajectory_collision_count_was_nonzero_for_straight_line():
    """Sanity check on the test itself: a hand-built straight box->shelf sweep
    DOES collide — proving the collision check still detects bad paths."""
    box = np.array(SCENE_LAYOUT["box_03"][:2])
    shelf = np.array(SCENE_LAYOUT["shelf_A"][:2])
    n = 200
    straight = np.zeros((n, 3))
    for k in range(n):
        straight[k, :2] = box + (shelf - box) * (k / (n - 1))
    assert count_path_collisions(straight) > 0


def test_trajectory_reaches_standoff_of_targets():
    from physgate.world.layout import SHELF_SIZE

    traj = synthesize_base_trajectory(_direct_plan(), DT)
    shelf = np.array(SCENE_LAYOUT["shelf_A"][:2])
    shelf_half = np.array(SHELF_SIZE[:2]) / 2
    # the final pose stands OUTSIDE the shelf footprint, within placing reach of
    # its edge (standoff is measured from the footprint edge since the rebuild)
    final_xy = traj.positions[-1][:2]
    edge_clearance = float((np.abs(final_xy - shelf) - shelf_half).max())
    assert 0.2 <= edge_clearance <= 1.0


def test_trajectory_has_pick_and_place_events():
    traj = synthesize_base_trajectory(_direct_plan(), DT)
    kinds = [e.kind for e in traj.events]
    assert kinds == ["attach", "release"]


def test_trajectory_deterministic():
    a = synthesize_base_trajectory(_direct_plan(), DT)
    b = synthesize_base_trajectory(_direct_plan(), DT)
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.yaws, b.yaws)


def test_all_mock_planner_trajectories_are_collision_free():
    """With navigation in the right layer, EVERY mock candidate routes cleanly —
    route feasibility is no longer a property of the plan."""
    from physgate.examples_lib.fetch_and_place import build_demo_scene

    plans = MockPlanner()(TASK, build_demo_scene(), 8, None)
    for plan in plans:
        traj = synthesize_base_trajectory(plan, DT)
        assert count_path_collisions(traj.positions) == 0, (
            f"{plan.plan_id} trajectory sweeps through an obstacle"
        )


# ---------------------------------------------------------------- missions


def test_mission_expands_goto_into_waypoint_sequence():
    """A single move_to_pose across the obstacle becomes MULTIPLE goto waypoints
    (the A* route), not one straight-line goal."""
    mission = compile_mission(_direct_plan(), SCENE_LAYOUT)
    goto_entries = [m for m in mission if m[0] == "goto"]
    # leg 1 (go2 -> box) is clear: 1 waypoint; leg 2 (box -> shelf) detours: >= 2
    assert len(goto_entries) >= 3


def test_mission_waypoints_clear_obstacle():
    from physgate.world.layout import OBSTACLE_SIZE

    mission = compile_mission(_direct_plan(), SCENE_LAYOUT)
    center = np.array(SCENE_LAYOUT["obstacle_P"][:2])
    half = np.array(OBSTACLE_SIZE[:2]) / 2 + 0.30  # robot radius
    for kind, payload, _speed in mission:
        if kind != "goto":
            continue
        assert not bool(np.all(np.abs(np.asarray(payload) - center) < half)), (
            f"mission waypoint {payload} is inside the inflated obstacle"
        )


def test_mission_preserves_pick_place_order():
    mission = compile_mission(_direct_plan(), SCENE_LAYOUT)
    kinds = [m[0] for m in mission if m[0] in ("pick", "place")]
    assert kinds == ["pick", "place"]


def test_mission_rejects_unknown_target():
    plan = Plan(
        plan_id="hallucinated",
        task=TASK,
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.MOVE_TO_POSE,
                args={"target": "waypoint_imaginary"},
                preconditions=[],
            )
        ],
    )
    with pytest.raises(UnknownTargetError):
        compile_mission(plan, SCENE_LAYOUT)


# ------------------------------------------------------------------- helpers


def test_box_on_shelf_geometry():
    from physgate.world.layout import BOX_SIZE, SHELF_TOP_Z

    shelf = SCENE_LAYOUT["shelf_A"]
    on_shelf = np.array([shelf[0], shelf[1], SHELF_TOP_Z + BOX_SIZE[2] / 2])
    on_floor = np.array([shelf[0], shelf[1], BOX_SIZE[2] / 2])
    assert box_on_shelf(on_shelf)
    assert not box_on_shelf(on_floor)
