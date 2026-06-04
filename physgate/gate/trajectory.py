"""Plan -> trajectory/mission compilation with deterministic navigation (pure logic).

REBUILD.md Phase 1: every ``move_to_pose`` step is expanded through the A* path
planner (``physgate/nav``), so compiled base trajectories and locomotion
missions ALWAYS route around obstacles. The straight-line driver is gone —
route feasibility is a property of the navigation layer, never of the LLM plan.

This module is pure (numpy + nav only, no Isaac import) so it is unit-testable
in the pure-logic venv. ``gate/l2_physics.py`` consumes it for GPU rollouts and
``executor/sim_backend.py`` for execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from physgate.nav.path_planner import (
    WAYPOINT_TRACKING_TOLERANCE,
    Obstacle,
    plan_standoff_route,
)
from physgate.planner.schemas import Plan, ToolName
from physgate.world.layout import (
    BOX_SIZE,
    ROBOT_BASE_HEIGHT,
    ROBOT_COLLISION_RADIUS,
    SCENE_LAYOUT,
    SHELF_SIZE,
    SHELF_TOP_Z,
    STATIC_FOOTPRINTS,
    navigation_obstacles,
    target_half_extents,
)

# tuning constants
SKILL_HOLD_STEPS = 60  # sim steps the robot pauses for a pick/place
SCAN_HOLD_STEPS = 30  # sim steps for a query_scene pause
SETTLE_STEPS = 240  # sim steps after the last plan step (2 s at 120 Hz)
RELEASE_VELOCITY_GAIN = 2.0  # released box inherits gain * last motion speed
PLACE_DROP_HEIGHT = 0.06  # box released this high above the shelf surface
ROBOT_MASS_KG = 15.0  # Go2 mass, for the energy proxy


class UnknownTargetError(ValueError):
    """A plan references an object id with no physical location in the scene layout.

    L3 (:func:`physgate.gate.l3_scene.check_step_targets`) should reject such
    plans before they reach L2; this exception is the defense-in-depth backstop
    for direct L2 callers. Silently skipping unknown targets is NOT acceptable:
    it degrades plans into do-nothing missions that fail with misleading
    "box not on shelf" reports (DECISIONS.md D-016).
    """


# ------------------------------------------------------------------ synthesis


@dataclass
class TrajectoryEvent:
    """Carry/release event at a specific step of a base trajectory."""

    step_index: int
    kind: str  # "attach" | "release"
    object_id: str
    release_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    release_position: tuple[float, float, float] | None = None


@dataclass
class PlanTrajectory:
    """A plan compiled to a kinematic base trajectory."""

    plan_id: str
    positions: np.ndarray  # (T, 3) base positions, env-local frame
    yaws: np.ndarray  # (T,) base headings
    events: list[TrajectoryEvent] = field(default_factory=list)
    motion_speeds: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (T,) commanded speed

    @property
    def duration_steps(self) -> int:
        """Return the total number of trajectory steps."""
        return len(self.positions)


def yaw_to_quat(yaw: float) -> np.ndarray:
    """Heading angle -> wxyz quaternion (rotation about +z)."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _route_for_step(
    current_xy: np.ndarray,
    target_id: str,
    standoff: float,
    lay: dict[str, tuple[float, float, float]],
    obstacles: list[Obstacle],
    tracking_error: float = 0.0,
) -> list[tuple[float, float]]:
    """Obstacle-avoiding waypoint route from the current position to a standoff
    pose near the target object (the navigation layer's contract).

    Args:
        tracking_error: extra clearance for imperfect path followers — the
            walking policy cuts corners by up to its waypoint arrival tolerance,
            so its routes are planned wider (kinematic followers pass 0).
    """
    target_xy = (lay[target_id][0], lay[target_id][1])
    route = plan_standoff_route(
        (float(current_xy[0]), float(current_xy[1])),
        target_xy,
        standoff=standoff,
        obstacles=obstacles,
        robot_radius=ROBOT_COLLISION_RADIUS,
        tracking_error=tracking_error,
        # footprint targets (shelf): standoff measured from the footprint edge,
        # on whichever face is reachable; point targets (box): from the center
        target_half_extents=target_half_extents(target_id),
    )
    # drop the start point — the robot is already there
    return route[1:] if len(route) > 1 else []


def synthesize_base_trajectory(
    plan: Plan,
    dt: float,
    layout: dict[str, tuple[float, float, float]] | None = None,
) -> PlanTrajectory:
    """Compile a Plan into a kinematic base trajectory + carry/release events.

    Every move_to_pose step is routed through the deterministic path planner —
    the trajectory NEVER sweeps through a navigation obstacle.
    """
    lay = layout or SCENE_LAYOUT
    obstacles = navigation_obstacles(lay)
    positions: list[np.ndarray] = []
    yaws: list[float] = []
    speeds: list[float] = []
    events: list[TrajectoryEvent] = []

    current = np.array(lay["go2"], dtype=np.float64)
    current[2] = ROBOT_BASE_HEIGHT
    yaw = 0.0
    held_object: str | None = None
    last_speed = 0.0

    def hold(steps: int) -> None:
        """Append stationary frames at the current position for the given step count."""
        for _ in range(steps):
            positions.append(current.copy())
            yaws.append(yaw)
            speeds.append(0.0)

    for step in plan.steps:
        if step.tool == ToolName.QUERY_SCENE:
            hold(SCAN_HOLD_STEPS)

        elif step.tool == ToolName.MOVE_TO_POSE:
            target_id = step.args["target"]
            if target_id not in lay:
                raise UnknownTargetError(
                    f"plan '{plan.plan_id}' step {step.step_id} moves to unknown "
                    f"object '{target_id}' (known: {sorted(lay.keys())})"
                )
            standoff = float(step.args.get("standoff_m", 0.3))
            speed = max(float(step.args.get("speed", 0.5)), 0.05)

            route = _route_for_step(current[:2], target_id, standoff, lay, obstacles)
            for waypoint in route:
                goal = np.array([waypoint[0], waypoint[1], ROBOT_BASE_HEIGHT])
                direction = goal[:2] - current[:2]
                seg_len = float(np.linalg.norm(direction))
                if seg_len < 1e-9:
                    continue
                yaw = math.atan2(direction[1], direction[0])
                n_steps = max(int(seg_len / (speed * dt)), 1)
                for k in range(1, n_steps + 1):
                    positions.append(current + (goal - current) * (k / n_steps))
                    yaws.append(yaw)
                    speeds.append(speed)
                current = goal
            last_speed = speed

        elif step.tool == ToolName.EXECUTE_SKILL:
            skill = step.args.get("skill")
            target_id = step.args.get("target", "")
            hold(SKILL_HOLD_STEPS)
            if skill == "pick":
                events.append(
                    TrajectoryEvent(
                        step_index=len(positions) - 1, kind="attach", object_id=target_id
                    )
                )
                held_object = target_id
            elif skill == "place":
                # release the carried box above the shelf surface, inheriting
                # momentum from the approach speed — physics decides if it stays
                shelf = np.array(lay.get(target_id, lay["shelf_A"]), dtype=np.float64)
                release_pos = (
                    float(shelf[0]),
                    float(shelf[1]),
                    SHELF_TOP_Z + BOX_SIZE[2] / 2 + PLACE_DROP_HEIGHT,
                )
                forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
                release_vel = forward * last_speed * RELEASE_VELOCITY_GAIN
                events.append(
                    TrajectoryEvent(
                        step_index=len(positions) - 1,
                        kind="release",
                        object_id=held_object or target_id,
                        release_velocity=tuple(release_vel.tolist()),
                        release_position=release_pos,
                    )
                )
                held_object = None

    # tail settle so released objects come to rest before readback
    hold(SETTLE_STEPS)

    return PlanTrajectory(
        plan_id=plan.plan_id,
        positions=np.array(positions),
        yaws=np.array(yaws),
        events=events,
        motion_speeds=np.array(speeds),
    )


# ----------------------------------------------------------- geometric checks


def count_path_collisions(
    positions: np.ndarray,
    layout: dict[str, tuple[float, float, float]] | None = None,
    robot_radius: float = ROBOT_COLLISION_RADIUS,
) -> int:
    """Count how many times a base path sweeps through static scene geometry.

    The base is kinematically driven, so obstacle penetration is a *geometric*
    check (a kinematic body does not generate contact responses).

    Static geometry includes the obstacle pillar AND large manipulation targets
    (the shelf): the robot must stop at a standoff outside their footprint, never
    drive through them. Small graspable objects (the box) are not counted.
    """
    lay = layout or SCENE_LAYOUT
    collisions = 0
    for obstacle_id, size in STATIC_FOOTPRINTS.items():
        if obstacle_id not in lay:
            continue
        center = np.array(lay[obstacle_id][:2])
        half = np.array(size[:2]) / 2 + robot_radius
        inside = np.all(np.abs(positions[:, :2] - center) < half, axis=1)
        # count entries (rising edges) into the collision volume
        entries = np.sum(inside[1:] & ~inside[:-1]) + int(inside[0])
        collisions += int(entries)
    return collisions


def box_on_shelf(box_pos: np.ndarray, layout: dict | None = None, tol_xy: float = 0.05) -> bool:
    """Did the box physically end up resting on the shelf surface?"""
    lay = layout or SCENE_LAYOUT
    shelf_center = np.array(lay["shelf_A"])
    half_x, half_y = SHELF_SIZE[0] / 2 + tol_xy, SHELF_SIZE[1] / 2 + tol_xy
    expected_z = SHELF_TOP_Z + BOX_SIZE[2] / 2
    return (
        abs(box_pos[0] - shelf_center[0]) <= half_x
        and abs(box_pos[1] - shelf_center[1]) <= half_y
        and abs(box_pos[2] - expected_z) <= 0.08
    )


# ------------------------------------------------------------------- missions


def compile_mission(
    plan: Plan, lay: dict[str, tuple[float, float, float]]
) -> list[tuple[str, object, float]]:
    """Compile a Plan into a waypoint mission for the walking robot.

    Mission entries: ("goto", goal_xy_position, speed) | ("pick", object_id, 0)
    | ("place", object_id, 0) | ("wait", duration_steps, 0).

    Every move_to_pose step expands into the FULL waypoint sequence of an
    obstacle-avoiding route (one "goto" per waypoint) — the walking robot
    follows the navigation layer's path, never a straight line.

    Raises:
        UnknownTargetError: a move_to_pose step targets an id not in ``lay``.
    """
    obstacles = navigation_obstacles(lay)
    mission: list[tuple[str, object, float]] = []
    current = np.array(lay["go2"][:2], dtype=np.float64)

    for step in plan.steps:
        if step.tool == ToolName.QUERY_SCENE:
            mission.append(("wait", 25, 0.0))  # ~0.5 s at 50 Hz control
        elif step.tool == ToolName.MOVE_TO_POSE:
            target_id = step.args.get("target")
            if target_id not in lay:
                raise UnknownTargetError(
                    f"plan '{plan.plan_id}' step {step.step_id} moves to unknown "
                    f"object '{target_id}' (known: {sorted(lay.keys())})"
                )
            standoff = float(step.args.get("standoff_m", 0.3))
            # clamp commanded speed to the envelope the locomotion stack
            # reliably executes (measured, DECISIONS D-018): below 0.4 m/s the
            # policy creeps/stalls; above ~0.6 m/s the policy + waypoint
            # follower combination intermittently collapses into a crouch-stall.
            # The plan's "speed preference" is respected within that envelope.
            speed = float(np.clip(float(step.args.get("speed", 0.5)), 0.4, 0.6))

            # the walking policy follows waypoints loosely (arrival tolerance):
            # plan its routes with matching extra clearance so corner-cutting
            # can never push the robot into the obstacle
            route = _route_for_step(
                current,
                target_id,
                standoff,
                lay,
                obstacles,
                tracking_error=WAYPOINT_TRACKING_TOLERANCE,
            )
            for waypoint in route:
                goal = np.array(waypoint, dtype=np.float64)
                mission.append(("goto", goal, speed))
                current = goal
        elif step.tool == ToolName.EXECUTE_SKILL:
            skill = step.args.get("skill")
            target_id = step.args.get("target", "")
            if skill == "pick":
                mission.append(("pick", target_id, 0.0))
            elif skill == "place":
                mission.append(("place", target_id, 0.0))
    return mission
