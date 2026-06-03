"""Sim-Gate Layer 2: parallel physics validation in Isaac Lab (C11).

Given N candidate plans, runs each one in its own Isaac Lab env (all envs reset
to an IDENTICAL initial state via gate/reset_workaround.py), simultaneously:

1. each plan is compiled to a base-motion trajectory + carry/release events
   (``synthesize_base_trajectory``),
2. all N robots are kinematically driven along their plan's trajectory while
   the scene objects (the box) simulate dynamically on the GPU,
3. physics outcomes are read back per env:
     - task success  = the box physically ended up resting on the shelf
       (after the placement release, dynamics decide whether it stays or
       slides/bounces off — fast, sloppy plans drop it),
     - collisions    = path segments sweeping through static obstacles,
     - time / energy = from the executed trajectory.

MVP scope note (see DECISIONS.md): the robot base is kinematically driven
(no learned locomotion policy yet); the box, contacts, and placement are
fully dynamic. Plan quality discrimination comes from real physics
(placement stability) + swept-path collision checks.

IMPORTANT: import only after SimulationApp launch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from physgate.gate.reset_workaround import reset_scene_to_identical_state
from physgate.gate.schemas import FailureCode, FailureReport, GateLayer, Scene, Violation
from physgate.gate.scoring import PhysicsResult
from physgate.planner.schemas import Plan, ToolName
from physgate.world.fetch_scene import (
    BOX_SIZE,
    CARRY_OFFSET,
    OBSTACLE_SIZE,
    ROBOT_BASE_HEIGHT,
    SCENE_LAYOUT,
    SHELF_SIZE,
    SHELF_TOP_Z,
    FetchSimWorld,
)

# tuning constants
SKILL_HOLD_STEPS = 60          # sim steps the robot pauses for a pick/place
SCAN_HOLD_STEPS = 30           # sim steps for a query_scene pause
SETTLE_STEPS = 240             # sim steps after the last plan step (2 s at 120 Hz)
RELEASE_VELOCITY_GAIN = 2.0    # released box inherits gain * last motion speed
ROBOT_COLLISION_RADIUS = 0.30  # Go2 half-width + margin for swept-path checks
PLACE_DROP_HEIGHT = 0.06       # box released this high above the shelf surface
ROBOT_MASS_KG = 15.0           # Go2 mass, for the energy proxy


# ------------------------------------------------------------------ synthesis


@dataclass
class TrajectoryEvent:
    """Carry/release event at a specific step of a base trajectory."""

    step_index: int
    kind: str                     # "attach" | "release"
    object_id: str
    release_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    release_position: tuple[float, float, float] | None = None


@dataclass
class PlanTrajectory:
    """A plan compiled to a kinematic base trajectory."""

    plan_id: str
    positions: np.ndarray         # (T, 3) base positions, env-local frame
    yaws: np.ndarray              # (T,) base headings
    events: list[TrajectoryEvent] = field(default_factory=list)
    motion_speeds: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (T,) commanded speed

    @property
    def duration_steps(self) -> int:
        return len(self.positions)


def _yaw_to_quat(yaw: float) -> np.ndarray:
    """Heading angle -> wxyz quaternion (rotation about +z)."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def synthesize_base_trajectory(
    plan: Plan,
    dt: float,
    layout: dict[str, tuple[float, float, float]] | None = None,
) -> PlanTrajectory:
    """Compile a Plan into a kinematic base trajectory + carry/release events."""
    lay = layout or SCENE_LAYOUT
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
                # unknown target: hold in place (L3 should have caught this)
                hold(SCAN_HOLD_STEPS)
                continue
            target = np.array(lay[target_id], dtype=np.float64)
            standoff = float(step.args.get("standoff_m", 0.3))
            speed = max(float(step.args.get("speed", 0.5)), 0.05)

            direction = target[:2] - current[:2]
            distance = float(np.linalg.norm(direction))
            if distance > 1e-6:
                direction = direction / distance
            else:
                direction = np.array([1.0, 0.0])
            goal_xy = target[:2] - direction * standoff
            goal = np.array([goal_xy[0], goal_xy[1], ROBOT_BASE_HEIGHT])
            yaw = math.atan2(direction[1], direction[0])

            travel = float(np.linalg.norm(goal[:2] - current[:2]))
            n_steps = max(int(travel / (speed * dt)), 1)
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
                    TrajectoryEvent(step_index=len(positions) - 1, kind="attach", object_id=target_id)
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
    """Count how many times a base path sweeps through a static obstacle.

    The base is kinematically driven, so obstacle penetration is a *geometric*
    check (a kinematic body does not generate contact responses).

    Note: manipulation targets (the shelf) are NOT obstacles — the robot must
    approach them to act on them. Interaction quality with targets is judged by
    the placement dynamics instead (box release physics).
    """
    lay = layout or SCENE_LAYOUT
    obstacles = {"obstacle_P": OBSTACLE_SIZE}
    collisions = 0
    for obstacle_id, size in obstacles.items():
        center = np.array(lay[obstacle_id][:2])
        half = np.array(size[:2]) / 2 + robot_radius
        inside = np.all(np.abs(positions[:, :2] - center) < half, axis=1)
        # count entries (rising edges) into the collision volume
        entries = np.sum(inside[1:] & ~inside[:-1]) + int(inside[0])
        collisions += int(entries)
    return collisions


def _box_on_shelf(box_pos: np.ndarray, layout: dict | None = None, tol_xy: float = 0.05) -> bool:
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


# -------------------------------------------------------------------- rollout


def rollout_plans(world: FetchSimWorld, plans: list[Plan]) -> list[PhysicsResult]:
    """Run N plans in N parallel Isaac Lab envs from an identical initial state."""
    if len(plans) > world.num_envs:
        raise ValueError(f"{len(plans)} plans but only {world.num_envs} parallel envs")

    device = world.device
    num_active = len(plans)

    # 1. identical initial state across every env (bug #2133 workaround)
    reset_scene_to_identical_state(world.scene, world.sim)

    # 2. compile plans to trajectories
    trajectories = [synthesize_base_trajectory(p, world.dt) for p in plans]
    max_steps = max(t.duration_steps for t in trajectories)

    # pad: robots that finish early hold their final pose
    def pose_at(traj: PlanTrajectory, idx: int) -> tuple[np.ndarray, float]:
        i = min(idx, traj.duration_steps - 1)
        return traj.positions[i], traj.yaws[i]

    # event lookup per env: step_index -> [events]
    event_map: list[dict[int, list[TrajectoryEvent]]] = []
    for traj in trajectories:
        m: dict[int, list[TrajectoryEvent]] = {}
        for ev in traj.events:
            m.setdefault(ev.step_index, []).append(ev)
        event_map.append(m)

    env_origins = world.env_origins.cpu().numpy()
    carrying = [False] * num_active

    # 3. step all envs in lockstep
    for step_idx in range(max_steps):
        robot_pos = np.zeros((world.num_envs, 3))
        robot_quat = np.zeros((world.num_envs, 4))
        robot_quat[:, 0] = 1.0

        for env_idx in range(world.num_envs):
            if env_idx < num_active:
                pos, yaw = pose_at(trajectories[env_idx], step_idx)
                robot_pos[env_idx] = pos + env_origins[env_idx]
                robot_quat[env_idx] = _yaw_to_quat(yaw)
            else:
                # inactive envs hold the spawn pose
                robot_pos[env_idx] = np.array(SCENE_LAYOUT["go2"]) + env_origins[env_idx]

        world.write_robot_poses(
            torch.tensor(robot_pos, dtype=torch.float32, device=device),
            torch.tensor(robot_quat, dtype=torch.float32, device=device),
        )

        # carried boxes ride with their robot; releases hand the box to physics
        carry_ids, carry_positions = [], []
        release_ids, release_positions, release_velocities = [], [], []
        for env_idx in range(num_active):
            for ev in event_map[env_idx].get(step_idx, []):
                if ev.kind == "attach":
                    carrying[env_idx] = True
                elif ev.kind == "release":
                    carrying[env_idx] = False
                    release_ids.append(env_idx)
                    release_positions.append(np.array(ev.release_position) + env_origins[env_idx])
                    release_velocities.append(np.array(ev.release_velocity))
            if carrying[env_idx]:
                pos, yaw = pose_at(trajectories[env_idx], step_idx)
                forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
                carry_pos = (
                    pos
                    + forward * CARRY_OFFSET[0]
                    + np.array([0.0, 0.0, CARRY_OFFSET[2]])
                    + env_origins[env_idx]
                )
                carry_ids.append(env_idx)
                carry_positions.append(carry_pos)

        if carry_ids:
            world.write_box_poses(
                torch.tensor(np.array(carry_positions), dtype=torch.float32, device=device),
                env_ids=torch.tensor(carry_ids, dtype=torch.long, device=device),
            )
        if release_ids:
            # teleport box to the release point, then give it the release velocity
            world.write_box_poses(
                torch.tensor(np.array(release_positions), dtype=torch.float32, device=device),
                env_ids=torch.tensor(release_ids, dtype=torch.long, device=device),
            )
            world.release_boxes(
                torch.tensor(np.array(release_velocities), dtype=torch.float32, device=device),
                env_ids=torch.tensor(release_ids, dtype=torch.long, device=device),
            )

        world.step()

    # 4. read back physics outcomes
    final_box_positions = world.box_positions().cpu().numpy()

    results: list[PhysicsResult] = []
    for env_idx, (plan, traj) in enumerate(zip(plans, trajectories)):
        placed = any(ev.kind == "release" for ev in traj.events)
        success = placed and _box_on_shelf(final_box_positions[env_idx])
        collisions = count_path_collisions(traj.positions)

        # time: trajectory length minus the settle tail
        active_steps = traj.duration_steps - SETTLE_STEPS
        completion_time = max(active_steps, 0) * world.dt
        # energy proxy: 1/2 m v^2 integrated over motion steps
        energy = float(0.5 * ROBOT_MASS_KG * np.sum(traj.motion_speeds**2) * world.dt)

        failure: FailureReport | None = None
        if not success:
            box_pos = final_box_positions[env_idx]
            detail = (
                f"box ended at ({box_pos[0]:.2f}, {box_pos[1]:.2f}, {box_pos[2]:.2f}); "
                f"shelf surface is at z={SHELF_TOP_Z:.2f}"
            )
            failure = FailureReport(
                failure_code=FailureCode.GRASP_FAILURE if placed else FailureCode.PRECONDITION_VIOLATION,
                layer=GateLayer.PHYSICS,
                violations=[
                    Violation(
                        type="placement_failed" if placed else "task_incomplete",
                        detail=detail if placed else "plan never placed the box",
                    )
                ],
                remediation_hint="approach the shelf more slowly before placing",
            )

        results.append(
            PhysicsResult(
                plan_id=plan.plan_id,
                success=success,
                collision_count=collisions,
                completion_time_s=round(completion_time, 2),
                energy_j=round(energy, 2),
                failure=failure,
            )
        )
    return results


# ------------------------------------------------------------------ L2Fn glue


class IsaacL2Gate:
    """L2Fn-compatible callable backed by Isaac Lab parallel physics."""

    def __init__(self, world: FetchSimWorld):
        self._world = world

    def __call__(self, plans: list[Plan], scene: Scene) -> list[PhysicsResult]:
        return rollout_plans(self._world, plans)

    # keep a friendly name for the demo report
    __name__ = "isaac_l2"
